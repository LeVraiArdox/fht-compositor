use std::rc::Rc;
use std::time::Duration;

use fht_compositor_config::{
    GestureAction, GestureDirection, WorkspaceSwitchAnimation, WorkspaceSwitchAnimationDirection,
};

use smithay::backend::renderer::element::utils::{Relocate, RelocateRenderElement};
use smithay::output::Output;
use smithay::utils::{IsAlive, Point, Rectangle};
use smithay::wayland::compositor::with_states;
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::shell::xdg::SurfaceCachedState;

use super::workspace::{Workspace, WorkspaceRenderElement};
use super::Config;
use crate::fht_render_elements;
use crate::output::OutputExt;
use crate::renderer::FhtRenderer;
use crate::window::Window;
use crate::input::resize_tile_grab::ResizeEdge;
use crate::space::tile::{Tile, TileRenderElement};
use crate::space::workspace::InteractiveResize;

const WORKSPACE_COUNT: usize = 9;

/// Swipe gesture state for a monitor
#[derive(Debug)]
pub struct MonitorSwipeState {
    pub direction: Option<GestureDirection>,
    pub total_offset: Point<f64, smithay::utils::Logical>,
    pub swipe_distance: f64,
    pub cancel_ratio: f64,
    pub min_speed_to_force: f64,
    pub direction_detection_threshold: f64,
    pub initial_workspace_idx: usize,
    pub last_update_time: Duration,
    pub current_velocity: f64,
    pub animation_direction: WorkspaceSwitchAnimationDirection,
}

pub struct Monitor {
    /// The output associated with the monitor.
    output: Output,
    /// The associated workspaces with the monitor.
    pub workspaces: [Workspace; WORKSPACE_COUNT],
    /// The active workspace index.
    pub active_idx: usize,
    /// Whether this monitor is the focused monitor.
    is_active: bool,
    /// Shared configuration with across the workspace system.
    pub config: Rc<Config>,
    /// Swipe gesture state for the current swipe, if applicable
    pub swipe_state: Option<MonitorSwipeState>,
    /// Pinned windows on this monitor
    pub pinned_tiles: Vec<Tile>,
    /// Active interactive resize for a pinned tile, if any
    interactive_resize: Option<InteractiveResize>,
}

pub struct MonitorRenderResult<R: FhtRenderer> {
    /// The elements rendered from this result.
    pub elements: Vec<MonitorRenderElement<R>>,
    /// The elements to be rendered above the top layer.
    pub elements_above_top: Vec<MonitorRenderElement<R>>,
}

fht_render_elements! {
    MonitorRenderElement<R> => {
        Workspace = WorkspaceRenderElement<R>,
        SwitchingWorkspace = RelocateRenderElement<WorkspaceRenderElement<R>>,
        Pinned = TileRenderElement<R>,
    }
}

impl Monitor {
    /// Create a new [`Monitor`].
    pub fn new(output: Output, config: Rc<Config>) -> Self {
        let workspaces = (0..WORKSPACE_COUNT)
            .map(|index| {
                let output = output.clone();
                Workspace::new(output, index, &config)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            output,
            workspaces,
            active_idx: 0,
            is_active: false,
            config,
            swipe_state: None,
            pinned_tiles: Vec::new(),
            interactive_resize: None,
        }
    }

    /// Run periodic update/clean-up tasks
    pub fn refresh(&mut self, is_active: bool) {
        crate::profile_function!();
        self.is_active = is_active;
        for workspace in &mut self.workspaces {
            workspace.refresh();
        }

        // Refresh pinned tiles
        self.pinned_tiles.retain(|tile| tile.window().alive());
        for tile in &mut self.pinned_tiles {
            // We pass 'false' for active here because focus is handled by State::update_keyboard_focus
            // However, we ensure they are entered on the output
            tile.refresh(&self.output, false);
        }
    }

    /// Merge this [`Monitor`] with another one.
    pub fn merge_with(&mut self, other: Self) {
        assert!(
            self.output != other.output,
            "tried to merge a monitor with itself!"
        );

        for (workspace, other_workspace) in self.workspaces_mut().zip(other.workspaces) {
            workspace.merge_with(other_workspace)
        }
    }

    /// Get a reference to this [`Monitor`]'s associated [`Output`].
    pub fn output(&self) -> &Output {
        &self.output
    }

    /// Get whether this monitor is active
    pub fn active(&self) -> bool {
        self.is_active
    }

    /// Get an iterator over the monitor's [`Workspace`]s.
    pub fn workspaces(&self) -> impl ExactSizeIterator<Item = &Workspace> {
        self.workspaces.iter()
    }

    /// Get a mutable iterator over the monitor's [`Workspace`]s.
    pub fn workspaces_mut(&mut self) -> impl ExactSizeIterator<Item = &mut Workspace> {
        self.workspaces.iter_mut()
    }

    /// Get a reference to a workspace [`Workspace`] from this [`Monitor`] by index.
    pub fn workspace_by_index(&self, index: usize) -> &Workspace {
        &self.workspaces[index]
    }

    /// Get a mutable reference to a workspace [`Workspace`] from this [`Monitor`] by index.
    pub fn workspace_mut_by_index(&mut self, index: usize) -> &mut Workspace {
        &mut self.workspaces[index]
    }

    /// Get the all the visible [`Window`] on this output.
    pub fn visible_windows(&self) -> impl ExactSizeIterator<Item = &Window> {
        self.workspaces[self.active_idx].windows()
    }

    /// Set the active [`Workspace`] index.
    pub fn set_active_workspace_idx(&mut self, idx: usize, animate: bool) -> Option<Window> {
        if self.active_idx == idx {
            return None;
        }

        // The workspace switch animation is done on a per-workspace level.
        // Each workspace has a render offset.
        if animate {
            if let Some((config, direction)) = &self.config.workspace_switch_animation {
                let (width, height) = self.output.geometry().size.into();

                // Get the current swipe offset if a swipe is active
                let current_offset = if let Some(swipe_state) = &self.swipe_state {
                    match direction {
                        WorkspaceSwitchAnimationDirection::Horizontal => {
                            Point::from((swipe_state.total_offset.x as i32, 0))
                        }
                        WorkspaceSwitchAnimationDirection::Vertical => {
                            Point::from((0, swipe_state.total_offset.y as i32))
                        }
                    }
                } else {
                    Point::default()
                };

                match direction {
                    WorkspaceSwitchAnimationDirection::Horizontal => {
                        if self.active_idx > idx {
                            self.workspaces[self.active_idx].start_render_offset_animation(
                                current_offset,
                                (width, 0).into(),
                                config,
                            );
                            self.workspaces[idx].start_render_offset_animation(
                                Point::from((-width, 0)) + current_offset,
                                Point::default(),
                                config,
                            );
                        } else {
                            self.workspaces[self.active_idx].start_render_offset_animation(
                                current_offset,
                                (-width, 0).into(),
                                config,
                            );
                            self.workspaces[idx].start_render_offset_animation(
                                Point::from((width, 0)) + current_offset,
                                Point::default(),
                                config,
                            );
                        }
                    }
                    WorkspaceSwitchAnimationDirection::Vertical => {
                        if self.active_idx > idx {
                            self.workspaces[self.active_idx].start_render_offset_animation(
                                current_offset,
                                (0, height).into(),
                                config,
                            );
                            self.workspaces[idx].start_render_offset_animation(
                                Point::from((0, -height)) + current_offset,
                                Point::default(),
                                config,
                            );
                        } else {
                            self.workspaces[self.active_idx].start_render_offset_animation(
                                current_offset,
                                (0, -height).into(),
                                config,
                            );
                            self.workspaces[idx].start_render_offset_animation(
                                Point::from((0, height)) + current_offset,
                                Point::default(),
                                config,
                            );
                        }
                    }
                };
                self.swipe_state.take();
            }
        }

        self.active_idx = idx;
        self.workspaces[self.active_idx].active_window()
    }

    /// Get a reference to the active [`Workspace`].
    pub fn active_workspace(&self) -> &Workspace {
        &self.workspaces[self.active_idx]
    }

    /// Get a the the active [`Workspace`] index.
    pub fn active_workspace_idx(&self) -> usize {
        self.active_idx
    }

    /// Get a reference to the active [`Workspace`].
    pub fn active_workspace_mut(&mut self) -> &mut Workspace {
        &mut self.workspaces[self.active_idx]
    }

    /// Advance animations for this [`Monitor`].
    pub fn advance_animations(&mut self, target_presentation_time: Duration) -> bool {
        crate::profile_function!();
        self.workspaces.iter_mut().fold(false, |acc, ws| {
            ws.advance_animations(target_presentation_time) || acc
        })
    }

    /// Returns whether this monitor has any blurred regions.
    pub fn has_blur(&self) -> bool {
        for workspace in &self.workspaces {
            if (workspace.index() == self.active_idx || workspace.render_offset().is_some())
                && workspace.tiles().any(|tile| tile.has_transparent_region())
            {
                return true;
            }
        }

        false
    }

    /// Return whether the monitor contents should be rendered above the top layer shells
    pub fn render_above_top(&self) -> bool {
        let ws = self.active_workspace();
        ws.fullscreened_tile().is_some() && !ws.render_offset().is_some()
    }

    /// Start a swipe gesture at the monitor level
    pub fn start_swipe_gesture(&mut self, animation_config: &WorkspaceSwitchAnimation) {
        let previous_offset = Point::from((0.0, 0.0));

        self.swipe_state = Some(MonitorSwipeState {
            direction: None,
            total_offset: previous_offset,
            swipe_distance: animation_config.swipe_distance,
            cancel_ratio: animation_config.swipe_cancel_ratio,
            min_speed_to_force: animation_config.swipe_min_speed_to_force,
            direction_detection_threshold: animation_config.direction_detection_threshold,
            initial_workspace_idx: self.active_idx,
            last_update_time: Duration::ZERO,
            current_velocity: 0.0,
            animation_direction: animation_config.direction,
        });
    }

    /// Update the swipe gesture with new delta movement
    pub fn update_swipe_gesture(
        &mut self,
        delta: Point<f64, smithay::utils::Logical>,
        time: Duration,
        detected_direction: Option<GestureDirection>,
    ) {
        let Some(state) = &mut self.swipe_state else {
            return;
        };

        // Detect the swipe direction if not already set
        if state.direction.is_none() {
            if let Some(dir) = detected_direction {
                state.direction = Some(dir);
            }
        }

        // Calculate the velocity
        if state.last_update_time != Duration::ZERO {
            let time_delta = time.saturating_sub(state.last_update_time);
            if time_delta > Duration::ZERO {
                let time_delta_secs = time_delta.as_secs_f64();
                let distance = match state.animation_direction {
                    WorkspaceSwitchAnimationDirection::Horizontal => delta.x.abs(),
                    WorkspaceSwitchAnimationDirection::Vertical => delta.y.abs(),
                };
                state.current_velocity = distance / time_delta_secs;
            }
        }

        state.total_offset += delta;
        state.last_update_time = time;
    }

    /// End the swipe gesture and determine the action to take
    pub fn end_swipe_gesture(&mut self) -> Option<GestureAction> {
        let state = self.swipe_state.as_ref()?;

        let Some(direction) = state.direction else {
            self.swipe_state.take();
            return None;
        };

        let progress = match state.animation_direction {
            WorkspaceSwitchAnimationDirection::Horizontal => state.total_offset.x,
            WorkspaceSwitchAnimationDirection::Vertical => state.total_offset.y,
        };

        let abs_progress = progress.abs();
        let cancel_threshold = state.swipe_distance * state.cancel_ratio;

        let at_limit = match direction {
            GestureDirection::Left | GestureDirection::Up => {
                state.initial_workspace_idx >= WORKSPACE_COUNT - 1
            }
            GestureDirection::Right | GestureDirection::Down => state.initial_workspace_idx == 0,
            _ => false,
        };

        if at_limit {
            if let Some(state) = self.swipe_state.take() {
                self.cancel_swipe_animation(state);
            }
            return None;
        }

        let should_trigger = abs_progress >= state.swipe_distance
            || (state.current_velocity >= state.min_speed_to_force
                && abs_progress >= cancel_threshold);

        if should_trigger {
            Some(match direction {
                GestureDirection::Left | GestureDirection::Up => GestureAction::FocusNextWorkspace,
                GestureDirection::Right | GestureDirection::Down => {
                    GestureAction::FocusPreviousWorkspace
                }
                _ => {
                    self.swipe_state.take();
                    return None;
                }
            })
        } else {
            if let Some(state) = self.swipe_state.take() {
                self.cancel_swipe_animation(state);
            }
            None
        }
    }

    fn cancel_swipe_animation(&mut self, swipe_state: MonitorSwipeState) {
        if let Some((config, _)) = &self.config.workspace_switch_animation {
            let output_size = self.output.geometry().size;
            let current_offset = match swipe_state.animation_direction {
                WorkspaceSwitchAnimationDirection::Horizontal => {
                    Point::from((swipe_state.total_offset.x as i32, 0))
                }
                WorkspaceSwitchAnimationDirection::Vertical => {
                    Point::from((0, swipe_state.total_offset.y as i32))
                }
            };

            // Animate the current workspace back to the center
            self.workspaces[self.active_idx].start_render_offset_animation(
                current_offset,
                Point::default(),
                config,
            );

            // Animate the adjacent workspace back to its original position off-screen
            let (adjacent_idx, adjacent_base_offset) = match swipe_state.direction {
                Some(GestureDirection::Left) | Some(GestureDirection::Up) => {
                    if self.active_idx < WORKSPACE_COUNT - 1 {
                        let offset = match swipe_state.animation_direction {
                            WorkspaceSwitchAnimationDirection::Horizontal => {
                                Point::from((output_size.w, 0))
                            }
                            WorkspaceSwitchAnimationDirection::Vertical => {
                                Point::from((0, output_size.h))
                            }
                        };
                        (Some(self.active_idx + 1), offset)
                    } else {
                        (None, Point::default())
                    }
                }
                Some(GestureDirection::Right) | Some(GestureDirection::Down) => {
                    if self.active_idx > 0 {
                        let offset = match swipe_state.animation_direction {
                            WorkspaceSwitchAnimationDirection::Horizontal => {
                                Point::from((-output_size.w, 0))
                            }
                            WorkspaceSwitchAnimationDirection::Vertical => {
                                Point::from((0, -output_size.h))
                            }
                        };
                        (Some(self.active_idx - 1), offset)
                    } else {
                        (None, Point::default())
                    }
                }
                _ => (None, Point::default()),
            };

            if let Some(idx) = adjacent_idx {
                self.workspaces[idx].start_render_offset_animation(
                    adjacent_base_offset + current_offset,
                    adjacent_base_offset,
                    config,
                );
            }
        }
    }

    /// Set whether a window should be pinned on this monitor
    pub fn set_window_pinned(&mut self, window: &Window, pinned: bool) {
        let is_currently_pinned = self.pinned_tiles.iter().any(|t| t.window() == window);

        if pinned && !is_currently_pinned {
            if let Some(tile) = self.workspaces[self.active_idx].detach_window(window) {
                window.request_tiled(false);
                self.pinned_tiles.push(tile);
            }
        } else if !pinned && is_currently_pinned {
            // Unpin: remove from pinned list and insert into the active workspace
            if let Some(idx) = self.pinned_tiles.iter().position(|t| t.window() == window) {
                let tile = self.pinned_tiles.remove(idx);
                self.workspaces[self.active_idx].insert_tile(tile, None);
            }
        }
    }

    /// Get the pinned window under a given point, if any
    pub fn pinned_window_under(&self, point: Point<f64, smithay::utils::Logical>) -> Option<(Window, Point<i32, smithay::utils::Logical>)> {
        self.pinned_tiles.iter().rev().find_map(|tile| {
            let loc = tile.location() + tile.window_loc();
            let tile_geo = smithay::utils::Rectangle::new(loc, tile.window().size());

            if !tile_geo.to_f64().contains(point) {
                return None;
            }

            tile.window().surface_under(point - loc.to_f64(), smithay::desktop::WindowSurfaceType::ALL)
                .map(|(_, surf_loc)| (tile.window().clone(), loc + surf_loc))
        })
    }

    pub fn start_interactive_resize(&mut self, window: &Window, edges: ResizeEdge) -> bool {
        if self.interactive_resize.is_none() {
            if let Some(tile) = self.pinned_tiles.iter().find(|t| t.window() == window) {
                let loc = tile.visual_location();
                let size = window.size();
                self.interactive_resize = Some(InteractiveResize {
                    window: window.clone(),
                    initial_window_geometry: Rectangle::new(loc, size),
                    edges,
                });
                return true;
            }
        }

        // Fallback to workspaces
        for workspace in &mut self.workspaces {
            if workspace.start_interactive_resize(window, edges) {
                return true;
            }
        }

        false
    }

    pub fn handle_interactive_resize_motion(
        &mut self,
        window: &Window,
        delta: Point<i32, smithay::utils::Logical>,
    ) -> bool {
        if let Some(interactive_resize) = &self.interactive_resize {
            if interactive_resize.window == *window {
                let mut new_size = interactive_resize.initial_window_geometry.size;
                let mut new_loc = interactive_resize.initial_window_geometry.loc;
                let (mut dx, mut dy) = (delta.x, delta.y);

                if interactive_resize.edges.intersects(ResizeEdge::LEFT) {
                    new_loc.x += dx;
                    dx = -dx;
                }
                if interactive_resize.edges.intersects(ResizeEdge::TOP) {
                    new_loc.y += dy;
                    dy = -dy;
                }
                if interactive_resize
                    .edges
                    .intersects(ResizeEdge::LEFT | ResizeEdge::RIGHT)
                {
                    new_size.w += dx;
                }
                if interactive_resize
                    .edges
                    .intersects(ResizeEdge::TOP | ResizeEdge::BOTTOM)
                {
                    new_size.h += dy;
                }

                let (min_size, max_size) =
                    with_states(&window.wl_surface().unwrap(), |data| {
                        let mut cached = data.cached_state.get::<SurfaceCachedState>();
                        let d = cached.current();
                        (d.min_size, d.max_size)
                    });
                new_size = crate::space::workspace::clamp_size(new_size, min_size, max_size);

                window.request_size(new_size);
                window.send_configure();

                if let Some(tile) = self.pinned_tiles.iter_mut().find(|t| t.window() == window) {
                    tile.set_location(new_loc, false);
                }
                return true;
            }
        }

        // Fallback to workspaces
        for workspace in &mut self.workspaces {
            if workspace.handle_interactive_resize_motion(window, delta) {
                return true;
            }
        }

        false
    }

    pub fn handle_interactive_resize_end(
        &mut self,
        window: &Window,
        position: Point<f64, smithay::utils::Logical>,
    ) -> bool {
        if let Some(ir) = &self.interactive_resize {
            if ir.window == *window {
                self.interactive_resize = None;
                return true;
            }
        }

        // Fallback to workspaces
        let position_in_workspace = position - self.output.current_location().to_f64();
        for workspace in &mut self.workspaces {
            if workspace.handle_interactive_resize_end(window, position_in_workspace) {
                return true;
            }
        }

        false
    }

    /// Create the render elements for this [`Monitor`]
    pub fn render<R: FhtRenderer>(&self, renderer: &mut R, scale: i32) -> MonitorRenderResult<R> {
        crate::profile_function!();
        let mut result = if let Some(swipe_state) = &self.swipe_state {
            self.render_with_swipe(renderer, scale, swipe_state)
        } else {
            let mut elements = vec![];
            let mut elements_above_top = vec![];
            for (idx, workspace) in self.workspaces.iter().enumerate() {
                if idx == self.active_idx || workspace.render_offset().is_some() {
                    let render_above_top = workspace.fullscreened_tile_idx().is_some();
                    let target = if render_above_top {
                        &mut elements_above_top
                    } else {
                        &mut elements
                    };

                    let render_offset = workspace.render_offset();
                    let ws_elements = workspace
                        .render(renderer, scale, None)
                        .into_iter()
                        .map(Into::into);

                    if let Some(offset) = render_offset {
                        let offset = offset.to_physical(scale);
                        target.extend(ws_elements.map(|e| {
                            RelocateRenderElement::from_element(
                                e,
                                offset,
                                smithay::backend::renderer::element::utils::Relocate::Relative,
                            )
                            .into()
                        }));
                    } else {
                        target.extend(ws_elements.map(Into::into));
                    }
                }
            }
            MonitorRenderResult {
                elements,
                elements_above_top,
            }
        };

        // Render pinned tiles on top of everything (elements_above_top)
        // Pinned windows should be visible even above fullscreen workspace windows
        for tile in &self.pinned_tiles {
            result.elements_above_top.extend(
                tile.render(renderer, scale, 1.0, &self.output, Point::default())
                    .map(Into::into),
            );
        }

        result
    }

    /// Render with a swipe gesture in progress
    fn render_with_swipe<R: FhtRenderer>(
        &self,
        renderer: &mut R,
        scale: i32,
        swipe_state: &MonitorSwipeState,
    ) -> MonitorRenderResult<R> {
        let mut elements = vec![];
        let mut elements_above_top = vec![];

        let output_size = self.output.geometry().size;

        // Calculate the offset of the current workspace based on the swipe progress
        let current_offset = match swipe_state.animation_direction {
            WorkspaceSwitchAnimationDirection::Horizontal => {
                Point::from((swipe_state.total_offset.x as i32, 0))
            }
            WorkspaceSwitchAnimationDirection::Vertical => {
                Point::from((0, swipe_state.total_offset.y as i32))
            }
        };

        // Render the current workspace with its offset
        let current_ws_elements = self.workspaces[self.active_idx]
            .render(renderer, scale, Some(current_offset))
            .into_iter();

        let current_offset_physical = current_offset.to_physical(scale);
        let current_ws_elements = current_ws_elements
            .map(|e| {
                RelocateRenderElement::from_element(
                    e.into(),
                    current_offset_physical,
                    Relocate::Relative,
                )
            })
            .map(Into::into);
        if self.workspaces[self.active_idx]
            .fullscreened_tile_idx()
            .is_some()
        {
            elements_above_top.extend(current_ws_elements);
        } else {
            elements.extend(current_ws_elements);
        }

        // DDetermine which adjacent workspace to render
        let (adjacent_idx, adjacent_base_offset) = match swipe_state.direction {
            Some(GestureDirection::Left) | Some(GestureDirection::Up) => {
                // Swipe vers la gauche/haut = workspace suivant
                if self.active_idx < WORKSPACE_COUNT - 1 {
                    let offset = match swipe_state.animation_direction {
                        WorkspaceSwitchAnimationDirection::Horizontal => {
                            Point::from((output_size.w, 0))
                        }
                        WorkspaceSwitchAnimationDirection::Vertical => {
                            Point::from((0, output_size.h))
                        }
                    };
                    (Some(self.active_idx + 1), offset)
                } else {
                    (None, Point::default())
                }
            }
            Some(GestureDirection::Right) | Some(GestureDirection::Down) => {
                // Swipe to the right/down = previous workspace
                if self.active_idx > 0 {
                    let offset = match swipe_state.animation_direction {
                        WorkspaceSwitchAnimationDirection::Horizontal => {
                            Point::from((-output_size.w, 0))
                        }
                        WorkspaceSwitchAnimationDirection::Vertical => {
                            Point::from((0, -output_size.h))
                        }
                    };
                    (Some(self.active_idx - 1), offset)
                } else {
                    (None, Point::default())
                }
            }
            _ => (None, Point::default()),
        };

        // Render the adjacent workspace if it exists
        if let Some(idx) = adjacent_idx {
            let adjacent_offset = adjacent_base_offset + current_offset;
            let adjacent_ws_elements = self.workspaces[idx]
                .render(renderer, scale, Some(adjacent_offset))
                .into_iter();

            let adjacent_offset_physical = adjacent_offset.to_physical(scale);
            let adjacent_ws_elements = adjacent_ws_elements
                .map(|e| {
                    RelocateRenderElement::from_element(
                        e.into(),
                        adjacent_offset_physical,
                        Relocate::Relative,
                    )
                })
                .map(Into::into);
            if self.workspaces[idx].fullscreened_tile_idx().is_some() {
                elements_above_top.extend(adjacent_ws_elements);
            } else {
                elements.extend(adjacent_ws_elements);
            }
        }

        MonitorRenderResult {
            elements,
            elements_above_top,
        }
    }
}
