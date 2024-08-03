use std::cell::RefCell;

use smithay::delegate_xdg_shell;
use smithay::desktop::{
    find_popup_root_surface, layer_map_for_output, PopupKeyboardGrab, PopupKind, PopupPointerGrab,
    PopupUngrabStrategy, Window, WindowSurfaceType,
};
use smithay::input::pointer::Focus;
use smithay::input::Seat;
use smithay::output::Output;
use smithay::reexports::wayland_protocols::xdg::shell::server::xdg_toplevel::{
    self, WmCapabilities,
};
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::protocol::{wl_output, wl_seat};
use smithay::utils::Serial;
use smithay::wayland::compositor::{
    add_pre_commit_hook, with_states, BufferAssignment, HookId, SurfaceAttributes,
};
use smithay::wayland::seat::WaylandFocus;
use smithay::wayland::shell::xdg::{
    Configure, PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState,
    XdgToplevelSurfaceData,
};

use crate::shell::grabs::ResizeState;
use crate::shell::workspaces::tile::WorkspaceElement;
use crate::shell::KeyboardFocusTarget;
use crate::state::{OutputState, State};

pub struct WindowPreCommitHook(RefCell<Option<HookId>>);

impl XdgShellHandler for State {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.fht.xdg_shell_state
    }

    fn new_toplevel(&mut self, toplevel: ToplevelSurface) {
        let window = Window::new_wayland_window(toplevel);
        add_window_pre_commit_hook(&window);
        self.fht.pending_windows.push(window.into());
    }

    fn toplevel_destroyed(&mut self, surface: ToplevelSurface) {
        if let Some(idx) = self.fht.unmapped_tiles.iter().position(|tile| {
            tile.inner
                .element
                .wl_surface()
                .is_some_and(|s| &*s == surface.wl_surface())
        }) {
            let _unmapped_tile = self.fht.unmapped_tiles.remove(idx);
            return;
        }

        let Some((tile, output)) = self.fht.find_tile_and_output(surface.wl_surface()) else {
            warn!("Destroyed toplevel missing from mapped tiles and unmapped tiles!");
            return;
        };
        OutputState::get(&output).render_state.queue();

        let scale = output.current_scale().fractional_scale().into();
        self.backend.with_renderer(|renderer| {
            tile.prepare_close_animation(renderer, scale);
        });
        self.backend.with_renderer(|renderer| {
            tile.start_close_animation(renderer, scale);
        });

        let (_, ws) = self
            .fht
            .find_window_and_workspace_mut(&surface.wl_surface())
            .unwrap();
        ws.arrange_tiles(true);
    }

    fn new_popup(&mut self, surface: PopupSurface, _positioner: PositionerState) {
        self.fht.unconstrain_popup(&surface);
        if let Err(err) = self.fht.popups.track_popup(PopupKind::from(surface)) {
            warn!(?err, "Failed to track popup!")
        }
    }

    fn move_request(&mut self, surface: ToplevelSurface, _: wl_seat::WlSeat, serial: Serial) {
        if let Some(window) = self.fht.find_window(surface.wl_surface()).cloned() {
            self.handle_move_request(window, serial);
        }
    }

    fn resize_request(
        &mut self,
        surface: ToplevelSurface,
        _seat: wl_seat::WlSeat,
        serial: Serial,
        edges: xdg_toplevel::ResizeEdge,
    ) {
        if let Some(window) = self.fht.find_window(surface.wl_surface()).cloned() {
            self.handle_resize_request(window, serial, edges.into())
        }
    }

    fn ack_configure(&mut self, surface: WlSurface, configure: Configure) {
        if let Configure::Toplevel(configure) = configure {
            if let Some(serial) = with_states(&surface, |states| {
                if let Some(data) = states.data_map.get::<RefCell<ResizeState>>() {
                    if let ResizeState::WaitingForFinalAck(_, serial) = *data.borrow() {
                        return Some(serial);
                    }
                }

                None
            }) {
                // When the resize grab is released the surface
                // resize state will be set to WaitingForFinalAck
                // and the client will receive a configure request
                // without the resize state to inform the client
                // resizing has finished. Here we will wait for
                // the client to acknowledge the end of the
                // resizing. To check if the surface was resizing
                // before sending the configure we need to use
                // the current state as the received acknowledge
                // will no longer have the resize state set
                let is_resizing = with_states(&surface, |states| {
                    states
                        .data_map
                        .get::<XdgToplevelSurfaceData>()
                        .unwrap()
                        .lock()
                        .unwrap()
                        .current
                        .states
                        .contains(xdg_toplevel::State::Resizing)
                });

                if configure.serial >= serial && is_resizing {
                    with_states(&surface, |states| {
                        let state = &mut *states
                            .data_map
                            .get::<RefCell<ResizeState>>()
                            .unwrap()
                            .borrow_mut();
                        *state = match std::mem::take(state) {
                            ResizeState::WaitingForFinalAck(data, _) => {
                                ResizeState::WaitingForCommit(data)
                            }
                            _ => unreachable!(),
                        }
                    });
                }
            }
        }
    }

    fn grab(&mut self, surface: PopupSurface, seat: wl_seat::WlSeat, serial: Serial) {
        let seat: Seat<State> = Seat::from_resource(&seat).unwrap();
        let popup_kind = PopupKind::Xdg(surface);

        if let Some(root) = find_popup_root_surface(&popup_kind).ok().and_then(|root| {
            self.fht
                .find_window(&root)
                .cloned()
                .map(KeyboardFocusTarget::Window)
                .or_else(|| {
                    self.fht
                        .outputs()
                        .find_map(|o| {
                            layer_map_for_output(o)
                                .layer_for_surface(&root, WindowSurfaceType::TOPLEVEL)
                                .cloned()
                        })
                        .map(KeyboardFocusTarget::LayerSurface)
                })
        }) {
            let grab = self.fht.popups.grab_popup(root, popup_kind, &seat, serial);

            if let Ok(mut grab) = grab {
                if let Some(keyboard) = seat.get_keyboard() {
                    if keyboard.is_grabbed()
                        && !(keyboard.has_grab(serial)
                            || keyboard.has_grab(grab.previous_serial().unwrap_or(serial)))
                    {
                        grab.ungrab(PopupUngrabStrategy::All);
                        return;
                    }
                    keyboard.set_focus(self, grab.current_grab(), serial);
                    keyboard.set_grab(self, PopupKeyboardGrab::new(&grab), serial);
                }
                if let Some(pointer) = seat.get_pointer() {
                    if pointer.is_grabbed()
                        && !(pointer.has_grab(serial)
                            || pointer
                                .has_grab(grab.previous_serial().unwrap_or_else(|| grab.serial())))
                    {
                        grab.ungrab(PopupUngrabStrategy::All);
                        return;
                    }
                    pointer.set_grab(self, PopupPointerGrab::new(&grab), serial, Focus::Keep);
                }
            }
        }
    }

    fn maximize_request(&mut self, toplevel: ToplevelSurface) {
        if let Some((window, ws)) = self
            .fht
            .find_window_and_workspace_mut(toplevel.wl_surface())
        {
            window.set_maximized(true);
            ws.arrange_tiles(true);
        }

        toplevel.send_configure();
    }

    fn unmaximize_request(&mut self, toplevel: ToplevelSurface) {
        if let Some((window, ws)) = self
            .fht
            .find_window_and_workspace_mut(toplevel.wl_surface())
        {
            window.set_maximized(false);
            ws.arrange_tiles(true);
        }

        toplevel.send_configure();
    }

    fn fullscreen_request(
        &mut self,
        surface: ToplevelSurface,
        wl_output: Option<wl_output::WlOutput>,
    ) {
        if surface
            .current_state()
            .capabilities
            .contains(WmCapabilities::Fullscreen)
        {
            let wl_surface = surface.wl_surface();
            let requested_output = wl_output.as_ref().and_then(Output::from_resource);

            // If the surface request for a specific output to be fullscreened on, move it to the
            // active workspace of that output, then fullscreen it.
            //
            // If not, then fullscreen it inside its active workspace.
            if let Some((window, requested_output, mut output)) =
                requested_output.and_then(|requested_output| {
                    let (window, current_output) = self.fht.find_window_and_output(wl_surface)?;
                    Some((window.clone(), requested_output, current_output.clone()))
                })
            {
                if requested_output != output {
                    output = requested_output;

                    let ws = self.fht.ws_mut_for(&window).unwrap();
                    let tile = ws.remove_tile(&window, true).unwrap();

                    let new_ws = self.fht.wset_mut_for(&output).active_mut();
                    new_ws.insert_tile(tile, true);
                }

                let ws = self.fht.ws_mut_for(&window).unwrap();
                ws.fullscreen_element(&window, true);
            } else if let Some(window) = self.fht.find_window(wl_surface).cloned() {
                let ws = self.fht.ws_mut_for(&window).unwrap();
                ws.fullscreen_element(&window, true);
            }
        }

        surface.send_configure();
    }

    fn unfullscreen_request(&mut self, surface: ToplevelSurface) {
        if let Some(window) = self.fht.find_window(surface.wl_surface()) {
            window.set_fullscreen(false);
        }

        surface.send_configure();
    }

    fn reposition_request(
        &mut self,
        surface: PopupSurface,
        positioner: PositionerState,
        token: u32,
    ) {
        surface.with_pending_state(|state| {
            let geometry = positioner.get_geometry();
            state.geometry = geometry;
            state.positioner = positioner;
        });
        self.fht.unconstrain_popup(&surface);
        surface.send_repositioned(token);
    }
}

fn add_window_pre_commit_hook(window: &Window) {
    // The workspace tile api is not responsible for actually starting the close animations, we are
    // the ones that should do this.
    let wl_surface = window.wl_surface().unwrap();
    let hook_id = add_pre_commit_hook::<State, _>(&wl_surface, |state, _dh, surface| {
        let Some((tile, output)) = state.fht.find_tile_and_output(surface) else {
            warn!("Window pre-commit hook should be removed when unmapped!");
            return;
        };

        // Before commiting, we check if the window's buffers are getting unmapped.
        // If that's the case, the window is likely closing (or minimizing, if the
        // compositor supports that)
        //
        // Since we are going to close, we take a snapshot of the window's elements,
        // like we do inside `Tile::render_elements` into a
        // GlesTexture and store that for future use.
        let (got_unmapped, configure_serial) = with_states(surface, |states| {
            let mut guard = states.cached_state.get::<SurfaceAttributes>();
            let attrs = guard.pending();
            let got_unmapped = matches!(attrs.buffer, Some(BufferAssignment::Removed) | None);

            let data = states
                .data_map
                .get::<XdgToplevelSurfaceData>()
                .unwrap()
                .lock()
                .unwrap();
            let configure_serial = data.configure_serial;

            (got_unmapped, configure_serial)
        });

        if got_unmapped {
            state.backend.with_renderer(|renderer| {
                let scale = output.current_scale().fractional_scale().into();
                tile.prepare_close_animation(renderer, scale);
            });
        } else {
            if configure_serial
                .is_some_and(|serial| tile.commit_will_cause_resize_animation(serial))
            {
                tile.start_resize_animation();
            }

            tile.clear_close_snapshot();
        }
    });

    window
        .user_data()
        .insert_if_missing(|| WindowPreCommitHook(RefCell::new(None)));
    let mut guard = window
        .user_data()
        .get::<WindowPreCommitHook>()
        .unwrap()
        .0
        .borrow_mut();
    *guard = Some(hook_id)
}

delegate_xdg_shell!(State);
