//! Egui implementation for the compositor.
//!
//! A lot of thd code ideas and impls are from
//! [smithay-egui](https://github.com/smithay/smithay-egui), with additional tailoring to fit the
//! compositor's needs.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::texture::{TextureRenderBuffer, TextureRenderElement};
use smithay::backend::renderer::element::Kind;
use smithay::backend::renderer::gles::{GlesError, GlesTexture};
use smithay::backend::renderer::glow::GlowRenderer;
use smithay::backend::renderer::{Bind, Color32F, Frame, Offscreen, Renderer};
use smithay::utils::{Logical, Physical, Point, Rectangle, Size, Transform};

use crate::renderer::texture_element::FhtTextureElement;

pub struct EguiElement {
    size: Size<i32, Logical>,
    ctx: egui::Context,
    render_buffer: Arc<Mutex<Option<(i32, TextureRenderBuffer<GlesTexture>)>>>,
}

impl EguiElement {
    pub fn new(size: Size<i32, Logical>) -> Self {
        Self {
            size,
            ctx: Default::default(),
            render_buffer: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_size(&mut self, new_size: Size<i32, Logical>) {
        self.size = new_size;
        // Reset render buffer on resize to avoid artifacts
        let _ = self.render_buffer.lock().unwrap().take();
    }

    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    pub fn render(
        &self,
        renderer: &mut GlowRenderer,
        scale: i32,
        alpha: f32,
        location: Point<i32, Physical>,
        ui: impl FnMut(&egui::Context),
    ) -> Result<EguiRenderElement, GlesError> {
        let size = self.size.to_physical(scale);
        let buffer_size = self.size.to_buffer(scale, Transform::Normal);

        if renderer
            .egl_context()
            .user_data()
            .get::<Rc<RefCell<egui_glow::Painter>>>()
            .is_none()
        {
            let painter = renderer
                .with_context(|context| {
                    // SAFETY: In the context of this compositor, the glow renderer/context
                    // lives for 'static, so the pointer to it should always be valid.
                    egui_glow::Painter::new(context.clone(), "", None, true)
                })?
                .map_err(|err| {
                    warn!(?err, "Failed to create egui glow painter for output");
                    GlesError::ShaderCompileError
                })?;

            renderer
                .egl_context()
                .user_data()
                .insert_if_missing(|| Rc::new(RefCell::new(painter)));
        }

        let painter = renderer
            .egl_context()
            .user_data()
            .get::<Rc<RefCell<egui_glow::Painter>>>()
            .cloned()
            .unwrap();
        let painter = &mut *RefCell::borrow_mut(&painter);

        let render_buffer = &mut *self.render_buffer.lock().unwrap();
        let _ = render_buffer.take_if(|(s, _)| *s != scale);
        let render_buffer = match render_buffer.as_mut() {
            Some((_, render_buffer)) => render_buffer,
            None => {
                let render_texture: GlesTexture = renderer
                    .create_buffer(Fourcc::Abgr8888, buffer_size)
                    .map_err(|err| {
                        warn!(?err, "Failed to create egui overlay texture buffer");
                        err
                    })?;

                let texture_buffer = TextureRenderBuffer::from_texture(
                    renderer,
                    render_texture,
                    scale,
                    Transform::Flipped180, // egui glow painter wants this.
                    None,
                );

                let render_buffer = render_buffer.insert((scale, texture_buffer));
                &mut render_buffer.1
            }
        };

        let input = egui::RawInput {
            viewport_id: egui::ViewportId::ROOT,
            viewports: egui::ViewportIdMap::from_iter([(
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(scale as f32),
                    ..Default::default()
                },
            )]),
            screen_rect: Some(egui::Rect {
                min: egui::pos2(0.0, 0.0),
                max: egui::pos2(size.w as f32, size.h as f32),
            }),
            predicted_dt: 1.0 / 60.0,
            focused: true,
            ..Default::default()
        };
        let egui::FullOutput {
            shapes,
            textures_delta,
            ..
        } = self.ctx.run(input.clone(), ui);

        render_buffer.render().draw(|texture| {
            {
                let mut tex = texture.clone();
                let mut fb = renderer.bind(&mut tex)?;
                let mut frame = renderer.render(&mut fb, size, Transform::Normal)?;
                frame.clear(Color32F::TRANSPARENT, &[Rectangle::from_size(size)])?;
                painter.paint_and_update_textures(
                    [size.w as u32, size.h as u32],
                    scale as f32,
                    &self.ctx.tessellate(shapes, scale as f32),
                    &textures_delta,
                );
            };

            // FIXME: This is not "optimal" per-se, but works fine.
            //
            // If we want the best way I would bee to access egui::Memory::visible_windows, but its
            // gated behind a pub(crate), and all the funtions needed to reproduce it are too...
            let egui::Rect { min, max } = self.ctx().used_rect();
            let used_rect = Rectangle::<i32, Logical>::from_extemities(
                (min.x.round() as i32, min.y.round() as i32),
                (max.x.round() as i32, max.y.round() as i32),
            );

            Result::<_, GlesError>::Ok(vec![used_rect.to_buffer(
                scale,
                Transform::Flipped180,
                &self.size,
            )])
        })?;

        let texture_element: FhtTextureElement = TextureRenderElement::from_texture_render_buffer(
            location.to_f64(),
            &render_buffer,
            Some(alpha),
            None,
            Some(self.size),
            Kind::Unspecified,
        )
        .into();
        Ok(texture_element.into())
    }
}

crate::fht_render_elements! {
    EguiRenderElement => {
        Texture = FhtTextureElement,
    }
}
