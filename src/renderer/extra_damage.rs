// From https://github.com/Yalter/Niri licensed under GPL-v3.0
// Thank you very much.
use smithay::backend::renderer::element::{Element, Id, RenderElement};
use smithay::backend::renderer::utils::CommitCounter;
use smithay::backend::renderer::Renderer;
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size};

#[derive(Debug, Clone)]
pub struct ExtraDamage {
    id: Id,
    commit: CommitCounter,
    geometry: Rectangle<i32, Logical>,
}

impl ExtraDamage {
    pub fn new(size: Size<i32, Logical>) -> Self {
        Self {
            id: Id::new(),
            commit: Default::default(),
            geometry: Rectangle::from_size(size),
        }
    }

    pub fn set_size(&mut self, size: Size<i32, Logical>) {
        if self.geometry.size == size {
            return;
        }

        self.geometry.size = size;
        self.commit.increment();
    }

    pub fn with_location(mut self, location: Point<i32, Logical>) -> Self {
        self.geometry.loc = location;
        self
    }
}

impl Default for ExtraDamage {
    fn default() -> Self {
        Self::new(Size::default())
    }
}

impl Element for ExtraDamage {
    fn id(&self) -> &Id {
        &self.id
    }

    fn current_commit(&self) -> CommitCounter {
        self.commit
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        Rectangle::new((0., 0.).into(), (1., 1.).into())
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.geometry.to_physical_precise_round(scale)
    }
}

impl<R: Renderer> RenderElement<R> for ExtraDamage {
    fn draw(
        &self,
        _frame: &mut R::Frame<'_, '_>,
        _src: Rectangle<f64, Buffer>,
        _dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        _or: &[Rectangle<i32, Physical>],
    ) -> Result<(), R::Error> {
        Ok(())
    }
}
