use clipboard::{ClipboardContext, ClipboardProvider};
use imgui::ClipboardBackend;

pub struct Clipboard {
    ctx: ClipboardContext,
}

impl Clipboard {
    pub fn new() -> Self {
        Self {
            ctx: ClipboardContext::new().unwrap(),
        }
    }
}

impl ClipboardBackend for Clipboard {
    fn get(&mut self) -> Option<String> {
        self.ctx.get_contents().ok()
    }

    fn set(&mut self, value: &str) {
		self.ctx.set_contents(value.to_owned()).ok();
	}
}
