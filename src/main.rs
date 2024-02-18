use imgui::Condition;
use imgui_desktop::app::App;
use anyhow::Result;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct State
{
	msg: String
}

impl State
{
	fn ui_setup(&mut self, ctx: &mut imgui::Context)
	{
		// disable imgui.ini
		ctx.set_ini_filename(None);
		self.msg = "Whattup State".to_string();
	}

	fn ui_update(&mut self, ui: &mut imgui::Ui)
	{
		ui.window("FAhhh fn nice!!")
			.size([ui.io().display_size[0], ui.io().display_size[1]], Condition::Always)
			.position([0f32, 0f32], Condition::Always)
			.collapsible(false)
			.movable(false)
			.menu_bar(true)
			.title_bar(false)
			.build(|| {
				ui.menu_bar(|| {
					ui.text(format!("FPS: {:.1}", 1.0 / ui.io().delta_time));
				});

				if ui.button("printMsg")
				{
					println!("{}", self.msg);
					self.msg = "Ayyyyyy".to_string();
				}

				ui.input_text("input", &mut self.msg).build();
			});
	}
}

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let shared_state = Arc::new(Mutex::new(State::default()));

	let shared_state_setup = shared_state.clone();
	let ui_setup = move |ctx: &mut imgui::Context|
	{
		let mut state = shared_state_setup.lock().unwrap();
		state.ui_setup(ctx);
	};

	let shared_state_update = shared_state.clone();
	let ui_update = move |ui: &mut imgui::Ui|
	{
		let mut state = shared_state_update.lock().unwrap();
		state.ui_update(ui);
	};

	App::new("Message Editor", Box::new(ui_setup))?.run(Box::new(ui_update));

	Ok(())
}

