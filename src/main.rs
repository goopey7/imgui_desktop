use imgui::Condition;
use imgui_desktop::app::App;
use anyhow::Result;

fn ui_fn(ui: &mut imgui::Ui)
{
	ui.main_menu_bar(||
	{
		ui.text(format!("FPS: {:.1}", 1.0 / ui.io().delta_time));
	}
	);

	ui.dockspace_over_main_viewport();

	ui.window("Ahhh fn nice!")
		.size([200.0, 100.0], Condition::FirstUseEver)
		.build(|| {
			ui.spacing();
			ui.text("Rotation");

			ui.spacing();
			ui.text("Position");

			ui.spacing();
			ui.text("Forward");

			ui.spacing();
			ui.text("Up");
		});
}

fn main() -> Result<()>
{
	pretty_env_logger::init();

	App::new("Message Editor")?.run(Box::new(ui_fn));

	Ok(())
}

