use imgui::Condition;
use imgui_desktop::app::App;
use anyhow::Result;

fn ui_fn(ctx: &mut imgui::Context) -> &mut imgui::Ui
{
	// disable imgui.ini
	ctx.set_ini_filename(None);

	let display_size = ctx.io().display_size;
	let ui = ctx.frame();

	ui.window("FAhhh fn nice!!")
		.size(display_size, Condition::Always)
		.position([0f32, 0f32], Condition::Always)
		.collapsible(false)
		.menu_bar(true)
		.title_bar(false)
		.build(|| {
			ui.menu_bar(|| {
				ui.text(format!("FPS: {:.1}", 1.0 / ui.io().delta_time));
			});
			ui.spacing();
			ui.text("Rotation");

			ui.spacing();
			ui.text("Position");

			ui.spacing();
			ui.text("Forward");

			ui.spacing();
			ui.text("Up");
		});
	ui
}

fn main() -> Result<()>
{
	pretty_env_logger::init();

	App::new("Message Editor")?.run(Box::new(ui_fn));

	Ok(())
}

