mod vulkan_helpers;
mod renderer;
mod app;
use app::App;

use anyhow::Result;

fn main() -> Result<()>
{
	pretty_env_logger::init();

	App::new("Goop Engine")?.run();

	Ok(())
}

