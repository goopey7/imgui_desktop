use anyhow::Result;
use vulkan_helpers::{create_instance, destroy_vulkan};

mod vulkan_helpers;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const _MAX_FRAMES_IN_FLIGHT: usize = 3;

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let entry = unsafe { ash::Entry::load()? };

	let (_instance, mut data) = create_instance(&entry, VALIDATION_ENABLED)?;

	destroy_vulkan(&mut data)?;

	Ok(())
}


