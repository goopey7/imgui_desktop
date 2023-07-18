use anyhow::Result;

mod vulkan_helpers;
use vulkan_helpers::vh;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const MAX_FRAMES_IN_FLIGHT: usize = 3;

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let entry = unsafe { ash::Entry::load()? };

	let instance_data = vh::create_instance(&entry, VALIDATION_ENABLED)?;
	let mut data = vh::create_logical_device(instance_data)?;

	vh::destroy_vulkan(&mut data)?;

	Ok(())
}


