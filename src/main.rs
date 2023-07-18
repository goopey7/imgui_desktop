use anyhow::Result;

mod vulkan_helpers;
use vulkan_helpers::vh;
use winit::{event_loop::EventLoop, window::WindowBuilder, dpi::LogicalSize};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const MAX_FRAMES_IN_FLIGHT: usize = 3;

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Vulkan Tutorial (Rust)")
		.with_inner_size(LogicalSize::new(1024, 768))
		.build(&event_loop)?;

	let entry = unsafe { ash::Entry::load()? };

	let instance_data = vh::create_instance(&entry, &window, VALIDATION_ENABLED)?;
	let surface_data = vh::create_surface(&entry, &instance_data, &window)?;
	let device_data = vh::create_logical_device(&instance_data, &surface_data)?;
	let mut data = vh::create_swapchain(&window, instance_data, surface_data, device_data)?;
	vh::create_swapchain_image_views(&mut data)?;

	vh::destroy_vulkan(&mut data)?;

	Ok(())
}


