use anyhow::Result;

mod vulkan_helpers;
use vulkan_helpers::vh::{self, Data};
use std::time::Instant;
use winit::{event_loop::{EventLoop, ControlFlow}, window::WindowBuilder, dpi::LogicalSize, event::{Event, WindowEvent}};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Vulkan Tutorial (Ash)")
		.with_inner_size(LogicalSize::new(1024, 768))
		.build(&event_loop)?;
	let mut destroying = false;
	let mut minimized = false;

	let start = Instant::now();

	let mut data = Data::default();
	let entry = unsafe { ash::Entry::load()? };
	let instance = vh::create_instance(&entry, &window, VALIDATION_ENABLED, &mut data)?;
	let surface = vh::create_surface(&entry, &instance, &window, &mut data)?;
	let device = vh::create_logical_device(&instance, &surface, &mut data)?;
	vh::create_swapchain(&instance, &device, &surface, &window, &mut data)?;
	vh::create_swapchain_image_views(&device, &mut data)?;
	vh::create_render_pass(&instance, &device, &mut data)?;
	vh::create_descriptor_set_layout(&device, &mut data)?;
	vh::create_pipeline(&device, &mut data)?;
	vh::create_command_pools(&instance, &device, &surface, &mut data)?;
	vh::create_depth_objects(&instance, &device, &mut data)?;
	vh::create_framebuffers(&device, &mut data)?;
	vh::create_texture_image(&instance, &device, &mut data)?;
	vh::create_texture_image_views(&device, &mut data)?;
	vh::create_texture_sampler(&device, &mut data)?;
	vh::create_vertex_buffer(&instance, &device, &mut data)?;
	vh::create_index_buffer(&instance, &device, &mut data)?;
	vh::create_uniform_buffers(&instance, &device, &mut data)?;
	vh::create_descriptor_pool(&device, &mut data)?;
	vh::create_descriptor_sets(&device, &mut data)?;
	vh::create_command_buffers(&device, &mut data)?;
	vh::create_sync_objects(&device, &mut data)?;

	event_loop.run(move |event,_,control_flow|
	{
		*control_flow = ControlFlow::Poll;
		match event
		{
			// Render a frame if our Vulkan app is not being destroyed.
			Event::MainEventsCleared if !destroying && !minimized =>
			{
				vh::render(&instance, &device, &surface, &window, &mut data, &start).unwrap();
			},
			// Check for resize
			Event::WindowEvent {event: WindowEvent::Resized(size), ..} =>
			{
				if size.width == 0 || size.height == 0
				{
					minimized = true;
				}
				else
				{
					minimized = false;
				}
			},
			// Handle Input
			Event::WindowEvent {event: WindowEvent::KeyboardInput { input, .. }, .. } =>
			{
			},
			// Destroy App
			Event::WindowEvent { event: WindowEvent::CloseRequested, .. } =>
			{
				destroying = true;
				*control_flow = ControlFlow::Exit;
				unsafe { vh::destroy(&instance, &device, &surface, &data); }
			},
			_ => {}
		}
	})
}

