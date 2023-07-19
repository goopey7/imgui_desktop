use anyhow::Result;

mod vulkan_helpers;
use vulkan_helpers::vh::{self, Data};
use winit::{event_loop::{EventLoop, ControlFlow}, window::{WindowBuilder, Window}, dpi::LogicalSize, event::{Event, WindowEvent}};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

struct App
{
	device: ash::Device,
	data: Data,
}

impl App
{
	fn create(window: &Window) -> Result<Self>
	{
		let mut data = Data::default();
		let entry = unsafe { ash::Entry::load()? };
		let instance = vh::create_instance(&entry, &window, VALIDATION_ENABLED, &mut data)?;
		let surface = vh::create_surface(&entry, &instance, &window, &mut data)?;
		let device = vh::create_logical_device(&instance, &surface, &mut data)?;
		vh::create_swapchain(&instance, &device, &surface, &window, &mut data)?;
		vh::create_swapchain_image_views(&device, &mut data)?;
		vh::create_render_pass(&device, &mut data)?;
		vh::create_pipeline(&device, &mut data)?;
		vh::create_framebuffers(&device, &mut data)?;
		vh::create_command_pools(&instance, &device, &surface, &mut data)?;
		vh::create_command_buffers(&device, &mut data)?;
		vh::create_sync_objects(&device, &mut data)?;
		Ok( Self { device, data, } )
	}

	fn render(&mut self)
	{
		vh::render(&self.device, &mut self.data).unwrap();
	}

	fn destroy(&mut self)
	{
		vh::destroy_vulkan(&self.device, &mut self.data);
	}
}

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Vulkan Tutorial (Rust)")
		.with_inner_size(LogicalSize::new(1024, 768))
		.build(&event_loop)?;
	let mut destroying = false;
	let mut minimized = false;

	let mut app = App::create(&window)?;
	event_loop.run(move |event,_,control_flow|
	{
		*control_flow = ControlFlow::Poll;
		match event
		{
			// Render a frame if our Vulkan app is not being destroyed.
			Event::MainEventsCleared if !destroying && !minimized =>
			{
				app.render();
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
				app.destroy();
			},
			_ => {}
		}
	})
}

