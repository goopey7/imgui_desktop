use anyhow::Result;

mod vulkan_helpers;

mod renderer;
use renderer::Renderer;

use std::time::Instant;
use winit::{event_loop::{EventLoop, ControlFlow}, window::WindowBuilder, dpi::LogicalSize, event::{Event, WindowEvent}};

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new()
		.with_title("Goop Renderer")
		.with_inner_size(LogicalSize::new(1024, 768))
		.build(&event_loop)?;

	let mut gr = Renderer::init(window)?;

	let start = Instant::now();

	let mut destroying = false;
	let mut minimized = false;

	event_loop.run(move |event,_,control_flow|
	{
		*control_flow = ControlFlow::Poll;
		match event
		{
			// Render a frame if our Vulkan app is not being destroyed.
			Event::MainEventsCleared if !destroying && !minimized =>
			{
				gr.render(start);
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
					gr.resize();
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
			},
			_ => {}
		}
	});
}

