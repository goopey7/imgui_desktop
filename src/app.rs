use std::time::Instant;
use winit::
{
	event_loop::{EventLoop, ControlFlow},
	window::{Window, WindowBuilder},
	dpi::LogicalSize,
	event::{Event, WindowEvent}
};
use anyhow::Result;
use crate::renderer::Renderer;

pub struct App
{
	renderer: Renderer,
	event_loop: EventLoop<()>,
	window: Window,
}

impl App
{
	pub fn new(app_name: &str) -> Result<Self>
	{
		let event_loop = EventLoop::new();
		let window = WindowBuilder::new()
			.with_title(app_name)
			.with_inner_size(LogicalSize::new(1024, 768))
			.build(&event_loop)?;
		let renderer = Renderer::init(&window)?;

		Ok(Self
		{
			renderer,
			event_loop,
			window,
		})
	}

	pub fn run(mut self)
	{
		let start = Instant::now();

		let mut destroying = false;
		let mut minimized = false;

		self.event_loop.run(move |event,_,control_flow|
		{
			*control_flow = ControlFlow::Poll;
			match event
			{
				// Render a frame if our Vulkan app is not being destroyed.
				Event::MainEventsCleared if !destroying && !minimized =>
				{
					self.renderer.render(&self.window, start);
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
						self.renderer.resize();
					}
				},
				// Handle Input
				Event::WindowEvent {event: WindowEvent::KeyboardInput { input, .. }, .. } =>
				{
				},
				// Destroy App (renderer gets dropped automatically)
				Event::WindowEvent { event: WindowEvent::CloseRequested, .. } =>
				{
					destroying = true;
					*control_flow = ControlFlow::Exit;
				},
				_ => {}
			}
		});
	}
}
