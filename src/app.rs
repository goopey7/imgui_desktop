use std::time::Instant;
use imgui::Context;
use imgui_winit_support::WinitPlatform;
use winit::
{
	event_loop::{EventLoop, ControlFlow},
	window::{Window, WindowBuilder},
	dpi::LogicalSize,
	event::{Event, WindowEvent}
};
use anyhow::Result;

use goop_renderer::renderer::Renderer;
pub struct App
{
	renderer: Renderer,
	event_loop: EventLoop<()>,
	window: Window,
	imgui: Context,
	platform: WinitPlatform,
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

		let mut imgui = Context::create();
		let mut platform = WinitPlatform::init(&mut imgui);
		platform.attach_window(imgui.io_mut(), &window, imgui_winit_support::HiDpiMode::Rounded);

		let renderer = Renderer::init(&window, app_name, &mut imgui)?;

		Ok(Self
		{
			renderer,
			event_loop,
			window,
			imgui,
			platform,
		})
	}

	pub fn run(mut self, ui_fn: Box<dyn Fn(&mut imgui::Ui)>)
	{
		let mut destroying = false;
		let mut minimized = false;

		let mut last_frame = Instant::now();

		self.event_loop.run(move |event,_,control_flow|
		{
			self.window.set_cursor_visible(self.renderer.cursor_visible());
			*control_flow = ControlFlow::Poll;
			self.platform.handle_event(self.imgui.io_mut(), &self.window, &event);
			match event
			{
				// New Frame
				Event::NewEvents(_) =>
				{
					let now = Instant::now();
                    self.imgui.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;

				}
				// Render a frame if our Vulkan app is not being destroyed.
				Event::MainEventsCleared if !destroying && !minimized =>
				{
					self.renderer.render(&self.window, &mut self.imgui, &mut self.platform, &*ui_fn);
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
