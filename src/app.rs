use std::time::Instant;
use imgui::Context;
use imgui_winit_support::WinitPlatform;
use winit::
{
	event_loop::{EventLoop, ControlFlow},
	window::{Window, WindowBuilder},
	dpi::LogicalSize,
	event::{Event, WindowEvent, VirtualKeyCode, DeviceEvent}
};
use anyhow::Result;
use nalgebra_glm as glm;

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

	pub fn run(mut self)
	{
		let start = Instant::now();

		let mut destroying = false;
		let mut minimized = false;

		let mut last_frame = Instant::now();

		self.event_loop.run(move |event,_,control_flow|
		{
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
					self.renderer.render(&self.window, start, &mut self.imgui, &mut self.platform);
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
				// Mouse Input
				Event::DeviceEvent {event: DeviceEvent::MouseMotion { delta }, .. } =>
				{
					let (dx, dy) = delta;
					let (dx, dy) = ((dx / 20.0) as f32, (dy / 20.0) as f32);
					self.renderer.update_camera_rotation(glm::vec3(dy, dx, 0.0))
				},
				Event::DeviceEvent {event: DeviceEvent::Key (input), .. } =>
				{
					if let Some(key) = input.virtual_keycode
					{
						if input.state == winit::event::ElementState::Pressed
						{
							if key == VirtualKeyCode::R
							{
								self.renderer.toggle_wireframe(&self.window);
							}
							if key == VirtualKeyCode::Tab
							{
								self.window.set_cursor_visible(!self.renderer.cursor_visible());
							}
							if key == VirtualKeyCode::W
							{
								self.renderer.move_camera_forward();
							}
							if key == VirtualKeyCode::S
							{
								self.renderer.move_camera_backward();
							}
							if key == VirtualKeyCode::A
							{
								self.renderer.move_camera_left();
							}
							if key == VirtualKeyCode::D
							{
								self.renderer.move_camera_right();
							}
						}
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
