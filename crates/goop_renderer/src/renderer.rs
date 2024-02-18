use anyhow::Result;
use winit::window::Window;
use crate::vulkan_helpers::vh::{Data, self};

use imgui::*;
use imgui_winit_support::WinitPlatform;
use imgui_rs_vulkan_renderer::Options;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub struct Renderer
{
	_entry: ash::Entry, // needs to live as long as other vulkan objects
	instance: ash::Instance,
	device: ash::Device,
	surface: ash::extensions::khr::Surface,
	data: Data,
	pub cursor_visible: bool,
    pub imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
}

impl Renderer
{
	pub fn init(window: &Window, app_name: &str, imgui: &mut Context, mut ui_setup: Box<dyn FnMut(&mut Context)>) -> Result<Self>
	{
		let (entry, instance, surface, device, data) = Renderer::init_renderer(window, app_name)?;

		let imgui_renderer = imgui_rs_vulkan_renderer::Renderer::with_default_allocator(
			&instance,
			data.physical_device,
			device.clone(),
			data.graphics_queue,
			data.graphics_command_pool,
			data.render_pass,
			imgui,
			Some(Options
				{
					in_flight_frames: 3,
					enable_depth_test: false,
					enable_depth_write: false,
				}
			),
		)?;

		ui_setup(imgui);
		Ok(Renderer
		{
			_entry: entry,
			instance,
			device,
			surface,
			data,
			imgui_renderer,
			cursor_visible: true,
		})
	}

	pub fn cursor_visible(&self) -> bool
	{
		self.cursor_visible
	}

	fn init_renderer(window: &Window, app_name: &str) -> Result<(ash::Entry, ash::Instance, ash::extensions::khr::Surface, ash::Device, Data)>
	{
		log::info!("Initializing Renderer........");

		let mut data = Data::default();
		let entry = unsafe { ash::Entry::load()? };
		let instance = vh::create_instance(&entry, window, VALIDATION_ENABLED, &mut data, app_name)?;
		let surface = vh::create_surface(&entry, &instance, window, &mut data)?;
		let device = vh::create_logical_device(&instance, &surface, &mut data)?;
		vh::set_msaa_samples(&mut data)?;
		vh::create_swapchain(&instance, &device, &surface, window, &mut data)?;
		vh::create_swapchain_image_views(&device, &mut data)?;
		vh::create_render_pass(&instance, &device, &mut data)?;
		vh::create_command_pools(&instance, &device, &surface, &mut data)?;

		vh::create_color_objects(&instance, &device, &mut data)?;
		vh::create_depth_objects(&instance, &device, &mut data)?;
		vh::create_framebuffers(&device, &mut data)?;
		vh::create_command_buffers(&device, &mut data)?;
		vh::create_sync_objects(&device, &mut data)?;

		log::info!("Renderer Initialized Successfully");
		Ok((entry, instance, surface, device, data))
	}

	pub fn render(&mut self, window: &Window, imgui: &mut Context, platform: &mut WinitPlatform, ui_fn: &dyn Fn(&mut Ui))
	{
		platform
			.prepare_frame(imgui.io_mut(), &window)
			.expect("Failed to prepare frame");

		let ui = imgui.frame();

		ui_fn(ui);

		platform.prepare_render(&ui, &window);
		let draw_data = imgui.render();

		vh::render(
			&self.instance,
			&self.device,
			&self.surface,
			&window,
			&mut self.data,
			&mut self.imgui_renderer,
			&draw_data,
		).unwrap();
	}

	pub fn resize(&mut self)
	{
		self.data.resized = true;
	}

	pub fn toggle_wireframe(&mut self, window: &Window)
	{
		vh::toggle_wireframe(
			&self.instance,
			&self.device,
			&self.surface,
			&window,
			&mut self.data,
		).unwrap();
	}
}

impl Drop for Renderer
{
	fn drop(&mut self)
	{
		log::info!("Destroying Renderer");
		unsafe
		{
			return vh::destroy(&self.device, &self.data);
		}
	}
}

