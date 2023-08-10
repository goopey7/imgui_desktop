use anyhow::Result;
use std::time::Instant;
use winit::window::Window;
use crate::vulkan_helpers::vh::{Data, self};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub struct Renderer
{
	_entry: ash::Entry, // needs to live as long as other vulkan objects
	instance: ash::Instance,
	device: ash::Device,
	surface: ash::extensions::khr::Surface,
	data: Data,
	window: Window,
}

impl Renderer
{
	pub fn init(window: Window) -> Result<Self>
	{
		let mut data = Data::default();
		data.resized = false;
		let entry = unsafe { ash::Entry::load()? };
		let instance = vh::create_instance(&entry, &window, VALIDATION_ENABLED, &mut data)?;
		let surface = vh::create_surface(&entry, &instance, &window, &mut data)?;
		let device = vh::create_logical_device(&instance, &surface, &mut data)?;
		vh::set_msaa_samples(&instance, &mut data)?;
		vh::create_swapchain(&instance, &device, &surface, &window, &mut data)?;
		vh::create_swapchain_image_views(&device, &mut data)?;
		vh::create_render_pass(&instance, &device, &mut data)?;
		vh::create_descriptor_set_layout(&device, &mut data)?;
		vh::create_pipeline(&device, &mut data)?;
		vh::create_command_pools(&instance, &device, &surface, &mut data)?;
		vh::create_color_objects(&instance, &device, &mut data)?;
		vh::create_depth_objects(&instance, &device, &mut data)?;
		vh::create_framebuffers(&device, &mut data)?;
		vh::create_texture_image(&instance, &device, &mut data)?;
		vh::create_texture_image_views(&device, &mut data)?;
		vh::create_texture_sampler(&device, &mut data)?;
		vh::load_model(&mut data)?;
		vh::create_vertex_buffer(&instance, &device, &mut data)?;
		vh::create_index_buffer(&instance, &device, &mut data)?;
		vh::create_uniform_buffers(&instance, &device, &mut data)?;
		vh::create_descriptor_pool(&device, &mut data)?;
		vh::create_descriptor_sets(&device, &mut data)?;
		vh::create_command_buffers(&device, &mut data)?;
		vh::create_sync_objects(&device, &mut data)?;

		Ok(Renderer
		{
			_entry: entry,
			instance,
			device,
			surface,
			data,
			window,
		})
	}

	pub fn render(&mut self, start: Instant)
	{
		vh::render(
			&self.instance,
			&self.device,
			&self.surface,
			&self.window,
			&mut self.data,
			&start
		).unwrap();
	}

	pub fn resize(&mut self)
	{
		self.data.resized = true;
	}
}

impl Drop for Renderer
{
	fn drop(&mut self)
	{
		log::info!("Destroying Renderer........");
		unsafe
		{
			vh::destroy(&self.instance, &self.device, &self.surface, &self.data)
		}
		log::info!("Renderer Destroyed Successfully")
	}
}

