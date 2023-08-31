use anyhow::Result;
use std::time::Instant;
use winit::window::Window;
use crate::vulkan_helpers::vh::{Data, self};
use nalgebra_glm as glm;

#[cfg(feature = "goop_imgui")]
use imgui::*;
#[cfg(feature = "goop_imgui")]
use imgui_winit_support::WinitPlatform;
#[cfg(feature = "goop_imgui")]
use imgui_rs_vulkan_renderer::Options;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub struct Renderer
{
	_entry: ash::Entry, // needs to live as long as other vulkan objects
	instance: ash::Instance,
	device: ash::Device,
	surface: ash::extensions::khr::Surface,
	data: Data,

	#[cfg(feature = "goop_imgui")]
    pub imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
}

impl Renderer
{
	#[cfg(not(feature = "goop_imgui"))]
	pub fn init(window: &Window, app_name: &str) -> Result<Self>
	{
		let (entry, instance, surface, device, data) = Renderer::init_renderer(window, app_name)?;

		Ok(
			Self
			{
				_entry: entry,
				instance,
				surface,
				device,
				data,
			}
		)
	}

	#[cfg(feature = "goop_imgui")]
	pub fn init(window: &Window, app_name: &str, imgui: &mut Context) -> Result<Self>
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

		Ok(Renderer
		{
			_entry: entry,
			instance,
			device,
			surface,
			data,
			imgui_renderer,
		})
	}

	fn init_renderer(window: &Window, app_name: &str) -> Result<(ash::Entry, ash::Instance, ash::extensions::khr::Surface, ash::Device, Data)>
	{
		log::info!("Initializing Renderer........");

		let mut data = Data::default();
		let entry = unsafe { ash::Entry::load()? };
		let instance = vh::create_instance(&entry, window, VALIDATION_ENABLED, &mut data, app_name)?;
		let surface = vh::create_surface(&entry, &instance, window, &mut data)?;
		let device = vh::create_logical_device(&instance, &surface, &mut data)?;
		vh::set_msaa_samples(&instance, &mut data)?;
		vh::create_swapchain(&instance, &device, &surface, window, &mut data)?;
		vh::create_swapchain_image_views(&device, &mut data)?;
		vh::create_render_pass(&instance, &device, &mut data)?;
		vh::create_command_pools(&instance, &device, &surface, &mut data)?;

		let earth_tex = vh::add_texture(&instance, &device, &mut data, "media/textures/earth.png")?;
		let moon_tex = vh::add_texture(&instance, &device, &mut data, "media/textures/moon.png")?;
		let viking_tex = vh::add_texture(&instance, &device, &mut data, "media/textures/viking_room.png")?;

		vh::create_descriptor_set_layout(&device, &mut data)?;
		vh::create_pipeline(&device, &mut data)?;
		vh::create_color_objects(&instance, &device, &mut data)?;
		vh::create_depth_objects(&instance, &device, &mut data)?;
		vh::create_framebuffers(&device, &mut data)?;
		vh::create_uniform_buffers(&instance, &device, &mut data)?;
		vh::create_descriptor_pool(&device, &mut data)?;
		vh::create_descriptor_sets(&device, &mut data)?;
		vh::create_command_buffers(&device, &mut data)?;
		vh::create_sync_objects(&device, &mut data)?;

		// LOAD MODELS
		let planet_model = vh::load_model(&mut data, "media/models/smallSphere.obj")?;
		let large_pm = vh::load_model(&mut data, "media/models/largeSphere.obj")?;
		let room_model = vh::load_model(&mut data, "media/models/viking_room.obj")?;

		let traingle_verts = vec![glm::vec3(0.0, 1.0, 0.0), glm::vec3(1.0, 0.0, 0.0), glm::vec3(-1.0, 0.0, 0.0)];
		let triangle_indices = vec![0, 1, 2];
		let tri = vh::load_vertics(&mut data, traingle_verts, triangle_indices, None, None)?;

		// LOAD INSTANCES
		vh::prep_instances(&mut data)?;
		let earth = vh::InstanceData::new(glm::Mat4::identity(), earth_tex);
		let moon = vh::InstanceData::new(glm::Mat4::identity(), moon_tex);
		//vh::load_instances(&mut data, planet_model, vec![moon])?;

		let room = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(2.0, 0.0, 0.0)), viking_tex);
		let room1 = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(-2.0, 0.0, 0.0)), viking_tex);
		let room2 = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(0.0, 0.0, 0.0)), viking_tex);
		let room3 = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(0.0, 2.0, 0.0)), viking_tex);
		let room4 = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(-2.0, 2.0, 0.0)), viking_tex);
		let room5 = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(2.0, 2.0, 0.0)), viking_tex);
		vh::add_instances(&mut data, room_model, vec![room, room1, room2, room3, room4, room5])?;

		let earth = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(0.0, -2.0, 0.0)), moon_tex);
		let earth1 = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(2.0, -2.0, 0.0)), earth_tex);
		let earth2 = vh::InstanceData::new(glm::translate(&glm::Mat4::identity(), &glm::vec3(-2.0, -2.0, 0.0)), earth_tex);
		vh::add_instances(&mut data, tri, vec![earth, earth1, earth2])?;

		vh::create_instance_buffer(&instance, &device, &mut data)?;
		vh::create_vertex_buffer(&instance, &device, &mut data)?;
		vh::create_index_buffer(&instance, &device, &mut data)?;

		log::info!("Renderer Initialized Successfully");
		Ok((entry, instance, surface, device, data))
	}

	#[cfg(not(feature = "goop_imgui"))]
	pub fn render(&mut self, window: &Window, start: Instant)
	{
		vh::render(
			&self.instance,
			&self.device,
			&self.surface,
			&window,
			&mut self.data,
			&start
		).unwrap();
	}

	#[cfg(feature = "goop_imgui")]
	pub fn render(&mut self, window: &Window, start: Instant, imgui: &mut Context, platform: &mut WinitPlatform)
	{
		platform
			.prepare_frame(imgui.io_mut(), &window)
			.expect("Failed to prepare frame");

		let ui = imgui.frame();

		// TODO CALL ALL ENTITY RENDER FUNCTIONS HERE

		ui.main_menu_bar(||
		{
			ui.menu("View", || {
				if ui.menu_item_config("Wireframe")
					.selected(self.data.wireframe)
					.build()
				{
					self.toggle_wireframe(&window);
				}
			});
			ui.spacing();
			ui.text(format!("FPS: {:.1}", 1.0 / ui.io().delta_time));
		}
		);

		platform.prepare_render(&ui, &window);
		let draw_data = imgui.render();

		vh::render(
			&self.instance,
			&self.device,
			&self.surface,
			&window,
			&mut self.data,
			&start,
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
			return vh::destroy(&self.instance, &self.device, &self.surface, &self.data);
		}
	}
}

