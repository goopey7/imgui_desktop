use anyhow::Result;
use std::time::Instant;
use winit::window::Window;
use crate::vulkan_helpers::vh::{Data, self};
use nalgebra_glm as glm;

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
	camera_eye: glm::Vec3,
	camera_forward: glm::Vec3,
	camera_up: glm::Vec3,
	camera_rotation: glm::Vec3,
	pub cursor_visible: bool,

    pub imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
}

impl Renderer
{
	pub fn init(window: &Window, app_name: &str, imgui: &mut Context) -> Result<Self>
	{
		let (entry, instance, surface, device, data) = Renderer::init_renderer(window, app_name)?;

		imgui.io_mut().config_flags |= imgui::ConfigFlags::IS_SRGB;
		imgui.io_mut().config_flags |= imgui::ConfigFlags::DOCKING_ENABLE;

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
			camera_eye: glm::vec3(0.0, 0.0, 8.0),
			camera_forward: glm::vec3(0.0, 0.0, -1.0),
			camera_up: glm::vec3(0.0, 1.0, 0.0),
			camera_rotation: glm::vec3(0.0, 0.0, 0.0),
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
		vh::set_msaa_samples(&instance, &mut data)?;
		vh::create_swapchain(&instance, &device, &surface, window, &mut data)?;
		vh::create_swapchain_image_views(&device, &mut data)?;
		vh::create_render_pass(&instance, &device, &mut data)?;
		vh::create_command_pools(&instance, &device, &surface, &mut data)?;

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

		log::info!("Renderer Initialized Successfully");
		Ok((entry, instance, surface, device, data))
	}

	pub fn render(&mut self, window: &Window, start: Instant, imgui: &mut Context, platform: &mut WinitPlatform)
	{
		platform
			.prepare_frame(imgui.io_mut(), &window)
			.expect("Failed to prepare frame");

		let ui = imgui.frame();

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

		ui.dockspace_over_main_viewport();

		ui.window("Camera Info")
			.size([200.0, 100.0], Condition::FirstUseEver)
			.build(|| {
				ui.spacing();
				ui.text("Rotation");
				ui.text(format!("Pitch: {:.1}", self.camera_rotation.x));
				ui.text(format!("Yaw: {:.1}", self.camera_rotation.y));
				ui.text(format!("Roll: {:.1}", self.camera_rotation.z));

				ui.spacing();
				ui.text("Position");
				ui.text(format!("X: {:.1}", self.camera_eye.x));
				ui.text(format!("Y: {:.1}", self.camera_eye.y));
				ui.text(format!("Z: {:.1}", self.camera_eye.z));

				ui.spacing();
				ui.text("Forward");
				ui.text(format!("X: {:.1}", self.camera_forward.x));
				ui.text(format!("Y: {:.1}", self.camera_forward.y));
				ui.text(format!("Z: {:.1}", self.camera_forward.z));

				ui.spacing();
				ui.text("Up");
				ui.text(format!("X: {:.1}", self.camera_up.x));
				ui.text(format!("Y: {:.1}", self.camera_up.y));
				ui.text(format!("Z: {:.1}", self.camera_up.z));
			});

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
			self.camera_eye,
			self.camera_forward,
			self.camera_up,
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

	pub fn move_camera_right(&mut self, dt: f32)
	{
		self.camera_eye += glm::normalize(&glm::cross(&self.camera_forward, &self.camera_up)) * dt;
	}

	pub fn move_camera_left(&mut self, dt: f32)
	{
		self.camera_eye -= glm::normalize(&glm::cross(&self.camera_forward, &self.camera_up)) * dt;
	}

	pub fn move_camera_backward(&mut self, dt: f32)
	{
		self.camera_eye -= self.camera_forward * dt;
	}

	pub fn move_camera_forward(&mut self, dt: f32)
	{
		self.camera_eye += self.camera_forward * dt;
	}

	pub fn move_camera_up(&mut self, dt: f32)
	{
		self.camera_eye += self.camera_up * dt;
	}

	pub fn move_camera_down(&mut self, dt: f32)
	{
		self.camera_eye -= self.camera_up * dt;
	}

	pub fn update_camera_rotation(&mut self, rotation: glm::Vec3)
	{
		self.camera_rotation += rotation;
		self.camera_rotation.x = self.camera_rotation.x.max(-89.0).min(89.0);

		let (cos_p, cos_y, cos_r) = (self.camera_rotation.x.to_radians().cos(), self.camera_rotation.y.to_radians().cos(), self.camera_rotation.z.to_radians().cos());
		let (sin_p, sin_y, sin_r) = (self.camera_rotation.x.to_radians().sin(), self.camera_rotation.y.to_radians().sin(), self.camera_rotation.z.to_radians().sin());

		self.camera_forward = glm::vec3(sin_y * cos_p, sin_p, cos_p * -cos_y);
		self.camera_forward = self.camera_forward.normalize();

		self.camera_up = glm::vec3(
			-cos_y * sin_r - sin_y * sin_p * cos_r,
			cos_p * cos_r,
			-sin_y * sin_r - sin_p * cos_r * -cos_y,
		);
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

