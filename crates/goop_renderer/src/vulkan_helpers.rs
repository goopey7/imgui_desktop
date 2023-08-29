pub mod vh
{
	use std::{ffi::CString, fs::File};
	use std::mem::size_of;
	use std::ptr::copy_nonoverlapping as memcpy;
	use std::collections::HashMap;
	use std::hash::{Hash, Hasher};
	use std::io::BufReader;
	use anyhow::{Result, anyhow};
	use ash::vk;
	use log::{trace, info, warn, error};
	use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
	use winit::window::Window;
	use nalgebra_glm as glm;

	const MAX_FRAMES_IN_FLIGHT: usize = 3;

	#[derive(Default, Clone)]
	pub struct Data
	{
		pub resized: bool,
		frame: usize,
		surface: vk::SurfaceKHR,
		pub physical_device: vk::PhysicalDevice,
		msaa_samples: vk::SampleCountFlags,
		pub graphics_queue: vk::Queue,
		transfer_queue: vk::Queue,
		presentation_queue: vk::Queue,
		swapchain_loader: Option<ash::extensions::khr::Swapchain>,
		swapchain: vk::SwapchainKHR,
		swapchain_images: Vec<vk::Image>,
		swapchain_format: vk::Format,
		swapchain_extent: vk::Extent2D,
		swapchain_image_views: Vec<vk::ImageView>,
		pub render_pass: vk::RenderPass,
		framebuffers: Vec<vk::Framebuffer>,
		pipeline_layout: vk::PipelineLayout,
		pipeline: vk::Pipeline,
		graphics_command_pools: Vec<vk::CommandPool>,
		pub graphics_command_pool: vk::CommandPool,
		transfer_command_pool: vk::CommandPool,
		graphics_command_buffers: Vec<vk::CommandBuffer>,
		in_flight_fences: Vec<vk::Fence>,
		image_available_semaphores: Vec<vk::Semaphore>,
		render_finished_semaphores: Vec<vk::Semaphore>,
		images_in_flight: Vec<vk::Fence>,
		vertices: Vec<Vertex>,
		indices: Vec<u32>,
		vertex_buffer: vk::Buffer,
		vertex_buffer_memory: vk::DeviceMemory,
		index_buffer: vk::Buffer,
		index_buffer_memory: vk::DeviceMemory,
		uniform_buffers: Vec<vk::Buffer>,
		uniform_buffers_memory: Vec<vk::DeviceMemory>,
		descriptor_set_layout: vk::DescriptorSetLayout,
		descriptor_pool: vk::DescriptorPool,
		descriptor_sets: Vec<vk::DescriptorSet>,
		mip_levels: u32,
		texture_image: vk::Image,
		texture_image_memory: vk::DeviceMemory,
		texture_image_view: vk::ImageView,
		texture_sampler: vk::Sampler,
		depth_image: vk::Image,
		depth_image_memory: vk::DeviceMemory,
		depth_image_view: vk::ImageView,
		color_image: vk::Image,
		color_image_memory: vk::DeviceMemory,
		color_image_view: vk::ImageView,
		debug_utils: Option<ash::extensions::ext::DebugUtils>,
		messenger: Option<vk::DebugUtilsMessengerEXT>,
	}

	#[derive(Copy, Clone, Debug)]
	struct QueueFamilyIndices
	{
		graphics: u32,
		transfer: u32,
		presentation: u32,
	}

	impl QueueFamilyIndices
	{
		fn get(
			instance: &ash::Instance,
			physical_device: vk::PhysicalDevice,
			surface: vk::SurfaceKHR,
			surface_loader: &ash::extensions::khr::Surface,
			) -> Result<Self>
		{
			let properties = unsafe {instance.get_physical_device_queue_family_properties(physical_device)};

			let graphics = properties
				.iter()
				.position(|properties| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
				.map(|index| index as u32);

			let mut transfer = properties
				.iter()
				.position(|properties| properties.queue_flags.contains(vk::QueueFlags::TRANSFER)
					&& !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
				.map(|index| index as u32);

			// TODO come up with better way to handle this
			if transfer.is_none()
			{
				transfer = graphics;
			}

			let mut presentation = None;
			for(index, _properties) in properties.iter().enumerate()
			{
				if unsafe {surface_loader.get_physical_device_surface_support
					(
						physical_device,
						index as u32,
						surface,
					)?}
				{
					presentation = Some(index as u32);
					break;
				}
			}

			if let (Some(graphics), Some(transfer), Some(presentation)) = (graphics, transfer, presentation)
			{
				Ok(Self {graphics, transfer, presentation})
			}
			else
			{
				Err(anyhow!("Missing required queue families"))
			}
		}
	}

	pub fn create_instance(entry: &ash::Entry, window: &Window, enable_validation: bool, data: &mut Data, app_name: &str) -> Result<ash::Instance>
	{
		let engine_name = std::ffi::CString::new("Goop Engine")?;
		let app_name = std::ffi::CString::new(app_name)?;

		let app_info = vk::ApplicationInfo::builder()
			.application_name(&app_name)
			.engine_name(&engine_name)
			.engine_version(vk::make_api_version(0, 0, 0, 0))
			.application_version(vk::make_api_version(0, 0, 0, 0))
			.api_version(vk::make_api_version(0, 1, 0, 106));

		let layer_names: Vec<std::ffi::CString> = 
			if enable_validation
			{
				vec![std::ffi::CString::new("VK_LAYER_KHRONOS_validation")?]
			}
			else
			{
				vec![]
			};

		let layer_name_ptrs: Vec<*const i8> = 
			if enable_validation
			{
				layer_names
					.iter()
					.map(|layer_name| layer_name.as_ptr())
					.collect()
			}
			else
			{
				vec![]
			};

		let mut extension_name_ptrs: Vec<*const i8> =
			ash_window::enumerate_required_extensions(window.raw_display_handle())?.to_vec();

		if enable_validation
		{
			extension_name_ptrs.push(ash::extensions::ext::DebugUtils::name().as_ptr());
		}

		let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
				.message_severity(
					vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
					| vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
					| vk::DebugUtilsMessageSeverityFlagsEXT::INFO
					| vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
				.message_type(
					vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
					| vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
					| vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
				.pfn_user_callback(Some(vulkan_debug_utils_callback));

		let mut instance_info = vk::InstanceCreateInfo::builder()
			.application_info(&app_info)
			.enabled_layer_names(&layer_name_ptrs)
			.enabled_extension_names(&extension_name_ptrs);

		if enable_validation
		{
			instance_info = instance_info.push_next(&mut debug_info);
		}
		let instance = unsafe { entry.create_instance(&instance_info, None)? };

		let mut debug_utils: Option<ash::extensions::ext::DebugUtils> = None;
		let mut messenger: Option<vk::DebugUtilsMessengerEXT> = None;

		if enable_validation
		{
			debug_utils = Some(ash::extensions::ext::DebugUtils::new(&entry, &instance));
			messenger = unsafe { Some(debug_utils.as_ref().unwrap().create_debug_utils_messenger(&debug_info, None)?) };
		}

		data.debug_utils = debug_utils;
		data.messenger = messenger;

		Ok(instance)
	}

	unsafe extern "system" fn vulkan_debug_utils_callback(
		severity: vk::DebugUtilsMessageSeverityFlagsEXT,
		type_: vk::DebugUtilsMessageTypeFlagsEXT,
		data: *const vk::DebugUtilsMessengerCallbackDataEXT,
		_p_user_data: *mut std::ffi::c_void,
	) -> vk::Bool32
	{
		let data = unsafe { *data };
		let message = unsafe { std::ffi::CStr::from_ptr(data.p_message) }.to_string_lossy();

		if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
		{
			error!("({:?}) {}", type_, message);
		}
		else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
		{
			warn!("({:?}) {}", type_, message);
		}
		else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO
		{
			info!("({:?}) {}", type_, message);
		}
		else
		{
			trace!("({:?}) {}", type_, message);
		}

		// Should we skip the call to the driver?
		vk::FALSE
	}

	fn get_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice>
	{
		let phys_devices = unsafe { instance.enumerate_physical_devices()? };
		let physical_device =
		{
			let mut chosen = Err(anyhow!("no appropriate physical device available"));
			for pd in phys_devices
			{
				chosen = Ok(pd);
				//let props = unsafe { instance.get_physical_device_properties(pd) };

				//TODO figure out better way
				/*
				if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
				{
					chosen = Ok(pd);
				}
				*/
			}
			chosen?
		};

		Ok(physical_device)
	}

	pub fn create_logical_device(instance: &ash::Instance, surface_loader: &ash::extensions::khr::Surface, data: &mut Data) -> Result<ash::Device>
	{
		let physical_device = get_physical_device(instance)?;
		let indices = QueueFamilyIndices::get(instance, physical_device, data.surface, surface_loader)?;
		let priorities = [1.0f32];
		let g_info = vk::DeviceQueueCreateInfo::builder()
						.queue_family_index(indices.graphics)
						.queue_priorities(&priorities);

		let t_info = vk::DeviceQueueCreateInfo::builder()
						.queue_family_index(indices.transfer)
						.queue_priorities(&priorities);

		let p_info = vk::DeviceQueueCreateInfo::builder()
						.queue_family_index(indices.presentation)
						.queue_priorities(&priorities);

		let mut queue_infos = if indices.graphics == indices.presentation
		{
			vec![*g_info]
		}
		else
		{
			vec![*g_info, *p_info]
		};

		if indices.graphics != indices.transfer
		{
			queue_infos.push(*t_info);
		}

		let enabled_extension_name_ptrs =
			vec![ash::extensions::khr::Swapchain::name().as_ptr()];
		let features = vk::PhysicalDeviceFeatures::builder()
			.sampler_anisotropy(true)
			// Enable sample shading feature (aa on texture)
			.sample_rate_shading(false);

		let device_info = vk::DeviceCreateInfo::builder()
			.enabled_features(&features)
			.enabled_extension_names(&enabled_extension_name_ptrs)
			.queue_create_infos(&queue_infos);
		let logical_device = unsafe { instance.create_device(physical_device, &device_info, None)? };
		let graphics_queue = unsafe { logical_device.get_device_queue(indices.graphics, 0) };
		let transfer_queue = unsafe { logical_device.get_device_queue(indices.transfer, 0) };
		let presentation_queue = unsafe { logical_device.get_device_queue(indices.presentation, 0) }; 

		data.physical_device = physical_device;
		data.graphics_queue = graphics_queue;
		data.transfer_queue = transfer_queue;
		data.presentation_queue = presentation_queue;

		Ok(logical_device)
	}

	pub fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &Window, data: &mut Data) -> Result<ash::extensions::khr::Surface>
	{
		let surface = unsafe {
			ash_window::create_surface(
				&entry,
				instance,
				window.raw_display_handle(),
				window.raw_window_handle(),
				None,
			)?
		};

		data.surface = surface;

		let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
		Ok(surface_loader)
	}

	fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR
	{
		formats
			.iter()
			.cloned()
			.find(|f|
				{
					f.format == vk::Format::B8G8R8A8_SRGB
								&& f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
				})
			.unwrap_or_else(|| formats[0])
	}

	fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR
	{
		present_modes
			.iter()
			.cloned()
			.find(|mode|
				{
					*mode == vk::PresentModeKHR::MAILBOX //triple buffering
				})
			.unwrap_or(vk::PresentModeKHR::FIFO)
	}
	
	fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D
	{
		if capabilities.current_extent.width != u32::max_value()
		{
			capabilities.current_extent
		}
		else
		{
			let size = window.inner_size();
			let clamp = |min: u32, max: u32, value: u32| min.max(max.min(value));
			vk::Extent2D::builder()
				.width(clamp(
						capabilities.min_image_extent.width,
						capabilities.max_image_extent.width,
						size.width
				))
				.height(clamp(
						capabilities.min_image_extent.height,
						capabilities.max_image_extent.height,
						size.height
				))
				.build()
		}
	}

	pub fn create_swapchain(instance: &ash::Instance, device: &ash::Device, surface_loader: &ash::extensions::khr::Surface, window: &Window, data: &mut Data) -> Result<()>
	{
		let surface_capabilities = unsafe
		{
			surface_loader.get_physical_device_surface_capabilities(data.physical_device, data.surface)?
		};
		let surface_present_modes = unsafe
		{
			surface_loader.get_physical_device_surface_present_modes(data.physical_device, data.surface)?
		};
		let surface_formats = unsafe
		{
			surface_loader.get_physical_device_surface_formats(data.physical_device, data.surface)?
		};

		let surface_present_mode = get_swapchain_present_mode(&surface_present_modes);
		let surface_format = get_swapchain_surface_format(&surface_formats);
		let swapchain_extent = get_swapchain_extent(window, surface_capabilities);

		// simply sticking to this minimum means that we may sometimes have to wait on the 
		// driver to complete internal operations before we can acquire another image to render to.
		// Therefore it is recommended to request at least one more image than the minimum
		let mut image_count = surface_capabilities.min_image_count + 1;

		if surface_capabilities.max_image_count != 0
			&& image_count > surface_capabilities.max_image_count
		{
			image_count = surface_capabilities.max_image_count;
		}

		let indices = QueueFamilyIndices::get(instance, data.physical_device, data.surface, surface_loader)?;

		let mut queue_family_indices = vec![];
		let image_sharing_mode = if indices.graphics != indices.presentation && indices.graphics != indices.transfer
			{
				queue_family_indices.push(indices.graphics);
				queue_family_indices.push(indices.presentation);
				queue_family_indices.push(indices.transfer);
				vk::SharingMode::EXCLUSIVE
			}
			else if indices.graphics != indices.presentation
			{
				queue_family_indices.push(indices.graphics);
				queue_family_indices.push(indices.presentation);
				vk::SharingMode::CONCURRENT
			}
			else
			{
				queue_family_indices.push(indices.graphics);
				vk::SharingMode::EXCLUSIVE
			};

		let info = vk::SwapchainCreateInfoKHR::builder()
			.surface(data.surface)
			.min_image_count(image_count)
			.image_format(surface_format.format)
			.image_color_space(surface_format.color_space)
			.image_extent(swapchain_extent)
			.present_mode(surface_present_mode)
			.image_array_layers(1)
			.image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
			.image_sharing_mode(image_sharing_mode)
			.queue_family_indices(&queue_family_indices)
			.pre_transform(surface_capabilities.current_transform)
			.composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
			.clipped(true)
			.old_swapchain(vk::SwapchainKHR::null());

		let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, device);
		let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None)? };

		data.swapchain = swapchain;
		data.swapchain_loader = Some(swapchain_loader);
		data.swapchain_format = surface_format.format;
		data.swapchain_extent = swapchain_extent;

		Ok(())
	}

	pub fn create_swapchain_image_views(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let swapchain_images = unsafe { data.swapchain_loader.as_ref().unwrap().get_swapchain_images(data.swapchain)? };
		let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());

		for image in &swapchain_images
		{
			let subresource = vk::ImageSubresourceRange::builder()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.base_mip_level(0)
				.level_count(1)
				.base_array_layer(0)
				.layer_count(1);
			let imageview_info = vk::ImageViewCreateInfo::builder()
				.image(*image)
				.view_type(vk::ImageViewType::TYPE_2D)
				.format(data.swapchain_format)
				.subresource_range(*subresource);
			let image_view = unsafe { device.create_image_view(&imageview_info, None)? };
			swapchain_image_views.push(image_view);
		}

		data.swapchain_images = swapchain_images;
		data.swapchain_image_views = swapchain_image_views;

		Ok(())
	}

	pub fn create_render_pass(instance: &ash::Instance, device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let color_attachment = vk::AttachmentDescription::builder()
			.format(data.swapchain_format)
			.samples(vk::SampleCountFlags::TYPE_1)
			.load_op(vk::AttachmentLoadOp::CLEAR)
			.store_op(vk::AttachmentStoreOp::STORE)
			.stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
			.stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
			.initial_layout(vk::ImageLayout::UNDEFINED)
			.final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

		let color_attachment_ref = vk::AttachmentReference::builder()
			.attachment(0)
			.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

		let color_attachments = &[*color_attachment_ref];

		let depth_stencil_attachment = vk::AttachmentDescription::builder()
			.format(unsafe { get_depth_format(instance, data)? })
			.samples(vk::SampleCountFlags::TYPE_1)
			.load_op(vk::AttachmentLoadOp::CLEAR)
			.store_op(vk::AttachmentStoreOp::DONT_CARE)
			.stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
			.stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
			.initial_layout(vk::ImageLayout::UNDEFINED)
			.final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

		let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
			.attachment(1)
			.layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

		let subpass = vk::SubpassDescription::builder()
			.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
			.depth_stencil_attachment(&depth_stencil_attachment_ref)
			.color_attachments(color_attachments);

		let dependency = vk::SubpassDependency::builder()
			.src_subpass(vk::SUBPASS_EXTERNAL)
			.dst_subpass(0)
			.src_stage_mask(
				vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
				| vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
				)
			.dst_stage_mask(
				vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
				| vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
				)
			.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE
				| vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
				);

		let attachments = &[*color_attachment, *depth_stencil_attachment];
		let subpasses = &[*subpass];
		let dependencies = &[*dependency];

		let info = vk::RenderPassCreateInfo::builder()
			.attachments(attachments)
			.subpasses(subpasses)
			.dependencies(dependencies);

		data.render_pass = unsafe { device.create_render_pass(&info, None)? };

		Ok(())
	}

	pub fn create_framebuffers(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		data.framebuffers = data.swapchain_image_views
			.iter()
			.map(|image_view|
				{
					let attachments = &[*image_view, data.depth_image_view];
					let info = vk::FramebufferCreateInfo::builder()
						.render_pass(data.render_pass)
						.attachments(attachments)
						.width(data.swapchain_extent.width)
						.height(data.swapchain_extent.height)
						.layers(1);
					unsafe { device.create_framebuffer(&info, None) }
				})
			.collect::<Result<Vec<_>,_>>()?;
		Ok(())
	}

	unsafe fn create_shader_module(device: &ash::Device, bytecode: &[u8]) -> Result<vk::ShaderModule>
	{
		let bytecode = Vec::<u8>::from(bytecode);
		let (prefix, code, suffix) = bytecode.align_to::<u32>();
		if !prefix.is_empty() || !suffix.is_empty()
		{
			return Err(anyhow!("Shader bytecode not properly aligned"));
		}

		let info = vk::ShaderModuleCreateInfo::builder()
			.code(code);

		Ok(device.create_shader_module(&info, None)?)
	}

	#[repr(C)]
	#[derive(Copy, Clone, Debug)]
	struct Vertex
	{
		pos: glm::Vec3,
		color: glm::Vec3,
		tex_coord: glm::Vec2,
	}

	impl Vertex
	{
		/*
		fn new(pos: glm::Vec3, color: glm::Vec3, tex_coord: glm::Vec2) -> Self
		{
			Self { pos, color, tex_coord, }
		}
		*/

		fn binding_description() -> vk::VertexInputBindingDescription
		{
			vk::VertexInputBindingDescription::builder()
				.binding(0)
				.stride(size_of::<Vertex>() as u32)
				.input_rate(vk::VertexInputRate::VERTEX)
				.build()
		}

		fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3]
		{
			let pos = vk::VertexInputAttributeDescription::builder()
				.binding(0)
				.location(0)
				.format(vk::Format::R32G32B32_SFLOAT)
				.offset(0)
				.build();

			let color = vk::VertexInputAttributeDescription::builder()
				.binding(0)
				.location(1)
				.format(vk::Format::R32G32B32_SFLOAT)
				.offset(size_of::<glm::Vec3>() as u32)
				.build();

			let tex_coord = vk::VertexInputAttributeDescription::builder()
				.binding(0)
				.location(2)
				.format(vk::Format::R32G32_SFLOAT)
				.offset((size_of::<glm::Vec3>() + size_of::<glm::Vec3>()) as u32)
				.build();

			[pos, color, tex_coord]
		}
	}

	impl PartialEq for Vertex
	{
		fn eq(&self, other: &Self) -> bool {
			self.pos == other.pos
				&& self.color == other.color
				&& self.tex_coord == other.tex_coord
		}
	}

	impl Eq for Vertex
	{
	}

	impl Hash for Vertex
	{
		fn hash<H: Hasher>(&self, state: &mut H) {
			self.pos[0].to_bits().hash(state);
			self.pos[1].to_bits().hash(state);
			self.pos[2].to_bits().hash(state);
			self.color[0].to_bits().hash(state);
			self.color[1].to_bits().hash(state);
			self.color[2].to_bits().hash(state);
			self.tex_coord[0].to_bits().hash(state);
			self.tex_coord[1].to_bits().hash(state);
		}
	}

	pub fn create_pipeline(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		// TODO find a better way to handle loading shaders
		let vert = include_bytes!("../../../shaders/vert.spv");
		let frag = include_bytes!("../../../shaders/frag.spv");

		let vert_sm = unsafe { create_shader_module(device, vert)? } ;
		let frag_sm = unsafe { create_shader_module(device, frag)? } ;

		let entry_func_name = CString::new("main").unwrap();

		let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::VERTEX)
			.module(vert_sm)
			.name(&entry_func_name);

		let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
			.stage(vk::ShaderStageFlags::FRAGMENT)
			.module(frag_sm)
			.name(&entry_func_name);

		let stages = &[*vert_stage, *frag_stage];

		let binding_descriptions = &[Vertex::binding_description()];
		let attribute_descriptions = Vertex::attribute_descriptions();
		let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
			.vertex_binding_descriptions(binding_descriptions)
			.vertex_attribute_descriptions(&attribute_descriptions);

		let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
			.topology(vk::PrimitiveTopology::TRIANGLE_LIST)
			.primitive_restart_enable(false);

		let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
			.depth_test_enable(true)
			.depth_write_enable(true)
			.depth_compare_op(vk::CompareOp::LESS)
			.depth_bounds_test_enable(false);

		let viewport = vk::Viewport::builder()
			.x(0.0)
			.y(0.0)
			.width(data.swapchain_extent.width as f32)
			.height(data.swapchain_extent.height as f32)
			.min_depth(0.0)
			.max_depth(1.0);
		let viewports = &[*viewport];

		let scissor = vk::Rect2D::builder()
			.offset(vk::Offset2D {x: 0, y:0 })
			.extent(data.swapchain_extent);
		let scissors = &[*scissor];

		let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
			.viewports(viewports)
			.scissors(scissors);

		let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
			.line_width(1.0)
			.front_face(vk::FrontFace::CLOCKWISE)
			.cull_mode(vk::CullModeFlags::BACK)
			.polygon_mode(vk::PolygonMode::FILL);

		let multisampler_info = vk::PipelineMultisampleStateCreateInfo::builder()
			// enable sample shading
			//.sample_shading_enable(true)
			//.min_sample_shading(0.2)
			.sample_shading_enable(false)
			.rasterization_samples(data.msaa_samples);

		let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
			.color_write_mask(vk::ColorComponentFlags::R
				| vk::ColorComponentFlags::G
				| vk::ColorComponentFlags::B
				| vk::ColorComponentFlags::A
				)
			.blend_enable(true)
			.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.color_blend_op(vk::BlendOp::ADD)
			.src_alpha_blend_factor(vk::BlendFactor::ONE)
			.dst_alpha_blend_factor(vk::BlendFactor::ZERO)
			.alpha_blend_op(vk::BlendOp::ADD);
		let blend_attachments = &[*color_blend_attachment];

		let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
			.logic_op_enable(false)
			.logic_op(vk::LogicOp::COPY)
			.attachments(blend_attachments)
			.blend_constants([0.0,0.0,0.0,0.0]);

		let vert_push_constant_range = vk::PushConstantRange::builder()
			.stage_flags(vk::ShaderStageFlags::VERTEX)
			.offset(0)
			.size(64 /* 16 x 4 byte floats */);

		let frag_push_constant_range = vk::PushConstantRange::builder()
			.stage_flags(vk::ShaderStageFlags::FRAGMENT)
			.offset(64)
			.size(4);

		let set_layouts = &[data.descriptor_set_layout];
		let push_constant_ranges = &[*vert_push_constant_range, *frag_push_constant_range];
		let layout_info = vk::PipelineLayoutCreateInfo::builder()
			.set_layouts(set_layouts)
			.push_constant_ranges(push_constant_ranges);
		data.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

		let info = vk::GraphicsPipelineCreateInfo::builder()
			.stages(stages)
			.vertex_input_state(&vertex_input_info)
			.input_assembly_state(&input_assembly_info)
			.viewport_state(&viewport_info)
			.rasterization_state(&rasterizer_info)
			.multisample_state(&multisampler_info)
			.color_blend_state(&color_blend_state)
			.depth_stencil_state(&depth_stencil_state)
			.layout(data.pipeline_layout)
			.render_pass(data.render_pass)
			.subpass(0);

		data.pipeline = unsafe { device
			.create_graphics_pipelines(
				vk::PipelineCache::null(),
				&[*info],
				None,
				).expect("Pipeline creation failed!")
		}[0];

		unsafe
		{
			device.destroy_shader_module(vert_sm, None);
			device.destroy_shader_module(frag_sm, None);
		}

		Ok(())
	}
	unsafe fn create_command_pool(
		device: &ash::Device,
		queue_family_index: u32,
		) -> Result<vk::CommandPool>
	{
		let info = vk::CommandPoolCreateInfo::builder()
			.flags(vk::CommandPoolCreateFlags::TRANSIENT)
			.queue_family_index(queue_family_index);

		Ok(device.create_command_pool(&info, None)?)
	}

	pub fn create_command_pools(instance: &ash::Instance, device: &ash::Device, surface_loader: &ash::extensions::khr::Surface, data: &mut Data) -> Result<()>
	{
		let indices = QueueFamilyIndices::get(instance, data.physical_device, data.surface, surface_loader)?;

		data.graphics_command_pool = unsafe { create_command_pool(device, indices.graphics)? };
		data.transfer_command_pool = unsafe { create_command_pool(device, indices.transfer)? };

		let num_images = data.swapchain_images.len();
		for _ in 0..num_images
		{
			let g_command_pool = unsafe { create_command_pool(device, indices.graphics)? };
			data.graphics_command_pools.push(g_command_pool);
		}

		Ok(())
	}

	unsafe fn create_image(
		instance: &ash::Instance,
		device: &ash::Device,
		data: &Data,
		width: u32,
		height: u32,
		mip_levels: u32,
		samples: vk::SampleCountFlags,
		format: vk::Format,
		tiling: vk::ImageTiling,
		usage: vk::ImageUsageFlags,
		properties: vk::MemoryPropertyFlags,
		) -> Result<(vk::Image, vk::DeviceMemory)>
	{
		let info = vk::ImageCreateInfo::builder()
			.image_type(vk::ImageType::TYPE_2D)
			.extent(vk::Extent3D {width, height, depth: 1})
			.mip_levels(mip_levels)
			.samples(samples)
			.array_layers(1)
			.format(format)
			.tiling(tiling)
			.initial_layout(vk::ImageLayout::UNDEFINED)
			.usage(usage)
			//TODO This could cause problems if we need to use both
			//graphics and transfer queue families
			.sharing_mode(vk::SharingMode::EXCLUSIVE);

		let image = device.create_image(&info, None)?;

		let requirements = device.get_image_memory_requirements(image);

		let info = vk::MemoryAllocateInfo::builder()
			.allocation_size(requirements.size)
			.memory_type_index(get_memory_type_index(
					instance,
					data,
					properties,
					requirements,
					)?);
		
		let texture_image_memory = device.allocate_memory(&info, None)?;
		device.bind_image_memory(image, texture_image_memory, 0)?;

		Ok((image, texture_image_memory))
	}

	unsafe fn transition_image_layout(
		device: &ash::Device,
		data: &Data,
		image: vk::Image,
		old_layout: vk::ImageLayout,
		new_layout: vk::ImageLayout,
		mip_levels: u32,
		) -> Result<()>
	{
		let (
			src_access_mask,
			dst_access_mask,
			src_stage_mask,
			dst_stage_mask,
		) = match (old_layout, new_layout)
		{
			(vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) =>
			{
				(
					vk::AccessFlags::empty(),
					vk::AccessFlags::TRANSFER_WRITE,
					vk::PipelineStageFlags::TOP_OF_PIPE,
					vk::PipelineStageFlags::TRANSFER,
				)
			},
			(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) =>
			{
				(
					vk::AccessFlags::TRANSFER_WRITE,
					vk::AccessFlags::SHADER_READ,
					vk::PipelineStageFlags::TRANSFER,
					vk::PipelineStageFlags::FRAGMENT_SHADER,
				)
			},
			_ => return Err(anyhow!("ImageLayout transition not supported")),
		};

		let command_buffer = begin_single_time_commands(device, data.graphics_command_pool)?;

		let subresource = vk::ImageSubresourceRange::builder()
			.aspect_mask(vk::ImageAspectFlags::COLOR)
			.base_mip_level(0)
			.level_count(mip_levels)
			.base_array_layer(0)
			.layer_count(1);

		let barrier = vk::ImageMemoryBarrier::builder()
			.old_layout(old_layout)
			.new_layout(new_layout)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.image(image)
			.subresource_range(*subresource)
			.src_access_mask(src_access_mask)
			.dst_access_mask(dst_access_mask);

		device.cmd_pipeline_barrier(
			command_buffer,
			src_stage_mask,
			dst_stage_mask,
			vk::DependencyFlags::empty(),
			&[] as &[vk::MemoryBarrier],
			&[] as &[vk::BufferMemoryBarrier],
			&[*barrier],
		);
		

		end_single_time_commands(
			device,
			command_buffer,
			data.graphics_queue,
			data.graphics_command_pool,
		)?;
		Ok(())
	}

	unsafe fn copy_buffer_to_image(
		device: &ash::Device,
		data: &Data,
		buffer: vk::Buffer,
		image: vk::Image,
		width: u32,
		height: u32,
		) -> Result<()>
	{
		let command_buffer = begin_single_time_commands(device, data.transfer_command_pool)?;

		let subresource = vk::ImageSubresourceLayers::builder()
			.aspect_mask(vk::ImageAspectFlags::COLOR)
			.mip_level(0)
			.base_array_layer(0)
			.layer_count(1);

		let region = vk::BufferImageCopy::builder()
			.buffer_offset(0)
			.buffer_row_length(0)
			.buffer_image_height(0)
			.image_subresource(*subresource)
			.image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
			.image_extent(vk::Extent3D { width, height, depth: 1 } );

		device.cmd_copy_buffer_to_image(
			command_buffer,
			buffer,
			image,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			&[*region],
		);

		end_single_time_commands(
			device,
			command_buffer,
			data.transfer_queue,
			data.transfer_command_pool,
		)?;

		Ok(())
	}

	unsafe fn generate_mipmaps(
		instance: &ash::Instance,
		device: &ash::Device,
		data: &Data,
		image: vk::Image,
		format: vk::Format,
		width: u32,
		height: u32,
		mip_levels: u32,
		) -> Result<()>
	{
		if !instance
			.get_physical_device_format_properties(data.physical_device, format)
			.optimal_tiling_features
			.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
		{
			return Err(anyhow!("Linear blitting not supported by texture image format"));
		}

		let command_buffer = begin_single_time_commands(device, data.graphics_command_pool)?;

		let subresource = vk::ImageSubresourceRange::builder()
			.aspect_mask(vk::ImageAspectFlags::COLOR)
			.base_array_layer(0)
			.layer_count(1)
			.level_count(1);

		let mut barrier = vk::ImageMemoryBarrier::builder()
			.image(image)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.subresource_range(*subresource);

		let mut mip_width = width;
		let mut mip_height = height;

		for i in 1..mip_levels
		{
			barrier.subresource_range.base_mip_level = i - 1;
			barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
			barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
			barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
			barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::TRANSFER,
				vk::DependencyFlags::empty(),
				&[] as &[vk::MemoryBarrier],
				&[] as &[vk::BufferMemoryBarrier],
				&[*barrier],
			);

			let src_subresource = vk::ImageSubresourceLayers::builder()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.mip_level(i - 1)
				.base_array_layer(0)
				.layer_count(1);

			let dst_subresource = vk::ImageSubresourceLayers::builder()
				.aspect_mask(vk::ImageAspectFlags::COLOR)
				.mip_level(i)
				.base_array_layer(0)
				.layer_count(1);

			let blit = vk::ImageBlit::builder()
				.src_offsets([
					vk::Offset3D { x: 0, y: 0, z: 0 },
					vk::Offset3D 
					{
						x: mip_width as i32,
						y: mip_height as i32,
						z: 1,
					},
				])
				.src_subresource(*src_subresource)
				.dst_offsets([
					vk::Offset3D { x: 0, y: 0, z: 0 },
					vk::Offset3D 
					{
						x: (if mip_width > 1 { mip_width / 2 } else { 1 } ) as i32,
						y: (if mip_height > 1 { mip_height / 2 } else { 1 } ) as i32,
						z: 1,
					},
				])
				.dst_subresource(*dst_subresource);

			device.cmd_blit_image(
				command_buffer,
				image,
				vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
				image,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				&[*blit],
				vk::Filter::LINEAR,
			);

			barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
			barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
			barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
			barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

			device.cmd_pipeline_barrier(
				command_buffer,
				vk::PipelineStageFlags::TRANSFER,
				vk::PipelineStageFlags::FRAGMENT_SHADER,
				vk::DependencyFlags::empty(),
				&[] as &[vk::MemoryBarrier],
				&[] as &[vk::BufferMemoryBarrier],
				&[*barrier],
			);

			if mip_width > 1
			{
				mip_width /= 2;
			}

			if mip_height > 1
			{
				mip_height /= 2;
			}
		}

		barrier.subresource_range.base_mip_level = mip_levels - 1;
		barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
		barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
		barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

		device.cmd_pipeline_barrier(
			command_buffer,
			vk::PipelineStageFlags::TRANSFER,
			vk::PipelineStageFlags::FRAGMENT_SHADER,
			vk::DependencyFlags::empty(),
			&[] as &[vk::MemoryBarrier],
			&[] as &[vk::BufferMemoryBarrier],
			&[*barrier],
		);

		end_single_time_commands(device,
			command_buffer,
			data.graphics_queue,
			data.graphics_command_pool
		)?;


		Ok(())
	}

	pub fn create_texture_image(instance: &ash::Instance, device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let image = File::open("media/textures/viking_room.png")?;

		let decoder = png::Decoder::new(image);
		let mut reader = decoder.read_info()?;

		//TODO handle png images that don't have an alpha channel
		if reader.info().color_type != png::ColorType::Rgba
		{
			panic!("Invalid texture image. Make sure it has an alpha channel");
		}

		let mut pixels = vec![0; reader.info().raw_bytes()];
		reader.next_frame(&mut pixels)?;

		let size = reader.info().raw_bytes() as u64;

		let (width, height) = reader.info().size();

		data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

		unsafe
		{
			let (staging_buffer, staging_buffer_memory) = create_buffer(
				instance,
				device,
				data,
				size,
				vk::BufferUsageFlags::TRANSFER_SRC,
				vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
			)?;

			let memory = device.map_memory(
				staging_buffer_memory,
				0,
				size,
				vk::MemoryMapFlags::empty(),
				)?;

			memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

			device.unmap_memory(staging_buffer_memory);

			data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

			let(texture_image, texture_image_memory) = create_image(
				instance,
				device,
				data,
				width,
				height,
				data.mip_levels,
				vk::SampleCountFlags::TYPE_1,
				vk::Format::R8G8B8A8_SRGB,
				vk::ImageTiling::OPTIMAL,
				vk::ImageUsageFlags::SAMPLED
					| vk::ImageUsageFlags::TRANSFER_SRC
					| vk::ImageUsageFlags::TRANSFER_DST,
				vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

			data.texture_image = texture_image;
			data.texture_image_memory = texture_image_memory;

			transition_image_layout(
				device,
				data,
				data.texture_image,
				vk::ImageLayout::UNDEFINED,
				vk::ImageLayout::TRANSFER_DST_OPTIMAL,
				data.mip_levels,
			)?;

			copy_buffer_to_image(
				device,
				data,
				staging_buffer,
				data.texture_image,
				width,
				height,
			)?;

			device.destroy_buffer(staging_buffer, None);
			device.free_memory(staging_buffer_memory, None);

			generate_mipmaps(
				instance,
				device,
				data,
				data.texture_image,
				vk::Format::R8G8B8A8_SRGB,
				width,
				height,
				data.mip_levels,
			)?;
		}
		Ok(())
	}

	unsafe fn create_image_view(
		device: &ash::Device,
		image: vk::Image,
		format: vk::Format,
		aspects: vk::ImageAspectFlags,
		mip_levels: u32,
		) -> Result<vk::ImageView>
	{
		let subresource_range = vk::ImageSubresourceRange::builder()
			.aspect_mask(aspects)
			.base_mip_level(0)
			.level_count(mip_levels)
			.base_array_layer(0)
			.layer_count(1);

		let info = vk::ImageViewCreateInfo::builder()
			.image(image)
			.view_type(vk::ImageViewType::TYPE_2D)
			.format(format)
			.subresource_range(*subresource_range);

		Ok(device.create_image_view(&info, None)?)
	}

	pub fn create_texture_image_views(
		device: &ash::Device,
		data: &mut Data
		) -> Result<()>
	{
		data.texture_image_view = unsafe { create_image_view(
			device,
			data.texture_image,
			vk::Format::R8G8B8A8_SRGB,
			vk::ImageAspectFlags::COLOR,
			data.mip_levels,
		)?};


		Ok(())
	}

	pub fn create_texture_sampler(
		device: &ash::Device,
		data: &mut Data,
		) -> Result<()>
	{
		let info = vk::SamplerCreateInfo::builder()
			.mag_filter(vk::Filter::LINEAR)
			.min_filter(vk::Filter::LINEAR)
			.address_mode_u(vk::SamplerAddressMode::REPEAT)
			.address_mode_v(vk::SamplerAddressMode::REPEAT)
			.address_mode_w(vk::SamplerAddressMode::REPEAT)
			.anisotropy_enable(true)
			.max_anisotropy(16.0)
			.border_color(vk::BorderColor::INT_OPAQUE_BLACK)
			.unnormalized_coordinates(false)
			.compare_enable(false)
			.compare_op(vk::CompareOp::ALWAYS)
			.mipmap_mode(vk::SamplerMipmapMode::LINEAR)
			.mip_lod_bias(0.0)
			.min_lod(0.0)
			.max_lod(data.mip_levels as f32);

		data.texture_sampler = unsafe { device.create_sampler(&info, None)? } ;
		Ok(())
	}

	unsafe fn get_memory_type_index(
		instance: &ash::Instance,
		data: &Data,
		properties: vk::MemoryPropertyFlags,
		requirements: vk::MemoryRequirements,
		) -> Result<u32>
	{
		let memory = instance.get_physical_device_memory_properties(data.physical_device);

		(0..memory.memory_type_count)
			.find(|i|
				{
					let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
					let memory_type = memory.memory_types[*i as usize];
					suitable && memory_type.property_flags.contains(properties)
				})
			.ok_or_else(|| anyhow!("failed to find appropriate memory type"))
	}

	unsafe fn create_buffer(
		instance: &ash::Instance,
		device: &ash::Device,
		data: &Data,
		size: vk::DeviceSize,
		usage: vk::BufferUsageFlags,
		properties: vk::MemoryPropertyFlags,
		) -> Result<(vk::Buffer, vk::DeviceMemory)>
	{
		let buffer_info = vk::BufferCreateInfo::builder()
			.size(size)
			.usage(usage)
			.sharing_mode(vk::SharingMode::EXCLUSIVE);

		let buffer = device.create_buffer(&buffer_info, None)?;

		let requirements = device.get_buffer_memory_requirements(buffer);

		let memory_info = vk::MemoryAllocateInfo::builder()
			.allocation_size(requirements.size)
			.memory_type_index(get_memory_type_index(
					instance,
					data,
					properties,
					requirements
					)?);

		let buffer_memory = device.allocate_memory(&memory_info, None)?;

		device.bind_buffer_memory(buffer, buffer_memory, 0)?;

		Ok((buffer, buffer_memory))
	}

	unsafe fn begin_single_time_commands(
		device: &ash::Device,
		command_pool: vk::CommandPool,
		) -> Result<vk::CommandBuffer>
	{
		let info = vk::CommandBufferAllocateInfo::builder()
			.level(vk::CommandBufferLevel::PRIMARY)
			.command_pool(command_pool)
			.command_buffer_count(1);

		let command_buffer = device.allocate_command_buffers(&info)?[0];

		let info = vk::CommandBufferBeginInfo::builder()
			.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

		device.begin_command_buffer(command_buffer, &info)?;

		Ok(command_buffer)
	}

	unsafe fn end_single_time_commands(
		device: &ash::Device,
		command_buffer: vk::CommandBuffer,
		queue: vk::Queue,
		command_pool: vk::CommandPool,
		) -> Result<()>
	{
		device.end_command_buffer(command_buffer)?;

		let command_buffers = &[command_buffer];
		let info = vk::SubmitInfo::builder()
			.command_buffers(command_buffers);

		device.queue_submit(queue, &[*info], vk::Fence::null())?;
		device.queue_wait_idle(queue)?;
		device.free_command_buffers(command_pool, command_buffers);

		Ok(())
	}

	unsafe fn copy_buffer(
		device: &ash::Device,
		data: &mut Data,
		source: vk::Buffer,
		destination: vk::Buffer,
		size: vk::DeviceSize,
		) -> Result<()>
	{
		let command_buffer = begin_single_time_commands(device, data.transfer_command_pool)?;

		let regions = vk::BufferCopy::builder().size(size);
		device.cmd_copy_buffer(command_buffer, source, destination, &[*regions]);

		end_single_time_commands(
			device,
			command_buffer,
			data.transfer_queue,
			data.transfer_command_pool
		)?;

		Ok(())
	}

	pub fn create_vertex_buffer(instance: &ash::Instance, device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let size = (size_of::<Vertex>() * data.vertices.len()) as u64;

		unsafe {
			let (staging_buffer, staging_buffer_memory) = create_buffer(
				instance,
				device,
				data,
				size,
				vk::BufferUsageFlags::TRANSFER_SRC,
				vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
			)?;

			let memory = device.map_memory(
				staging_buffer_memory,
				0,
				size,
				vk::MemoryMapFlags::empty()
				)?;

				memcpy(data.vertices.as_ptr(), memory.cast(), data.vertices.len());
				device.unmap_memory(staging_buffer_memory);

			let (vertex_buffer, vertex_buffer_memory) = create_buffer(
				instance,
				device,
				data,
				size,
				vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
				vk::MemoryPropertyFlags::DEVICE_LOCAL,
			)?;

			data.vertex_buffer = vertex_buffer;
			data.vertex_buffer_memory = vertex_buffer_memory;

			copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

			device.destroy_buffer(staging_buffer, None);
			device.free_memory(staging_buffer_memory, None);
		}

		Ok(())
	}

	pub fn create_index_buffer(instance: &ash::Instance, device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let size = (size_of::<u32>() * data.indices.len()) as u64;

		unsafe
		{
			let (staging_buffer, staging_buffer_memory) = create_buffer(
				instance,
				device,
				data,
				size,
				vk::BufferUsageFlags::TRANSFER_SRC,
				vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
			)?;

			let memory = device.map_memory(
				staging_buffer_memory,
				0,
				size,
				vk::MemoryMapFlags::empty()
				)?;

			memcpy(data.indices.as_ptr(), memory.cast(), data.indices.len());

			device.unmap_memory(staging_buffer_memory);

			let (index_buffer, index_buffer_memory) = create_buffer(
				instance,
				device,
				data,
				size,
				vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
				vk::MemoryPropertyFlags::DEVICE_LOCAL
			)?;

			data.index_buffer = index_buffer;
			data.index_buffer_memory = index_buffer_memory;

			copy_buffer(device, data, staging_buffer, index_buffer, size)?;

			device.destroy_buffer(staging_buffer, None);
			device.free_memory(staging_buffer_memory, None);
		}

		Ok(())
	}

	pub fn create_uniform_buffers(
		instance: &ash::Instance,
		device: &ash::Device,
		data: &mut Data,
		) -> Result<()>
	{
		data.uniform_buffers.clear();
		data.uniform_buffers_memory.clear();

		for _ in 0..data.swapchain_images.len()
		{
			let (uniform_buffer, uniform_buffer_memory) = unsafe { create_buffer(
				instance,
				device,
				data,
				size_of::<UniformBufferObject>() as u64,
				vk::BufferUsageFlags::UNIFORM_BUFFER,
				vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
			)? };

			data.uniform_buffers.push(uniform_buffer);
			data.uniform_buffers_memory.push(uniform_buffer_memory);
		}

		Ok(())
	}

	#[repr(C)]
	#[derive(Copy, Clone, Debug)]
	struct UniformBufferObject
	{
		view: glm::Mat4,
		proj: glm::Mat4,
	}

	pub fn create_descriptor_pool(
		device: &ash::Device,
		data: &mut Data
		) -> Result<()>
	{
		let ubo_size = vk::DescriptorPoolSize::builder()
			.ty(vk::DescriptorType::UNIFORM_BUFFER)
			.descriptor_count(data.swapchain_images.len() as u32);

		let sampler_size = vk::DescriptorPoolSize::builder()
			.ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.descriptor_count(data.swapchain_images.len() as u32);

		let pool_sizes = &[*ubo_size, *sampler_size];
		let info = vk::DescriptorPoolCreateInfo::builder()
			.pool_sizes(pool_sizes)
			.max_sets(data.swapchain_images.len() as u32);

		data.descriptor_pool = unsafe { device.create_descriptor_pool(&info, None)? };
		Ok(())
	}

	pub fn create_descriptor_sets(
		device: &ash::Device,
		data: &mut Data,
		) -> Result<()>
	{
		let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
		let info = vk::DescriptorSetAllocateInfo::builder()
			.descriptor_pool(data.descriptor_pool)
			.set_layouts(&layouts);

		data.descriptor_sets = unsafe { device.allocate_descriptor_sets(&info)? };

		for i in 0..data.swapchain_images.len()
		{
			let info = vk::DescriptorBufferInfo::builder()
				.buffer(data.uniform_buffers[i])
				.offset(0)
				.range(size_of::<UniformBufferObject>() as u64);

			let buffer_info = &[*info];
			let ubo_write = vk::WriteDescriptorSet::builder()
				.dst_set(data.descriptor_sets[i])
				.dst_binding(0)
				.dst_array_element(0)
				.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
				.buffer_info(buffer_info);

			let info = vk::DescriptorImageInfo::builder()
				.image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
				.image_view(data.texture_image_view)
				.sampler(data.texture_sampler);

			let image_info = &[*info];
			let sampler_write = vk::WriteDescriptorSet::builder()
				.dst_set(data.descriptor_sets[i])
				.dst_binding(1)
				.dst_array_element(0)
				.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.image_info(image_info);

			unsafe { device.update_descriptor_sets(
				&[*ubo_write, *sampler_write],
				&[] as &[vk::CopyDescriptorSet]
			) };
		}
		Ok(())
	}

	pub fn create_descriptor_set_layout(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
			.binding(0)
			.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
			.descriptor_count(1)
			.stage_flags(vk::ShaderStageFlags::VERTEX);

		let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
			.binding(1)
			.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
			.descriptor_count(1)
			.stage_flags(vk::ShaderStageFlags::FRAGMENT);

		let bindings = &[*ubo_binding, *sampler_binding];
		let info = vk::DescriptorSetLayoutCreateInfo::builder()
			.bindings(bindings);

		data.descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };

		Ok(())
	}

	pub fn create_command_buffers(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let num_images = data.swapchain_images.len();
		for image_index in 0..num_images
		{
			let allocate_info = vk::CommandBufferAllocateInfo::builder()
				.command_pool(data.graphics_command_pools[image_index])
				.level(vk::CommandBufferLevel::PRIMARY)
				.command_buffer_count(1);

			let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info) }?[0];
			data.graphics_command_buffers.push(command_buffer);
		}

		Ok(())

	}

	pub fn create_sync_objects(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let semaphore_info = vk::SemaphoreCreateInfo::builder();
		let fence_info = vk::FenceCreateInfo::builder()
						.flags(vk::FenceCreateFlags::SIGNALED);
		for _ in 0..MAX_FRAMES_IN_FLIGHT
		{
			unsafe
			{
				data.image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
				data.render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
				data.in_flight_fences.push(device.create_fence(&fence_info, None)?);
			}
		}

		data.images_in_flight = data.swapchain_images.iter().map(|_| vk::Fence::null()).collect();

		Ok(())
	}

	fn update_uniform_buffer(device: &ash::Device, image_index: usize, data: &Data) -> Result<()>
	{
		let view = glm::look_at(
			&glm::vec3(0.0,0.0,8.0),
			&glm::vec3(0.0,0.0,0.0),
			&glm::vec3(0.0,1.0,0.0),
		);

		let proj = glm::perspective_rh_zo(
			data.swapchain_extent.width as f32 / data.swapchain_extent.height as f32,
			glm::radians(&glm::vec1(45.0))[0],
			0.1,
			10.0,
		);

		//proj[(1,1)] *= -1.0;

		let ubo = UniformBufferObject { view, proj };

		unsafe
		{
			let memory = device.map_memory(
				data.uniform_buffers_memory[image_index],
				0,
				size_of::<UniformBufferObject>() as u64,
				vk::MemoryMapFlags::empty(),
				)?;

			memcpy(&ubo, memory.cast(), 1);

			device.unmap_memory(data.uniform_buffers_memory[image_index]);
		}

		Ok(())
	}

	unsafe fn get_supported_format(
		instance: &ash::Instance,
		data: &Data,
		canditates: &[vk::Format],
		tiling: vk::ImageTiling,
		features: vk::FormatFeatureFlags,
		) -> Result<vk::Format>
	{
		canditates
			.iter()
			.cloned()
			.find(|f|
				{
					let properties = instance.get_physical_device_format_properties(
						data.physical_device,
						*f
					);
					match tiling
					{
						vk::ImageTiling::LINEAR =>
							properties.linear_tiling_features.contains(features),
						vk::ImageTiling::OPTIMAL =>
							properties.optimal_tiling_features.contains(features),
						_ => false,
					}
				})
			.ok_or_else(|| anyhow!("Failed to find supported format"))
	}

	unsafe fn get_depth_format(
		instance: &ash::Instance,
		data: &Data,
		) -> Result<vk::Format>
	{
		let candidates = &[
			vk::Format::D32_SFLOAT,
			vk::Format::D32_SFLOAT_S8_UINT,
			vk::Format::D24_UNORM_S8_UINT,
		];

		get_supported_format(
			instance,
			data,
			candidates,
			vk::ImageTiling::OPTIMAL,
			vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
		)
	}

	pub fn create_depth_objects(instance: &ash::Instance, device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let format = unsafe { get_depth_format(instance, data)? };
		let (depth_image, depth_image_memory) = unsafe { create_image(
			instance,
			device,
			data,
			data.swapchain_extent.width,
			data.swapchain_extent.height,
			1,
			data.msaa_samples,
			format,
			vk::ImageTiling::OPTIMAL,
			vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		)? } ;

		data.depth_image = depth_image;
		data.depth_image_memory = depth_image_memory;
		data.depth_image_view = unsafe { create_image_view(
			device,
			data.depth_image,
			format,
			vk::ImageAspectFlags::DEPTH,
			1,
		)? };
		Ok(())
	}

	pub fn load_model(data: &mut Data) -> Result<()>
	{
		let mut reader = BufReader::new(File::open("media/models/viking_room.obj")?);

		let (models, _) = tobj::load_obj_buf(
			&mut reader,
			&tobj::LoadOptions { triangulate: true, ..Default::default() },
			|_| Ok(Default::default()),
		)?;

		let mut unique_vertices = HashMap::new();

		for model in &models
		{
			for index in &model.mesh.indices
			{
				let pos_offset = (3 * index) as usize;
				let tex_coord_offset = (2 * index) as usize;

				let vertex = Vertex {
					pos: glm::vec3(
							 model.mesh.positions[pos_offset],
							 model.mesh.positions[pos_offset + 1],
							 model.mesh.positions[pos_offset + 2],
							 ),
					color: glm::vec3(1.0,1.0,1.0),
					tex_coord: glm::vec2(
						model.mesh.texcoords[tex_coord_offset],
						1.0 - model.mesh.texcoords[tex_coord_offset + 1],
						)
				};

				if let Some(index) = unique_vertices.get(&vertex)
				{
					data.indices.push(*index as u32);
				}
				else
				{
					let index = data.vertices.len();
					unique_vertices.insert(vertex, index);
					data.vertices.push(vertex);
					data.indices.push(index as u32);
				}
			}
		}
		Ok(())
	}

	pub fn set_msaa_samples(instance: &ash::Instance, data: &mut Data) -> Result<()>
	{
		data.msaa_samples = get_max_msaa_samples(instance, data);
		Ok(())
	}

	fn get_max_msaa_samples(
		instance: &ash::Instance,
		data: &Data,
		) -> vk::SampleCountFlags
	{
		return vk::SampleCountFlags::TYPE_1;
		let properties = unsafe { instance.get_physical_device_properties(data.physical_device) };
		let counts = properties.limits.framebuffer_color_sample_counts
			& properties.limits.framebuffer_depth_sample_counts;

		[
			vk::SampleCountFlags::TYPE_64,
			vk::SampleCountFlags::TYPE_32,
			vk::SampleCountFlags::TYPE_16,
			vk::SampleCountFlags::TYPE_8,
			vk::SampleCountFlags::TYPE_4,
			vk::SampleCountFlags::TYPE_2,
		]
		.iter()
		.cloned()
		.find(|count| counts.contains(*count))
		.unwrap_or(vk::SampleCountFlags::TYPE_1)
	}

	pub fn create_color_objects(
		instance: &ash::Instance,
		device: &ash::Device,
		data: &mut Data,
		) -> Result<()>
	{
		let (color_image, color_image_memory) = unsafe { create_image(
			instance,
			device,
			data,
			data.swapchain_extent.width,
			data.swapchain_extent.height,
			1,
			data.msaa_samples,
			data.swapchain_format,
			vk::ImageTiling::OPTIMAL,
			vk::ImageUsageFlags::COLOR_ATTACHMENT
				| vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
			vk::MemoryPropertyFlags::DEVICE_LOCAL,
		)?};

		data.color_image = color_image;
		data.color_image_memory = color_image_memory;

		data.color_image_view = unsafe { create_image_view(
			device,
			data.color_image,
			data.swapchain_format,
			vk::ImageAspectFlags::COLOR,
			1,
		)? };

		Ok(())
	}

	fn update_command_buffer(
		device: &ash::Device,
		image_index: usize,
		data: &mut Data,
		start: &std::time::Instant,
		#[cfg(feature = "goop_imgui")]
		renderer: &mut imgui_rs_vulkan_renderer::Renderer,
		#[cfg(feature = "goop_imgui")]
		draw_data: &imgui::DrawData,
		) -> Result<()>
	{
		let cp = data.graphics_command_pools[image_index];
		unsafe { device.reset_command_pool(cp, vk::CommandPoolResetFlags::empty())? };
		let cb = data.graphics_command_buffers[image_index];

		let time = start.elapsed().as_secs_f32();

		let model = glm::rotate(
			&glm::Mat4::identity(),
			glm::radians(&glm::vec1(90.0))[0],
			&glm::vec3(1.0, 0.0, 0.0),
		);

		let model = glm::rotate(
			&model,
			time,
			&glm::vec3(0.0, 0.0, 1.0),
		);

		let (_, model_bytes, _) = unsafe { model.as_slice().align_to::<u8>() };

		let opacity = (0.5 * start.elapsed().as_secs_f32()).sin().abs();

		let g_begin_info = vk::CommandBufferBeginInfo::builder();
		unsafe { device.begin_command_buffer(cb, &g_begin_info)? };

		let render_area = vk::Rect2D::builder()
			.offset(vk::Offset2D::default())
			.extent(data.swapchain_extent);

		let color_clear_value = vk::ClearValue {
			color: vk::ClearColorValue {
				float32: [0.0,0.0,0.0,1.0],
			}
		};
		
		let depth_clear_value = vk::ClearValue {
			depth_stencil: vk::ClearDepthStencilValue
				{
					depth: 1.0,
					stencil: 0,
				}
			};
		
		let clear_values = &[color_clear_value, depth_clear_value];

		let info = vk::RenderPassBeginInfo::builder()
			.render_pass(data.render_pass)
			.framebuffer(data.framebuffers[image_index])
			.render_area(*render_area)
			.clear_values(clear_values);

		unsafe
		{
			device.cmd_begin_render_pass(cb, &info, vk::SubpassContents::INLINE);

						device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
			device.cmd_bind_vertex_buffers(cb, 0, &[data.vertex_buffer], &[0]);
			device.cmd_bind_index_buffer(cb, data.index_buffer, 0, vk::IndexType::UINT32);
			device.cmd_bind_descriptor_sets(
				cb,
				vk::PipelineBindPoint::GRAPHICS,
				data.pipeline_layout,
				0,
				&[data.descriptor_sets[image_index]],
				&[],
			);
			device.cmd_push_constants(
				cb,
				data.pipeline_layout,
				vk::ShaderStageFlags::VERTEX,
				0,
				model_bytes,
			);
			device.cmd_push_constants(
				cb,
				data.pipeline_layout,
				vk::ShaderStageFlags::FRAGMENT,
				64,
				&opacity.to_ne_bytes()[..],
			);
			device.cmd_draw_indexed(cb, data.indices.len() as u32, 3, 0, 0, 0);

			#[cfg(feature = "goop_imgui")]
			renderer.cmd_draw(cb, draw_data)?;

			device.cmd_end_render_pass(cb);
			device.end_command_buffer(cb)?;
		}

		Ok(())
	}

	pub fn render(
		instance: &ash::Instance,
		device: &ash::Device,
		surface_loader: &ash::extensions::khr::Surface,
		window: &Window,
		data: &mut Data,
		start: &std::time::Instant,
		#[cfg(feature = "goop_imgui")]
		renderer: &mut imgui_rs_vulkan_renderer::Renderer,
		#[cfg(feature = "goop_imgui")]
		draw_data: &imgui::DrawData,
		) -> Result<()>
	{
		let swapchain_loader = data.swapchain_loader.clone().unwrap();
		let in_flight_fence = data.in_flight_fences[data.frame];

		unsafe { device.wait_for_fences(&[in_flight_fence], true, u64::max_value())? };

		let result = unsafe { swapchain_loader.acquire_next_image(
			data.swapchain,
			u64::max_value(),
			data.image_available_semaphores[data.frame],
			vk::Fence::null(),
			)
		};

		let image_index = match result
		{
			Ok((image_index, _)) => image_index as usize,
			Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => return recreate_swapchain(instance, device, surface_loader, window, data),
			Err(e) => return Err(anyhow!(e)),
		};

		let image_in_flight = data.images_in_flight[image_index];

		if image_in_flight != vk::Fence::null()
		{
			unsafe { device.wait_for_fences(&[image_in_flight], true, u64::max_value())? };
		}

		update_command_buffer(
			device,
			image_index,
			data,
			start,
			#[cfg(feature = "goop_imgui")]
			renderer,
			#[cfg(feature = "goop_imgui")]
			&draw_data,
		)?;
		update_uniform_buffer(device, image_index, data)?;

		let wait_semaphores = &[data.image_available_semaphores[data.frame]];
		let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
		let command_buffers = &[data.graphics_command_buffers[image_index]];
		let signal_semaphores = &[data.render_finished_semaphores[data.frame]];

		let submit_info = vk::SubmitInfo::builder()
			.wait_semaphores(wait_semaphores)
			.wait_dst_stage_mask(wait_stages)
			.command_buffers(command_buffers)
			.signal_semaphores(signal_semaphores);

		unsafe
		{
			device.reset_fences(&[in_flight_fence])?;
			device.queue_submit(data.graphics_queue, &[*submit_info], in_flight_fence)?;
		}

		let swapchains = &[data.swapchain];
		let image_indices = &[image_index as u32];
		let present_info = vk::PresentInfoKHR::builder()
			.wait_semaphores(signal_semaphores)
			.swapchains(swapchains)
			.image_indices(image_indices);

		unsafe
		{
			let result = swapchain_loader.queue_present(data.presentation_queue, &present_info);
			let changed = result == Err(vk::Result::SUBOPTIMAL_KHR) || result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
			if changed || data.resized
			{
				data.resized = false;
				recreate_swapchain(instance, device, surface_loader, window, data)?;
			}
			else if let Err(e) = result
			{
				return Err(anyhow!(e));
			}
		}

		data.frame = (data.frame + 1) % MAX_FRAMES_IN_FLIGHT;

		Ok(())
	}

	fn recreate_swapchain(instance: &ash::Instance, device: &ash::Device, surface_loader: &ash::extensions::khr::Surface, window: &Window, data: &mut Data) -> Result<()>
	{
		unsafe
		{
			device.device_wait_idle()?;
			destroy_swapchain(device, data);
		}

		create_swapchain(instance, device, surface_loader, window, data)?;
		create_swapchain_image_views(device, data)?;
		create_render_pass(instance, device, data)?;
		create_pipeline(device, data)?;
		create_color_objects(instance, device, data)?;
		create_depth_objects(instance, device, data)?;
		create_framebuffers(device, data)?;
		create_uniform_buffers(instance, device, data)?;
		create_descriptor_pool(device, data)?;
		create_descriptor_sets(device, data)?;
		create_command_buffers(device, data)?;
		data.images_in_flight
			.resize(data.swapchain_images.len(), vk::Fence::null());

		Ok(())
	}

	unsafe fn destroy_swapchain(device: &ash::Device, data: &Data)
	{
		device.destroy_image(data.color_image, None);
		device.destroy_image_view(data.color_image_view, None);
		device.free_memory(data.color_image_memory, None);
		device.destroy_image(data.depth_image, None);
		device.destroy_image_view(data.depth_image_view, None);
		device.free_memory(data.depth_image_memory, None);
		device.destroy_descriptor_pool(data.descriptor_pool, None);
		data.uniform_buffers
			.iter()
			.for_each(|ub| device.destroy_buffer(*ub, None));
		data.uniform_buffers_memory
			.iter()
			.for_each(|ub| device.free_memory(*ub, None));
		data.framebuffers
			.iter()
			.for_each(|fb|
			{
				device.destroy_framebuffer(*fb, None)
			});
		device.destroy_pipeline(data.pipeline, None);
		device.destroy_pipeline_layout(data.pipeline_layout, None);
		device.destroy_render_pass(data.render_pass, None);
		data.swapchain_image_views
			.iter()
			.for_each(|iv|
			{
				device.destroy_image_view(*iv, None)
			}
		);
		let swap_loader = data.swapchain_loader.as_ref().unwrap();
		swap_loader.destroy_swapchain(data.swapchain, None);
	}

	pub unsafe fn destroy(instance: &ash::Instance, device: &ash::Device, surface_loader: &ash::extensions::khr::Surface, data: &Data)
	{
		device.device_wait_idle().unwrap();
		destroy_swapchain(device, data);
		data.graphics_command_pools
			.iter()
			.for_each(|cp| device.destroy_command_pool(*cp, None));
		device.destroy_sampler(data.texture_sampler, None);
		device.destroy_image_view(data.texture_image_view, None);
		device.destroy_image(data.texture_image, None);
		device.free_memory(data.texture_image_memory, None);
		device.destroy_descriptor_set_layout(data.descriptor_set_layout, None);
		device.destroy_buffer(data.index_buffer, None);
		device.free_memory(data.index_buffer_memory, None);
		device.destroy_buffer(data.vertex_buffer, None);
		device.free_memory(data.vertex_buffer_memory, None);
		data.images_in_flight
			.iter()
			.for_each(|f| device.destroy_fence(*f, None));
		data.in_flight_fences
			.iter()
			.for_each(|f| device.destroy_fence(*f, None));
		data.render_finished_semaphores
			.iter()
			.for_each(|s| device.destroy_semaphore(*s, None));
		data.image_available_semaphores
			.iter()
			.for_each(|s| device.destroy_semaphore(*s, None));
		device.destroy_command_pool(data.graphics_command_pool, None);
		device.destroy_command_pool(data.transfer_command_pool, None);

		#[cfg(not(feature = "goop_imgui"))]
		device.destroy_device(None);
		#[cfg(not(feature = "goop_imgui"))]
		surface_loader.destroy_surface(data.surface, None);
		#[cfg(not(feature = "goop_imgui"))]
		if let (Some(du), Some(msg)) = (data.debug_utils.as_ref(), data.messenger.as_ref())
		{
			du.destroy_debug_utils_messenger(*msg, None);
		}
		#[cfg(not(feature = "goop_imgui"))]
		instance.destroy_instance(None);
	}
}
