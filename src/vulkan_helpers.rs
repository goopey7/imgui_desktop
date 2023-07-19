pub mod vh
{
	use std::ffi::CString;
	use anyhow::{Result, anyhow};
	use ash::vk;
	use log::{trace, info, warn, error};
	use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
	use winit::window::Window;

const MAX_FRAMES_IN_FLIGHT: usize = 3;

	#[derive(Default, Clone)]
	pub struct Data
	{
		frame: usize,
		surface: vk::SurfaceKHR,
		physical_device: vk::PhysicalDevice,
		graphics_queue: vk::Queue,
		transfer_queue: vk::Queue,
		presentation_queue: vk::Queue,
		swapchain_loader: Option<ash::extensions::khr::Swapchain>,
		swapchain: vk::SwapchainKHR,
		swapchain_images: Vec<vk::Image>,
		swapchain_format: vk::Format,
		swapchain_extent: vk::Extent2D,
		swapchain_image_views: Vec<vk::ImageView>,
		render_pass: vk::RenderPass,
		framebuffers: Vec<vk::Framebuffer>,
		pipeline_layout: vk::PipelineLayout,
		pipeline: vk::Pipeline,
		graphics_command_pool: vk::CommandPool,
		transfer_command_pool: vk::CommandPool,
		graphics_command_buffers: Vec<vk::CommandBuffer>,
		in_flight_fences: Vec<vk::Fence>,
		image_available_semaphores: Vec<vk::Semaphore>,
		render_finished_semaphores: Vec<vk::Semaphore>,
		images_in_flight: Vec<vk::Fence>,
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

	pub fn create_instance(entry: &ash::Entry, window: &Window, enable_validation: bool, data: &mut Data) -> Result<ash::Instance>
	{
		let engine_name = std::ffi::CString::new("goopEngine")?;
		let app_name = std::ffi::CString::new("Crab Game")?;

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
		let device_info = vk::DeviceCreateInfo::builder()
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

	pub fn create_render_pass(device: &ash::Device, data: &mut Data) -> Result<()>
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

		let attachments = &[*color_attachment];

		let subpass = vk::SubpassDescription::builder()
			.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
			.color_attachments(color_attachments);

		let dependency = vk::SubpassDependency::builder()
			.src_subpass(vk::SUBPASS_EXTERNAL)
			.dst_subpass(0)
			.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

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
					let attachments = &[*image_view];
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
		let (prefix, code, suffix) = bytecode.align_to::<u32>();
		if !prefix.is_empty() || !suffix.is_empty()
		{
			return Err(anyhow!("Shader bytecode not properly aligned"));
		}

		let info = vk::ShaderModuleCreateInfo::builder()
			.code(code);

		Ok(device.create_shader_module(&info, None)?)
	}


	pub fn create_pipeline(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let vert = include_bytes!("../shaders/vert.spv");
		let frag = include_bytes!("../shaders/frag.spv");

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

		let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder();

		let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
			.topology(vk::PrimitiveTopology::TRIANGLE_LIST)
			.primitive_restart_enable(false);

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
			.rasterization_samples(vk::SampleCountFlags::TYPE_1);

		let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
			.color_write_mask(vk::ColorComponentFlags::R
				| vk::ColorComponentFlags::G
				| vk::ColorComponentFlags::B
				| vk::ColorComponentFlags::A
				)
			.blend_enable(false)
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

		let layout_info = vk::PipelineLayoutCreateInfo::builder();
		data.pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

		let info = vk::GraphicsPipelineCreateInfo::builder()
			.stages(stages)
			.vertex_input_state(&vertex_input_info)
			.input_assembly_state(&input_assembly_info)
			.viewport_state(&viewport_info)
			.rasterization_state(&rasterizer_info)
			.multisample_state(&multisampler_info)
			.color_blend_state(&color_blend_state)
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

	pub fn create_command_pools(instance: &ash::Instance, device: &ash::Device, surface_loader: &ash::extensions::khr::Surface, data: &mut Data) -> Result<()>
	{
		let indices = QueueFamilyIndices::get(instance, data.physical_device, data.surface, surface_loader)?;

		let graphics_pool_info = vk::CommandPoolCreateInfo::builder()
			.queue_family_index(indices.graphics);
		data.graphics_command_pool = unsafe { device.create_command_pool(&graphics_pool_info, None)? };

		let transfer_pool_info = vk::CommandPoolCreateInfo::builder()
			.queue_family_index(indices.transfer);
		data.transfer_command_pool = unsafe { device.create_command_pool(&transfer_pool_info, None)? };

		Ok(())
	}

	pub fn create_command_buffers(device: &ash::Device, data: &mut Data) -> Result<()>
	{
		let g_allocate_info = vk::CommandBufferAllocateInfo::builder()
			.command_pool(data.graphics_command_pool)
			.command_buffer_count(data.framebuffers.len() as u32);
		data.graphics_command_buffers = unsafe { device.allocate_command_buffers(&g_allocate_info)? };

		for (i, &cb) in data.graphics_command_buffers.iter().enumerate()
		{
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
			
			let clear_values = &[color_clear_value];

			let info = vk::RenderPassBeginInfo::builder()
				.render_pass(data.render_pass)
				.framebuffer(data.framebuffers[i])
				.render_area(*render_area)
				.clear_values(clear_values);

			unsafe
			{
				device.cmd_begin_render_pass(cb, &info, vk::SubpassContents::INLINE);
				device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
				device.cmd_draw(cb, 3, 1, 0, 0);
				device.cmd_end_render_pass(cb);
				device.end_command_buffer(cb)?;
			}
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

	pub fn render(instance: &ash::Instance, device: &ash::Device, surface_loader: &ash::extensions::khr::Surface, window: &Window, data: &mut Data) -> Result<()>
	{
		let swapchain_loader = data.swapchain_loader.as_ref().unwrap();
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
			if changed
			{
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
		create_render_pass(&device, data)?;
		create_pipeline(&device, data)?;
		create_framebuffers(&device, data)?;
		create_command_buffers(&device, data)?;
		data.images_in_flight
			.resize(data.swapchain_images.len(), vk::Fence::null());

		Ok(())
	}

	unsafe fn destroy_swapchain(device: &ash::Device, data: &Data)
	{
		data.framebuffers
			.iter()
			.for_each(|fb|
			{
				device.destroy_framebuffer(*fb, None)
			});
		device.free_command_buffers(data.graphics_command_pool, &data.graphics_command_buffers);
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
			device.destroy_device(None);
			surface_loader.destroy_surface(data.surface, None);
			if let (Some(du), Some(msg)) = (data.debug_utils.as_ref(), data.messenger.as_ref())
			{
				du.destroy_debug_utils_messenger(*msg, None);
			}
			instance.destroy_instance(None);
	}
}
