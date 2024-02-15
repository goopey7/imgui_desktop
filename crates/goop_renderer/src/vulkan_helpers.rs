pub mod vh
{
	use anyhow::{Result, anyhow};
	use ash::vk;
	use log::{trace, info, warn, error};
	use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
	use winit::window::Window;

	const MAX_FRAMES_IN_FLIGHT: usize = 3;

	#[derive(Default, Clone)]
	pub struct Data
	{
		pub wireframe: bool,
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
		index_buffer: vk::Buffer,
		index_buffer_memory: vk::DeviceMemory,
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
					f.format == vk::Format::R8G8B8A8_UNORM
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

	pub fn set_msaa_samples(data: &mut Data) -> Result<()>
	{
		data.msaa_samples = vk::SampleCountFlags::TYPE_1;
		Ok(())
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
		renderer: &mut imgui_rs_vulkan_renderer::Renderer,
		draw_data: &imgui::DrawData,
		) -> Result<()>
	{
		let cp = data.graphics_command_pools[image_index];
		unsafe { device.reset_command_pool(cp, vk::CommandPoolResetFlags::empty())? };
		let cb = data.graphics_command_buffers[image_index];

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
			renderer.cmd_draw(cb, draw_data)?;
			device.cmd_end_render_pass(cb);
			device.end_command_buffer(cb)?;
		}

		Ok(())
	}

	pub fn toggle_wireframe(instance: &ash::Instance, device: &ash::Device, surface_loader: &ash::extensions::khr::Surface, window: &Window, data: &mut Data) -> Result<()>
	{
		data.wireframe = !data.wireframe;

		recreate_swapchain(instance, device, surface_loader, window, data)?;

		Ok(())
	}

	pub fn render(
		instance: &ash::Instance,
		device: &ash::Device,
		surface_loader: &ash::extensions::khr::Surface,
		window: &Window,
		data: &mut Data,
		renderer: &mut imgui_rs_vulkan_renderer::Renderer,
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
			renderer,
			&draw_data,
		)?;

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
		create_color_objects(instance, device, data)?;
		create_depth_objects(instance, device, data)?;
		create_framebuffers(device, data)?;
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

	pub unsafe fn destroy(device: &ash::Device, data: &Data)
	{
		device.device_wait_idle().unwrap();
		destroy_swapchain(device, data);
		data.graphics_command_pools
			.iter()
			.for_each(|cp| device.destroy_command_pool(*cp, None));
		device.destroy_buffer(data.index_buffer, None);
		device.free_memory(data.index_buffer_memory, None);
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
	}
}
