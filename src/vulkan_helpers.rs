pub mod vh
{
	use std::ffi::CString;

use anyhow::{Result, anyhow};
	use ash::vk;
	use log::{trace, info, warn, error};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

	#[derive(Default)]
	pub struct Data
	{
		surface_loader: Option<ash::extensions::khr::Surface>,
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

			let transfer = properties
				.iter()
				.position(|properties| properties.queue_flags.contains(vk::QueueFlags::TRANSFER)
					&& !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
				.map(|index| index as u32);

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

	fn get_physical_device(instance: &ash::Instance, data: &Data) -> Result<vk::PhysicalDevice>
	{
		let phys_devices = unsafe { instance.enumerate_physical_devices()? };
		let physical_device =
		{
			let mut chosen = Err(anyhow!("no appropriate physical device available"));
			for pd in phys_devices
			{
				let props = unsafe { instance.get_physical_device_properties(pd) };
				if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
				{
					chosen = Ok(pd);
				}
			}
			chosen?
		};

		Ok(physical_device)
	}

	pub fn create_logical_device(instance: &ash::Instance, data: &mut Data) -> Result<ash::Device>
	{
		let physical_device = get_physical_device(instance, &data)?;
		let indices = QueueFamilyIndices::get(instance, physical_device, data.surface, &data.surface_loader.as_ref().unwrap())?;
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

		let queue_infos = if indices.graphics == indices.presentation
		{
			vec![g_info.build(), t_info.build()]
		}
		else
		{
			vec![g_info.build(), t_info.build(), p_info.build()]
		};

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

	pub fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &Window, data: &mut Data) -> Result<()>
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
		let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

		data.surface_loader = Some(surface_loader);
		data.surface = surface;

		Ok(())
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

	pub fn create_swapchain(instance: &ash::Instance, device: &ash::Device, window: &Window, data: &mut Data) -> Result<()>
	{
		let surface_capabilities = unsafe
		{
			data.surface_loader.as_ref().unwrap().get_physical_device_surface_capabilities(data.physical_device, data.surface)?
		};
		let surface_present_modes = unsafe
		{
			data.surface_loader.as_ref().unwrap().get_physical_device_surface_present_modes(data.physical_device, data.surface)?
		};
		let surface_formats = unsafe
		{
			data.surface_loader.as_ref().unwrap().get_physical_device_surface_formats(data.physical_device, data.surface)?
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

		let indices = QueueFamilyIndices::get(instance, data.physical_device, data.surface, &data.surface_loader.as_ref().unwrap())?;

		let mut queue_family_indices = vec![];
		let image_sharing_mode = if indices.graphics != indices.presentation
			{
				queue_family_indices.push(indices.graphics);
				queue_family_indices.push(indices.presentation);
				queue_family_indices.push(indices.transfer);
				vk::SharingMode::EXCLUSIVE
			}
			else
			{
				queue_family_indices.push(indices.graphics);
				queue_family_indices.push(indices.transfer);
				vk::SharingMode::CONCURRENT
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

		let color_attachments = &[color_attachment_ref.build()];

		let attachments = &[color_attachment.build()];

		let subpass = vk::SubpassDescription::builder()
			.pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
			.color_attachments(color_attachments);

		let dependency = vk::SubpassDependency::builder()
			.src_subpass(vk::SUBPASS_EXTERNAL)
			.dst_subpass(0)
			.src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
			.dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

		let subpasses = &[subpass.build()];
		let dependencies = &[dependency.build()];

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

		let stages = &[vert_stage.build(), frag_stage.build()];

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
		let viewports = &[viewport.build()];

		let scissor = vk::Rect2D::builder()
			.offset(vk::Offset2D {x: 0, y:0 })
			.extent(data.swapchain_extent);
		let scissors = &[scissor.build()];

		let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
			.viewports(viewports)
			.scissors(scissors);

		let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
			.line_width(1.0)
			.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
		let blend_attachments = &[color_blend_attachment.build()];

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
				&[info.build()],
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

	pub fn destroy_vulkan(instance: &ash::Instance, device: &ash::Device, data: &mut Data)
	{
		let swap_loader = data.swapchain_loader.as_ref().unwrap();
		let surf_loader = data.surface_loader.as_ref().unwrap();
		unsafe
		{
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
			swap_loader.destroy_swapchain(data.swapchain, None);
			device.destroy_device(None);
			surf_loader.destroy_surface(data.surface, None);
			if let (Some(deb_utils), Some(msgr)) = (data.debug_utils.as_ref(), data.messenger.as_ref())
			{
				deb_utils.destroy_debug_utils_messenger(*msgr, None);
			}

			instance.destroy_instance(None)
		};
	}
}
