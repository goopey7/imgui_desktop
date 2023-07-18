pub mod vh
{
	use anyhow::{Result, anyhow};
	use ash::vk;
	use log::{trace, info, warn, error};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

	pub struct Data
	{
		instance: ash::Instance,
		device: ash::Device,
		surface_loader: ash::extensions::khr::Surface,
		surface: vk::SurfaceKHR,
		physical_device: vk::PhysicalDevice,
		graphics_queue: vk::Queue,
		transfer_queue: vk::Queue,
		presentation_queue: vk::Queue,
		swapchain_loader: ash::extensions::khr::Swapchain,
		swapchain: vk::SwapchainKHR,
		swapchain_images: Vec<vk::Image>,
		swapchain_format: vk::Format,
		swapchain_extent: vk::Extent2D,
		swapchain_image_views: Vec<vk::ImageView>,
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

	pub struct CreateInstanceData
	{
		instance: ash::Instance,
		debug_utils: Option<ash::extensions::ext::DebugUtils>,
		messenger: Option<vk::DebugUtilsMessengerEXT>,
	}
	pub fn create_instance(entry: &ash::Entry, window: &Window, enable_validation: bool) -> Result<CreateInstanceData>
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

		Ok( CreateInstanceData {
			instance,
			debug_utils,
			messenger,
		})
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

	fn get_physical_device(instance_data: &CreateInstanceData) -> Result<vk::PhysicalDevice>
	{
		let instance = &instance_data.instance;
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

	pub struct DeviceData
	{
		device: ash::Device,
		physical_device: vk::PhysicalDevice,
		graphics_queue: vk::Queue,
		transfer_queue: vk::Queue,
		presentation_queue: vk::Queue,
	}
	pub fn create_logical_device(instance_data: &CreateInstanceData, surface_data: &CreateSurfaceData) -> Result<DeviceData>
	{
		let instance = &instance_data.instance;
		let physical_device = get_physical_device(&instance_data)?;
		let indices = QueueFamilyIndices::get(instance, physical_device, surface_data.surface, &surface_data.surface_loader)?;
		let priorities = [1.0f32];
		let queue_infos = [
			vk::DeviceQueueCreateInfo::builder()
				.queue_family_index(indices.graphics)
				.queue_priorities(&priorities)
				.build(),
			vk::DeviceQueueCreateInfo::builder()
				.queue_family_index(indices.transfer)
				.queue_priorities(&priorities)
				.build(),
		];
		let enabled_extension_name_ptrs =
			vec![ash::extensions::khr::Swapchain::name().as_ptr()];
		let device_info = vk::DeviceCreateInfo::builder()
			.enabled_extension_names(&enabled_extension_name_ptrs)
			.queue_create_infos(&queue_infos);
		let logical_device = unsafe { instance.create_device(physical_device, &device_info, None)? };
		let graphics_queue = unsafe { logical_device.get_device_queue(indices.graphics, 0) };
		let transfer_queue = unsafe { logical_device.get_device_queue(indices.transfer, 0) };
		let presentation_queue = unsafe { logical_device.get_device_queue(indices.presentation, 0) }; 

		Ok(DeviceData
			{
				device: logical_device,
				physical_device,
				graphics_queue,
				transfer_queue,
				presentation_queue,
			})

	}

	pub struct CreateSurfaceData
	{
		surface: vk::SurfaceKHR,
		surface_loader: ash::extensions::khr::Surface,
	}
	pub fn create_surface(entry: &ash::Entry, instance_data: &CreateInstanceData, window: &Window) -> Result<CreateSurfaceData>
	{
		let surface = unsafe {
			ash_window::create_surface(
				&entry,
				&instance_data.instance,
				window.raw_display_handle(),
				window.raw_window_handle(),
				None,
			)?
		};
		let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance_data.instance);

		Ok(CreateSurfaceData { surface, surface_loader, })
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

	pub fn create_swapchain(window: &Window, instance_data: CreateInstanceData, surface_data: CreateSurfaceData, device_data: DeviceData) -> Result<Data>
	{
		let surface_capabilities = unsafe
		{
			surface_data.surface_loader.get_physical_device_surface_capabilities(device_data.physical_device, surface_data.surface)?
		};
		let surface_present_modes = unsafe
		{
			surface_data.surface_loader.get_physical_device_surface_present_modes(device_data.physical_device, surface_data.surface)?
		};
		let surface_formats = unsafe
		{
			surface_data.surface_loader.get_physical_device_surface_formats(device_data.physical_device, surface_data.surface)?
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

		let indices = QueueFamilyIndices::get(&instance_data.instance, device_data.physical_device, surface_data.surface, &surface_data.surface_loader)?;

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
			.surface(surface_data.surface)
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

		let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance_data.instance, &device_data.device);
		let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None)? };

		Ok(Data
			{
				instance: instance_data.instance,
				device: device_data.device,
				surface_loader: surface_data.surface_loader,
				surface: surface_data.surface,
				physical_device: device_data.physical_device,
				graphics_queue: device_data.graphics_queue,
				transfer_queue: device_data.transfer_queue,
				presentation_queue: device_data.presentation_queue,
				swapchain_loader,
				swapchain,
				debug_utils: instance_data.debug_utils,
				messenger: instance_data.messenger,
			}
		)
	}

	pub fn create_swapchain_image_views(data: &mut Data) -> Result<()>
	{
		let swapchain_images = unsafe { data.swapchain_loader.get_swapchain_images(data.swapchain)? };
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
				.format()
				.subresource_range(*subresource);
			let image_view = unsafe { data.device.create_image_view(&imageview_info, None)? };
			swapchain_image_views.push(image_view);
		}

		Ok(())
	}

	pub fn destroy_vulkan(data: &mut Data) -> Result<()>
	{
		unsafe
		{
			data.swapchain_loader.destroy_swapchain(data.swapchain, None);
			data.device.destroy_device(None);
			data.surface_loader.destroy_surface(data.surface, None);
			if let (Some(deb_utils), Some(msgr)) = (data.debug_utils.as_ref(), data.messenger.as_ref())
			{
				deb_utils.destroy_debug_utils_messenger(*msgr, None);
			}

			data.instance.destroy_instance(None)
		};
		Ok(())
	}
}
