pub mod vh
{
	use anyhow::{Result, anyhow};
	use ash::vk;
	use log::{trace, info, warn, error};

	pub struct Data
	{
		instance: ash::Instance,
		device: ash::Device,
		physical_device: vk::PhysicalDevice,
		graphics_queue: vk::Queue,
		transfer_queue: vk::Queue,
		debug_utils: Option<ash::extensions::ext::DebugUtils>,
		messenger: Option<vk::DebugUtilsMessengerEXT>,
	}

	#[derive(Copy, Clone, Debug)]
	struct QueueFamilyIndices
	{
		graphics: u32,
		transfer: u32,
	}

	impl QueueFamilyIndices
	{
		fn get(
			instance: &ash::Instance,
			physical_device: vk::PhysicalDevice,
			) -> Result<Self>
		{
			let properties = unsafe {instance.get_physical_device_queue_family_properties(physical_device)};

			let graphics = properties
				.iter()
				.position(|properties| properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
				.map(|index| index as u32);

			let transfer = properties
				.iter()
				.position(|properties|
					properties.queue_flags.contains(vk::QueueFlags::TRANSFER)
					&& !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS))
				.map(|index| index as u32);

			if let (Some(graphics), Some(transfer)) = (graphics, transfer)
			{
				Ok(Self {graphics, transfer})
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
	pub fn create_instance(entry: &ash::Entry, enable_validation: bool) -> Result<CreateInstanceData>
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
			vec![];

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

	pub fn create_logical_device(instance_data: CreateInstanceData) -> Result<Data>
	{
		let instance = &instance_data.instance;
		let physical_device = get_physical_device(&instance_data)?;
		let indices = QueueFamilyIndices::get(instance, physical_device)?;
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
		let device_info = vk::DeviceCreateInfo::builder()
			.queue_create_infos(&queue_infos);
		let logical_device = unsafe { instance.create_device(physical_device, &device_info, None)? };
		let graphics_queue = unsafe { logical_device.get_device_queue(indices.graphics, 0) };
		let transfer_queue = unsafe { logical_device.get_device_queue(indices.transfer, 0) };

		Ok(Data
			{
				instance: instance_data.instance,
				device: logical_device,
				physical_device,
				graphics_queue,
				transfer_queue,
				debug_utils: instance_data.debug_utils,
				messenger: instance_data.messenger,
			}
		)
	}

	pub fn destroy_vulkan(data: &mut Data) -> Result<()>
	{
		unsafe
		{
			data.device.destroy_device(None);
			if let (Some(deb_utils), Some(msgr)) = (data.debug_utils.as_ref(), data.messenger.as_ref())
			{
				deb_utils.destroy_debug_utils_messenger(*msgr, None);
			}

			data.instance.destroy_instance(None)
		};
		Ok(())
	}
}
