use anyhow::Result;
use ash::vk;
use log::{trace, info, warn, error};

pub struct AppData
{
	instance: ash::Instance,
	debug_utils: Option<ash::extensions::ext::DebugUtils>,
	messenger: Option<vk::DebugUtilsMessengerEXT>,
}

pub fn create_instance(entry: &ash::Entry, enable_validation: bool) -> Result<(ash::Instance, AppData)>
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

	let data = AppData
				{
					instance: instance.clone(),
					debug_utils,
					messenger,
				};

		Ok( (instance, data) )
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

pub fn destroy_vulkan(data: &mut AppData) -> Result<()>
{
	if let (Some(deb_utils), Some(msgr)) = (data.debug_utils.as_ref(), data.messenger.as_ref())
	{
		unsafe { deb_utils.destroy_debug_utils_messenger(*msgr, None) };
	}

	unsafe { data.instance.destroy_instance(None) };
	Ok(())
}
