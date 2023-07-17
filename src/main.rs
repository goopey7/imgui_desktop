use anyhow::Result;
use ash::vk;
use log::{error, warn, info, trace};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const MAX_FRAMES_IN_FLIGHT: usize = 3;

#[derive(Default)]
struct AppData
{
	debug_utils: Option<ash::extensions::ext::DebugUtils>,
	messenger: Option<vk::DebugUtilsMessengerEXT>,
}

fn main() -> Result<()>
{
	pretty_env_logger::init();

	let mut data = AppData::default();

	let entry = unsafe { ash::Entry::load()? };

	let engine_name = std::ffi::CString::new("goopEngine")?;
	let app_name = std::ffi::CString::new("Crab Game")?;

	let app_info = vk::ApplicationInfo::builder()
		.application_name(&app_name)
		.engine_name(&engine_name)
		.engine_version(vk::make_api_version(0, 0, 0, 0))
		.application_version(vk::make_api_version(0, 0, 0, 0))
		.api_version(vk::make_api_version(0, 1, 0, 106));

	let layer_names: Vec<std::ffi::CString> = 
		if VALIDATION_ENABLED
		{
			vec![std::ffi::CString::new("VK_LAYER_KHRONOS_validation")?]
		}
		else
		{
			vec![]
		};

	let layer_name_ptrs: Vec<*const i8> = 
		if VALIDATION_ENABLED
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

	if VALIDATION_ENABLED
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

	if VALIDATION_ENABLED
	{
		instance_info = instance_info.push_next(&mut debug_info);
	}

	let instance = unsafe { entry.create_instance(&instance_info, None)? };

	if VALIDATION_ENABLED
	{
		data.debug_utils = Some(ash::extensions::ext::DebugUtils::new(&entry, &instance));
		data.messenger = unsafe { Some(data.debug_utils.as_ref().unwrap().create_debug_utils_messenger(&debug_info, None)?) };
	}

	if let (Some(deb_utils), Some(msgr)) = (data.debug_utils, data.messenger)
	{
		unsafe { deb_utils.destroy_debug_utils_messenger(msgr, None) };
	}

	unsafe { instance.destroy_instance(None) };
	Ok(())
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
