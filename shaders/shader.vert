#version 450

layout(binding = 0) uniform UniformBufferObject
{
	mat4 view;
	mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants
{
	mat4 model;
} pcs;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inCol;
layout(location = 2) in vec2 inTexCoord;

// offset transform
layout(location = 3) in vec4 transform0;
layout(location = 4) in vec4 transform1;
layout(location = 5) in vec4 transform2;
layout(location = 6) in vec4 transform3;

// output color
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

// gets invoked for each vertex
void main()
{
	mat4 transform = mat4(transform0, transform1, transform2, transform3);
	gl_Position = ubo.proj * ubo.view * pcs.model * transform * vec4(inPos, 1.0);
	fragColor = inCol;
	fragTexCoord = inTexCoord;
}

