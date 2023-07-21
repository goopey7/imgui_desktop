#version 450

layout(binding = 0) uniform UniformBufferObject
{
	mat4 model;
	mat4 view;
	mat4 proj;
} ubo;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inCol;
layout(location = 2) in vec2 inTexCoord;

// output color
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

// gets invoked for each vertex
void main()
{
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos, 1.0);
	fragColor = inCol;
	fragTexCoord = inTexCoord;
}

