#version 450

// input color from vertex shader
layout(location=0) in vec3 fragColor;
layout(location=1) in vec2 fragTexCoord;
layout(location=2) in flat uint fragTexId;

layout(push_constant) uniform PushConstants
{
	layout(offset=64) float opacity;
} pcs;

// create variable for framebuffer (we have one so index 0)
layout(location=0) out vec4 outColor;

// called for every fragment (which was output from the vertex shader)
void main()
{
	//outColor = vec4(texture(texSamplers[fragTexId], fragTexCoord).rgb, pcs.opacity);
	//outColor = vec4(texture(texSamplers[fragTexId], fragTexCoord).rgb, 1.0f);

	outColor = vec4(1.f, 0.f, 0.f, pcs.opacity);
}

