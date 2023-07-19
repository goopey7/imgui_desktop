#version 450

// input color from vertex shader
layout(location=0) in vec3 fragColor;
layout(location=1) in vec2 fragTexCoord;

layout(binding=1) uniform sampler2D texSampler;

// create variable for framebuffer (we have one so index 0)
layout(location=0) out vec4 outColor;

// called for every fragment (which was output from the vertex shader)
void main()
{
	outColor = texture(texSampler, fragTexCoord);
}

