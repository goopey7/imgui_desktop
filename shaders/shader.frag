#version 450

// create variable for framebuffer (we have one so index 0)
layout(location=0) out vec4 outColor;

// input color from vertex shader
layout(location=0) in vec3 fragColor;

// called for every fragment (which was output from the vertex shader)
void main()
{
	outColor = vec4(fragColor, 1.0);
}

