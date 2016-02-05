#version 330
in vec4 model_vertex; // The location of the vertex in model space
in vec3 in_dx;
in vec3 in_left_edge;
in vec3 in_right_edge;
out vec4 v_model;
out vec3 v_camera_pos;
flat out mat4 inverse_proj;
flat out mat4 inverse_view;
out vec3 dx;
out vec3 left_edge;
out vec3 right_edge;

//// Uniforms
uniform vec3 camera_pos;
uniform mat4 lookat;
uniform mat4 projection;

void main()
{
    v_model = model_vertex;
    v_camera_pos = camera_pos;
    inverse_proj = inverse(projection);
    inverse_view = inverse(lookat);
    gl_Position = projection * lookat * model_vertex;
    dx = in_dx;
    left_edge = in_left_edge;
    right_edge = in_right_edge;

}
