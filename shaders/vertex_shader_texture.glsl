#version 330

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 uv;

out vec2 uv_coords;
out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){

    FragPos = vec3(model * vec4(pos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal; 

    gl_Position = projection * view * vec4(FragPos, 1.0);
    uv_coords = uv;

    //transpose texture image
    uv_coords = vec2(uv.s, 1 - uv.t);
}