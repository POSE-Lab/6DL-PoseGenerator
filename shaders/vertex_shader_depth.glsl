#version 330

layout(location = 0) in vec3 pos;
        
        
uniform mat4 model; 
uniform mat4 view;
uniform mat4 projection;
uniform float depth_scale;

out float depth;
void main() {
 
        gl_Position = projection * view * model * vec4(pos, 1.0f);
        vec3 v_eye_pos = (view * vec4(pos, 1.0f)).xyz;
        depth = - v_eye_pos.z * depth_scale;
 
}
