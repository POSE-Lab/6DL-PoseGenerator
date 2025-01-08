#version 330
 
in float depth;     
out vec4 outColor;

void main() {
 
  outColor = vec4(depth,depth,depth,1.0f);
 
}
