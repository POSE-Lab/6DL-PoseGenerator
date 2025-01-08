#version 330
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos; 
in vec2 uv_coords;

uniform vec3 lightPos; 
uniform vec3 viewPos; 
uniform vec3 lightColor;

uniform float ambientStrength;
uniform float specularStrength;

uniform sampler2D f_texture;

void main(){
    
    //uncomments for constant color hundles by the color-picker
    //outColor = vec4( 1.0f, 1.0f, 1.0f, 0.1f );

    vec4 outColor = texture(f_texture,uv_coords);

     // ambient
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  
        
    vec3 result = (ambient + diffuse + specular) * (outColor.xyz * 0.6);
    if(uv_coords == vec2(0.0,0.0)) {
        FragColor = vec4(0.0,0.0,0.0,1.0);
    }else {FragColor = vec4(result, 1.0);}
    
}