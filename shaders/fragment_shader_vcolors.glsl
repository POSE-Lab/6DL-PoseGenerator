#version 330 core
in vec3 FragPos;
in vec3 Normal;
in vec3 vColor;

uniform vec3 lightPos;    // Light position
uniform vec3 viewPos;     // Camera position
uniform vec3 lightColor;  // Light color

uniform float ambientStrength;
uniform float specularStrength;

out vec4 FragColor;

void main() {
    // Ambient
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

    // Combine results
    vec3 result = (ambient + diffuse + specular) * vColor;
    FragColor = vec4(result, 1.0);
}