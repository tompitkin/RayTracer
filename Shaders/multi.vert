#version 330

// Incoming per vertex... position, normal, texture coords
layout(location=0) in vec4 vertex;
layout(location=1) in vec3 normal;
layout(location=2) in vec2 inTexCoord;

uniform mat4 viewMat;
uniform mat4 projMat;
uniform mat4 modelMat;
uniform mat3 normalMat;
uniform mat4 mInverseCamera;
uniform bool cubeMapping;
struct fogParameters {
       float start;
       float end;
       float density;
       int eqn;
       vec4 fogColor;
       int fogSwitch;
};
uniform fogParameters fogSettings;

// Outgoing
smooth out vec3 N;
smooth out vec4 vertCam;
smooth out float fog;
smooth out  vec2 texCoord;
smooth out vec3 cubemapTexCoord;

void main() {

       // Normal in Eye Space
   vec3 eyeNormal = normalMat * normal;
   mat4 mvpMat = projMat * viewMat * modelMat;
       N = normalize(normalMat * normal);


       mat4 mvMat = viewMat * modelMat;
       vertCam = mvMat * vertex;
       if(cubeMapping){
               // Stuff for cubemapping- Vertex position in Eye Space
               vec4 vVert4 = mvMat * vertex;
               vec3 eyeVertexVect = normalize(vVert4.xyz / vVert4.w);
       //vertCam = mvMat * vertex;

               // cubemapping - Get reflected vector
               vec4 reflVect = vec4(reflect(eyeVertexVect, eyeNormal), 1.0);

               // cubemapping - Rotate by flipped camera
               reflVect  = mInverseCamera * reflVect;
               cubemapTexCoord.xyz = normalize(reflVect.xyz);
       }

       // Fog effects - needs to be tied into interface
       //float start = 30.0;
       //float end = 50.0;
       //float density = 0.1;
       float vertDist = length(vertCam);

       if(fogSettings.fogSwitch == 1){
               // linear computation
               if(fogSettings.eqn == 1)
                       fog = (fogSettings.end - vertDist)/(fogSettings.end - fogSettings.start);
               else if(fogSettings.eqn == 2) //exp computation
                       fog = exp(-fogSettings.density * vertDist);
               else //exp2 computation
                       fog = exp(-fogSettings.density * fogSettings.density * vertDist*vertDist);
               fog = clamp(fog, 0.0, 1.0);
       }

   // Set the position of the current vertex
       gl_Position = mvpMat * vertex;
       if(!cubeMapping)
               texCoord  = inTexCoord.st;

}
