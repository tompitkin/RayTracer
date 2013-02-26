#version 330
smooth in vec3 N;
smooth in vec4 vertCam;
smooth in float fog;
smooth in  vec2 texCoord;
smooth in vec3 cubemapTexCoord;
out vec4 fragColor;

struct lightSourceParameters {
//	vec4 ambient; 		//Acli
        vec4 diffuse; 		//Dcli
        vec4 position; 		//Ppli
//	vec4 specular; 		//Scli
        vec3 spotDirection; //Sdli
        float spotExponent; //Srli
        float spotCutoff; 	//Crli (range: [0.0,90.0], 180.0)
        int lightSwitch;
};
uniform lightSourceParameters lightSettings[8];

//uniform vec4 globalAmbient;

struct materialParameters {
        vec4 ambient;   // Acm
        vec4 diffuse;   // Dcm
        vec4 specular;  // Scm
        float shininess; // Srm
        bool ambTex;
        sampler2D ambientTex; //lightmap tex or -1
        bool diffTex;
        sampler2D diffuseTex; //diffuse tex or -1
        bool specTex;
        samplerCube cubeMap; // specular (cubemap) tex or -1
};
uniform materialParameters materialSettings;
uniform mat4 viewMat;

struct fogParameters {
        float start;
        float end;
        float density;
        int eqn;
        vec4 fogColor;
        int fogSwitch;
};
uniform fogParameters fogSettings;

void main() {
        vec4 ambientTexColor = vec4(0.0);
        vec4 diffuseTexColor = vec4(0.0);
        vec4 specTexColor = vec4(0.0);

        vec4 ambientShade = vec4(0.0);
        vec4 specularShade = vec4(0.0);
        vec4 diffuseShade = vec4(0.0);

        bool lightmap;
        if(materialSettings.ambTex)
                ambientTexColor = texture(materialSettings.ambientTex, texCoord.st);
        if(materialSettings.diffTex)
                diffuseTexColor = texture(materialSettings.diffuseTex, texCoord.st);
        if(materialSettings.specTex)
                specTexColor = texture(materialSettings.cubeMap, cubemapTexCoord.stp);

        float spotEffect,spotEffectPow;

        if(materialSettings.ambTex && materialSettings.diffTex){
                lightmap = true;
                //diffuseShade += ambientTexColor * diffuseTexColor;// lightmapping - no light calculations
        }
        else lightmap=false;
//if(!gl_FrontFacing){
//	fragColor = vec4(1.0, 1.0, 0.0, 1.0);
//	return;
//}
        vec4 ambShd; // variables for storing shade from this light
        vec4 diffShd;
        vec4 specShd;
        for(int i = 0; i < 8; i++){
                ambShd = vec4(0.0); // variables for storing shade from this light
                diffShd = vec4(0.0);
                specShd = vec4(0.0);

                if(lightSettings[i].lightSwitch == 1){
                        vec4 Ldir = (viewMat * lightSettings[i].position) - vertCam;
                        vec3 L = normalize(vec3(Ldir));
                        float NdotL;
                        if(!gl_FrontFacing){
                                vec3 flip = -1.0 * N;
                                NdotL = max(0.0, dot(flip, L));
                        }
                        else
                                NdotL = max(0.0, dot(N, L));

                        if(lightmap){
                                diffShd += lightSettings[i].diffuse * ambientTexColor * (diffuseTexColor * vec4(NdotL));
                        }
                        else{
                                ambShd += lightSettings[i].diffuse * ((materialSettings.ambTex) ? ambientTexColor : materialSettings.ambient);
                                diffShd += lightSettings[i].diffuse * vec4(NdotL) *((materialSettings.diffTex) ? diffuseTexColor : materialSettings.diffuse);
                        }
                        vec3 I = L * vec3(-1.0);
                        vec3 R = normalize(reflect(I,N));
                        vec3 V = normalize(vec3(0.0) - vertCam.xyz);
                        //calculate specular lighting
                        float RdotV = max(0.0, dot(R, V));

                        if(!materialSettings.specTex)
                                specShd += vec4(pow(RdotV, materialSettings.shininess)* materialSettings.specular * lightSettings[i].diffuse);

                        // Spotlight section- scales existing Shade values
                        if(lightSettings[i].spotCutoff <= 90.0){ // spotlight if 0-90 not if greater
                                // calculate cos of angle between spot direction and light to vertex vector (-L)
                                spotEffect = dot(normalize(lightSettings[i].spotDirection), normalize(I));
                                if(spotEffect > cos(radians(lightSettings[i].spotCutoff))){
                                        // now apply the cosine power
                                        spotEffectPow = pow(spotEffect, lightSettings[i].spotExponent);
                                        // now use spotEffect to attenuate at least diffuse and specular lighting
                                        ambShd = ambShd *spotEffectPow;
                                        diffShd = diffShd * spotEffectPow;
                                        specShd = specShd * spotEffectPow;
                                }
                                else { // is a spotlight but outside cutoff radius
                                        diffShd = ambShd = specShd = vec4(0.0);
                                }

                        }

                        ambientShade += ambShd;
                        diffuseShade+= diffShd;
                        specularShade += specShd;
                }
        }
        //}

        // Assign fragColor
        // sum up component Shade values with texture values and assign fragColor
        if (materialSettings.specTex){
                //fragColor = texture(materialSettings.cubeMap, cubemapTexCoord.stp);
                fragColor = (0.6* specTexColor)+ (0.2*diffuseShade)+ (0.2 * ambientShade);
                //fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        else
                        fragColor = ambientShade + diffuseShade + specularShade;

        if(fogSettings.fogSwitch ==1) // Mix in fog with fragColor
                //fragColor = mix(vec4(0.1, 0.1, 0.1, 1.0), fragColor, fog);
                fragColor = mix(fogSettings.fogColor, fragColor, fog);

}
