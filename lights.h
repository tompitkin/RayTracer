#ifndef LIGHTS_H
#define LIGHTS_H

#include <vector>
#include <string>
#include "Uniforms/vec4uniform.h"
#include "Uniforms/intuniform.h"

using namespace std;

class Scene;

class Lights
{
public:
    class Light
    {
    public:
        Light();

        const static int OFF = 0;
        const static int ON = 1;
        const static int DIRECTIONAL = 0;
        const static int LOCAL = 1;
        int lightSwitch;
        vector<float> ambient;
        vector<float> diffuse;
        vector<float> specular;
        vector<float> position;
        vector<float> viewPos;
        vector<float> direction;
        float spotCutoff;
        float spotExponent;
        vector<float> spotDirection;
        float constAttenuation;
        float linearAttenuation;
        float quadraticAttenuation;
        bool spotLight;
        int location;
    };

    Lights(Scene *parent);
    virtual ~Lights();

    void updateLight(int lightIndex);

    Light lights[8];
    Scene *theScene;

    vector<Vec4Uniform*> lightAmb = vector<Vec4Uniform*>(8, nullptr);
};

#endif // LIGHTS_H
