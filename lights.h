#ifndef LIGHTS_H
#define LIGHTS_H

#include <vector>
#include <string>
#include <memory>
#include "Uniforms/vec4uniform.h"
#include "Uniforms/vec3uniform.h"
#include "Uniforms/intuniform.h"
#include "Uniforms/floatuniform.h"

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

    vector<shared_ptr<Vec4Uniform>> lightAmb = vector<shared_ptr<Vec4Uniform>>(8, shared_ptr<Vec4Uniform>());
    vector<shared_ptr<Vec4Uniform>> lightDiff = vector<shared_ptr<Vec4Uniform>>(8, shared_ptr<Vec4Uniform>());
    vector<shared_ptr<Vec4Uniform>> lightSpec = vector<shared_ptr<Vec4Uniform>>(8, shared_ptr<Vec4Uniform>());
    vector<shared_ptr<Vec4Uniform>> lightPosition = vector<shared_ptr<Vec4Uniform>>(8, shared_ptr<Vec4Uniform>());
    vector<shared_ptr<IntUniform>> lightSwitch = vector<shared_ptr<IntUniform>>(8, shared_ptr<IntUniform>());
    vector<shared_ptr<FloatUniform>> spotExponent = vector<shared_ptr<FloatUniform>>(8, shared_ptr<FloatUniform>());
    vector<shared_ptr<FloatUniform>> spotCutoff = vector<shared_ptr<FloatUniform>>(8, shared_ptr<FloatUniform>());
    vector<shared_ptr<Vec3Uniform>> spotDirection = vector<shared_ptr<Vec3Uniform>>(8, shared_ptr<Vec3Uniform>());
};

#endif // LIGHTS_H
