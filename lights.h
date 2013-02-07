#ifndef LIGHTS_H
#define LIGHTS_H

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
        float ambient[4] = {0.3f, 0.3f, 0.3f, 1.0f};
        float diffuse[4] = {0.5f, 0.5f, 0.5f, 1.0f};
        float specular[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float position[4] = {0.0f, 0.0f, 200.0f, 1.0f};
        float viewPos[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float direction[4] = {0.0f, 0.0f, -1.0f, 0.0f};
        float spotCutoff;
        float spotExponent;
        float spotDirection[3] = {0.0f, 0.0f, -1.0f};
        float constAttenuation;
        float linearAttenuation;
        float quadraticAttenuation;
        bool spotLight;
        int location;
    };

    Lights(Scene *parent);
    virtual ~Lights();

    Light lights[8];
    Scene *theScene;
};

#endif // LIGHTS_H
