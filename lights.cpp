#include "lights.h"

Lights::Lights(Scene *parent)
{
    theScene = parent;
    lights[0].lightSwitch = Light::ON;
    lights[0].location = Light::LOCAL;
}

Lights::~Lights()
{
    theScene = nullptr;
}

Lights::Light::Light()
{
    lightSwitch = OFF;
    spotLight = false;
    ambient = {0.3f, 0.3f, 0.3f, 1.0f};
    diffuse = {0.5f, 0.5f, 0.5f, 1.0f};
    spotCutoff = 180.0f;
    spotExponent = 0.0f;
    constAttenuation = 1.0f;
    linearAttenuation = 0.0f;
    quadraticAttenuation = 0.0f;
    location = LOCAL;
}
