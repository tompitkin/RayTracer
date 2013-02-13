#include "lights.h"

Lights::Lights(Scene *parent)
{
    theScene = parent;
    lights[0].lightSwitch = Light::ON;
    lights[0].location = Light::LOCAL;
    lights[0].position = {0.0f, 0.0f, 10.0f, 1.0f};
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
    specular = {1.0f, 1.0f, 1.0f, 1.0f};
    position = {0.0f, 0.0f, 200.0f, 1.0f};
    viewPos = {0.0f, 0.0f, 0.0f, 0.0f};
    direction = {0.0f, 0.0f, -1.0f, 0.0f};
    spotCutoff = 180.0f;
    spotExponent = 0.0f;
    spotDirection = {0.0f, 0.0f, -1.0f};
    constAttenuation = 1.0f;
    linearAttenuation = 0.0f;
    quadraticAttenuation = 0.0f;
    location = LOCAL;
}
