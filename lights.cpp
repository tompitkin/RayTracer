#include "lights.h"
#include "scene.h"

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
    location = Light::LOCAL;
}

void Lights::updateLight(int lightIndex)
{
    string varNum = "lightSettings["+to_string(lightIndex)+"].";
    if (lightSwitch[lightIndex] == nullptr)
        lightSwitch[lightIndex] = shared_ptr<IntUniform>(new IntUniform(lights[lightIndex].lightSwitch, varNum+"lightSwitch"));
    else
        lightSwitch[lightIndex]->theInt = lights[lightIndex].lightSwitch;
    lightSwitch[lightIndex]->needsUpdate = true;

    if (lightAmb[lightIndex] == nullptr)
    {
        lightAmb[lightIndex] = shared_ptr<Vec4Uniform>(new Vec4Uniform(nullptr, varNum+"ambient"));
        lightAmb[lightIndex]->theBuffer = new GLfloat[lights[lightIndex].ambient.size()];
        copy(lights[lightIndex].ambient.begin(), lights[lightIndex].ambient.end(), lightAmb[lightIndex]->theBuffer);
    }
    else
    {
        if (!lightAmb[lightIndex]->off)
        {
            copy(lights[lightIndex].ambient.begin(), lights[lightIndex].ambient.end(), lightAmb[lightIndex]->theBuffer);
            lightAmb[lightIndex]->needsUpdate = true;
        }
        else
            lightAmb[lightIndex]->needsUpdate = false;
    }

    if (lightDiff[lightIndex] == nullptr)
    {
        lightDiff[lightIndex] = shared_ptr<Vec4Uniform>(new Vec4Uniform(nullptr, varNum+"diffuse"));
        lightDiff[lightIndex]->theBuffer = new GLfloat[lights[lightIndex].diffuse.size()];
        copy(lights[lightIndex].diffuse.begin(), lights[lightIndex].diffuse.end(), lightDiff[lightIndex]->theBuffer);
    }
    else
    {
        if (!lightDiff[lightIndex]->off)
        {
            copy(lights[lightIndex].diffuse.begin(), lights[lightIndex].diffuse.end(), lightDiff[lightIndex]->theBuffer);
            lightDiff[lightIndex]->needsUpdate = true;
        }
        else
            lightDiff[lightIndex]->needsUpdate = false;
    }

    if (lightPosition[lightIndex] == nullptr)
    {
        lightPosition[lightIndex] = shared_ptr<Vec4Uniform>(new Vec4Uniform(nullptr, varNum+"position"));
        lightPosition[lightIndex]->theBuffer = new GLfloat[lights[lightIndex].position.size()];
        copy(lights[lightIndex].position.begin(), lights[lightIndex].position.end(), lightPosition[lightIndex]->theBuffer);
    }
    else
    {
        if (!lightPosition[lightIndex]->off)
        {
            copy(lights[lightIndex].position.begin(), lights[lightIndex].position.end(), lightPosition[lightIndex]->theBuffer);
           lightPosition[lightIndex]->needsUpdate = true;
        }
        else
            lightPosition[lightIndex]->needsUpdate = false;
    }

    if (lightSpec[lightIndex] == nullptr)
    {
        lightSpec[lightIndex] = shared_ptr<Vec4Uniform>(new Vec4Uniform(nullptr, varNum+"specular"));
        lightSpec[lightIndex]->theBuffer = new GLfloat[lights[lightIndex].specular.size()];
        copy(lights[lightIndex].specular.begin(), lights[lightIndex].specular.end(), lightSpec[lightIndex]->theBuffer);
    }
    else
    {
        if (!lightSpec[lightIndex]->off)
        {
            copy(lights[lightIndex].specular.begin(), lights[lightIndex].specular.end(), lightSpec[lightIndex]->theBuffer);
            lightSpec[lightIndex]->needsUpdate = true;
        }
        else
            lightSpec[lightIndex]->needsUpdate = false;
    }

    if (spotDirection[lightIndex] == nullptr)
    {
        spotDirection[lightIndex] = shared_ptr<Vec3Uniform>(new Vec3Uniform(nullptr, varNum+"spotDirection"));
        spotDirection[lightIndex]->theBuffer = new GLfloat[lights[lightIndex].direction.size()];
        copy(lights[lightIndex].direction.begin(), lights[lightIndex].direction.end(), spotDirection[lightIndex]->theBuffer);
    }
    else
    {
        if (!spotDirection[lightIndex]->off)
        {
            copy(lights[lightIndex].direction.begin(), lights[lightIndex].direction.end(), spotDirection[lightIndex]->theBuffer);
            spotDirection[lightIndex]->needsUpdate = true;
        }
        else
            spotDirection[lightIndex]->needsUpdate = false;
    }

    if (spotExponent[lightIndex] == nullptr)
        spotExponent[lightIndex] = shared_ptr<FloatUniform>(new FloatUniform(lights[lightIndex].spotExponent, varNum+"spotExponent"));
    else
    {
        if (!spotExponent[lightIndex]->off)
        {
            spotExponent[lightIndex]->theFloat = lights[lightIndex].spotExponent;
            spotExponent[lightIndex]->needsUpdate = true;
        }
        else
            spotExponent[lightIndex]->needsUpdate = false;
    }

    if (spotCutoff[lightIndex] == nullptr)
        spotCutoff[lightIndex] = shared_ptr<FloatUniform>(new FloatUniform(lights[lightIndex].spotCutoff, varNum+"spotCutoff"));
    else
    {
        if (!spotCutoff[lightIndex]->off)
        {
            spotCutoff[lightIndex]->theFloat = lights[lightIndex].spotCutoff;
            spotCutoff[lightIndex]->needsUpdate = true;
        }
        else
            spotCutoff[lightIndex]->needsUpdate = false;
    }

    theScene->updateLights[lightIndex] = false;
}
