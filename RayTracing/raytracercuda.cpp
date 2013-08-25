#include "raytracercuda.h"
#include "scene.h"
#include "RayTracing/raytracer.h"

RayTracerCuda::RayTracerCuda(Scene *theScene, RayTracer *rayTracer)
{
    this->theScene = theScene;
    this->rayTracer = rayTracer;
}

RayTracerCuda::~RayTracerCuda()
{
    theScene = nullptr;
    rayTracer = nullptr;
}

void RayTracerCuda::start()
{
    Bitmap bitmap;
    bitmap.width = theScene->camera->getViewportWidth();
    bitmap.height = theScene->camera->getViewportHeight();
    bitmap.pixelWidth = theScene->camera->getWindowWidth() / bitmap.width;
    bitmap.pixelHeight = theScene->camera->getWindowHeight() / bitmap.height;
    bitmap.firstPixel = Double3D(theScene->camera->windowLeft + bitmap.pixelWidth / 2, theScene->camera->windowBottom + bitmap.pixelHeight / 2, -theScene->camera->near);

    cudaStart(&bitmap);

    if (rayTracer->data != nullptr)
    {
        free(rayTracer->data);
        rayTracer->data = nullptr;
    }
    rayTracer->data = bitmap.data;
}
