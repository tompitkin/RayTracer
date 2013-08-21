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
    if (bitmap != nullptr)
        delete bitmap;
}

void RayTracerCuda::start()
{
    if (bitmap != nullptr)
        delete bitmap;
    bitmap = new Bitmap(theScene->camera->getViewportWidth(), theScene->camera->getViewportHeight(), theScene->camera->getWindowWidth(), theScene->camera->getWindowHeight(), theScene->camera->windowLeft, theScene->camera->windowBottom, theScene->camera->near);
    rayTracer->data = (unsigned char*)malloc(sizeof(unsigned char)*(bitmap->width * bitmap->height * 3));
    cudaStart(bitmap);
    memcpy(rayTracer->data, bitmap->data, (bitmap->width * bitmap->height * 3));
}
