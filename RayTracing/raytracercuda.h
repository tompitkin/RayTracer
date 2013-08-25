#ifndef RAYTRACERCUDA_H
#define RAYTRACERCUDA_H

#include "RayTracing/cudaKernel.h"

class Scene;
class RayTracer;

class RayTracerCuda
{
public:
    RayTracerCuda(Scene *theScene, RayTracer *rayTracer);
    virtual ~RayTracerCuda();

    void start();
    void buildBitmap();

    Scene *theScene;
    RayTracer *rayTracer;
};

#endif // RAYTRACERCUDA_H
