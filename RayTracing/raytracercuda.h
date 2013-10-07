#ifndef RAYTRACERCUDA_H
#define RAYTRACERCUDA_H

#include "RayTracing/cudaKernel.h"
#include "pmesh.h"

class Scene;
class RayTracer;

class RayTracerCuda
{
public:
    RayTracerCuda(Scene *theScene, RayTracer *rayTracer);
    virtual ~RayTracerCuda();

    void start();
    void buildBitmap();
    void loadObjects(BoundingSphere *spheres, Material *materials, int *numMaterials, float4 **verts, int *numVerts);
    void loadLights(LightCuda *output);
    void doViewTrans();

    Scene *theScene;
    RayTracer *rayTracer;
};

#endif // RAYTRACERCUDA_H
