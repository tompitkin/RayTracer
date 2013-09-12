#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include "MatrixManipulation/double3d.h"
#include "Utilities/doublecolor.h"

#define CHECK_ERROR(err) checkError(err, __FILE__, __LINE__)
#define CHECK_ERROR_FREE(err, nullObject) checkError(err, __FILE__, __LINE__, (void**)nullObject);

static const int EYE = 0;
static const int REFLECT = 0x1;
static const int INTERNAL_REFRACT = 0x2;
static const int EXTERNAL_REFRACT = 0x4;
static const double rhoAIR = 1.0;

struct Bitmap
{
    unsigned char *data;
    double pixelWidth, pixelHeight;
    int width, height;
    Double3D firstPixel;
};

struct BoundingSphere
{
    Double3D center;
    double radius;
    double radiusSq;

    BoundingSphere()
    {
        radius = 0;
        radiusSq = 0;
    }

    BoundingSphere(Double3D center, double radius)
    {
        this->center = center;
        this->radius = radius;
        radiusSq = radius * radius;
    }
};

struct Material
{
    DoubleColor ka;
    DoubleColor kd;
    DoubleColor ks;
    DoubleColor reflectivity;
    DoubleColor refractivity;
    double refractiveIndex;
    double shiny;
};

struct Surface
{
    int numVerts, material;
    int *verts;

    ~Surface()
    {
        if (verts != NULL)
            delete [] verts;
    }
};

struct Mesh
{
    int numMats;
    int numSurfs;
    int numVerts;
    BoundingSphere boundingSphere;
    Double3D viewCenter;
    Material *materials;
    Surface *surfaces;
    Double3D *vertArray;
    Double3D *viewNormArray;

    ~Mesh()
    {
        if (materials != NULL)
            delete [] materials;
        if (surfaces != NULL)
            delete [] surfaces;
        if (vertArray != NULL)
            delete [] vertArray;
        if (viewNormArray != NULL)
            delete [] viewNormArray;
    }
};

struct Ray
{
    Double3D Rd;
    Double3D Ro;
    int flags;

    __device__ Ray(Double3D dir, Double3D origin)
    {
        Rd = dir;
        Ro = origin;
        flags = 0;
    }

    __device__ Ray(Double3D dir, Double3D origin, int type)
    {
        Rd = dir;
        Ro = origin;
        flags = type;
    }
};

struct LightCuda
{
    DoubleColor ambient;
    DoubleColor diffuse;
    DoubleColor specular;
    Double3D viewPosition;
};

struct Options
{
    bool spheresOnly;
    bool reflections;
    bool refractions;
    bool cull;
    bool shadows;
    int maxRecursiveDepth;
};

struct HitRecord
{
    double t, u, v;
    bool backfacing;
    Double3D intersectPoint, normal;

    __device__ HitRecord()
    {
        t = 0.0;
        u = 0.0;
        v = 0.0;
        backfacing = false;
    }
};

__device__ DoubleColor trace(Ray ray, int numRecurs);
__device__ DoubleColor shade(Mesh *theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, Ray ray, int numRecurs);
__device__ bool traceLightRay(Ray ray);
__device__ bool intersectSphere(Ray ray, Mesh *theObj, double *t);
__device__ bool intersectTriangle(Ray *ray, Mesh *theObj, int v1, int v2, int v3, HitRecord *hrec, bool cull);

void cudaStart(Bitmap *bitmap, Mesh *objects, int numObjects, LightCuda *lights, int numLights, Options *options);
void checkError(cudaError_t error, const char *file, int line, void **nullObject = NULL);

#endif // CUDAKERNEL_H
