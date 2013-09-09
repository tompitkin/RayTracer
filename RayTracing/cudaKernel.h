#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include "MatrixManipulation/double3d.h"
#include "Utilities/doublecolor.h"

#define CHECK_ERROR(err) checkError(err, __FILE__, __LINE__)

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

struct Mesh
{
    int numMats;
    BoundingSphere boundingSphere;
    Double3D viewCenter;
    Material *materials;
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

__device__ DoubleColor trace(Ray ray, int numRecurs);
__device__ DoubleColor shade(Mesh *theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, Ray ray, int numRecurs);
__device__ bool traceLightRay(Ray ray);
__device__ bool intersectSphere(Ray ray, Mesh *theObj, double *t);

void cudaStart(Bitmap *bitmap, Mesh *objects, int numObjects, LightCuda *lights, int numLights, Options *options);
void checkError(cudaError_t error, const char *file, int line);

#endif // CUDAKERNEL_H
