#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "Utilities/doublecolor.h"
#include "MatrixManipulation/double3d.h"
#include "cutil_math.h"

#define CHECK_ERROR(err) checkError(err, __FILE__, __LINE__)
#define CHECK_ERROR_FREE(err, nullObject) checkError(err, __FILE__, __LINE__, (void**)nullObject);

#define CHUNK 102

static const int EYE = 0;
static const int REFLECT = 0x1;
static const int INTERNAL_REFRACT = 0x2;
static const int EXTERNAL_REFRACT = 0x4;
static const float rhoAIR = 1.0;

struct Bitmap
{
    unsigned char *data;
    float pixelWidth, pixelHeight;
    int width, height;
    float3 firstPixel;
};

struct BoundingSphere
{
    float3 center;
    float radius;
    float radiusSq;

    __device__ BoundingSphere()
    {
        center = make_float3(0.0, 0.0, 0.0);
        radius = 0;
        radiusSq = 0;
    }

    BoundingSphere(float3 center, float radius)
    {
        this->center = center;
        this->radius = radius;
        radiusSq = radius * radius;
    }
};

struct Material
{
    float3 ka;
    float3 kd;
    float3 ks;
    float3 reflectivity;
    float3 refractivity;
    float refractiveIndex;
    float shiny;
};

struct Surface
{
    int numVerts, material;
    float3 *vertArray;
    float3 *viewNormArray;

    ~Surface()
    {
        if (vertArray != NULL)
            delete [] vertArray;
        if (viewNormArray != NULL)
            delete [] viewNormArray;
    }
};

struct Mesh
{
    int numMats;
    int numSurfs;
    BoundingSphere boundingSphere;
    float3 viewCenter;
    Material *materials;
    Surface *surfaces;


    ~Mesh()
    {
        if (materials != NULL)
            delete [] materials;
        if (surfaces != NULL)
            delete [] surfaces;
    }
};

struct Ray
{
    float3 Rd;
    float3 Ro;
    int flags;

    __device__ Ray(){};

    /*__device__ Ray()
    {
        Rd = Float3D();
        Ro = Float3D();
        flags = 0;
    }*/

    __device__ Ray(float3 dir, float3 origin)
    {
        Rd = dir;
        Ro = origin;
        flags = 0;
    }

    __device__ Ray(float3 dir, float3 origin, int type)
    {
        Rd = dir;
        Ro = origin;
        flags = type;
    }
};

struct LightCuda
{
    float3 ambient;
    float3 diffuse;
    float3 specular;
    float3 viewPosition;
};

struct Options
{
    bool spheresOnly;
    bool reflections;
    bool refractions;
    bool shadows;
    int maxRecursiveDepth;
};

struct HitRecord
{
    float t, u, v;
    bool backfacing;
    float3 intersectPoint, normal;
};

struct Intersect
{
    int materialIndex;
    float distance;
    bool backFacing;
    Mesh *theObj;
    float3 point;
    float3 normal;

    __device__ Intersect()
    {
        theObj = NULL;
        distance = 100000000.0;
    }

    __device__ Intersect(int matIndex, bool backFacing, Mesh *obj, float3 point, float3 normal, float distance)
    {
        materialIndex = matIndex;
        this->backFacing = backFacing;
        theObj = obj;
        this->point = point;
        this->normal = normal;
        this->distance = distance;
    }
};

__device__ bool intersectSphere(Ray *ray, float radiusSq, float3 viewCenter, float *t);
__device__ bool intersectTriangle(Ray *ray, float3 *v1, float3 *n1, HitRecord *hrec);
__global__ void baseKrnl(Ray *rays, Bitmap bitmap);
__global__ void initIntersectKrnl(int numIntrs, Intersect *intrs);
__global__ void intersectSphereKrnl(Ray *rays, int numRays, Mesh *objects, int numObjects, bool spheresOnly, Intersect *intrs, bool *hits);
__global__ void intersectTriangleKrnl(Ray *rays, int numRays, Intersect *intrs, bool *hits, Mesh *theObj, float3 *verts, float3 *norms, int numVerts, int mat);
__global__ void shadeKrnl(Ray *rays, int numRays, Intersect *intrs, unsigned char *layer, LightCuda *lights, int numLights, Options options, bool finalPass);
__global__ void composeKrnl(Bitmap bitmap, unsigned char *layer, bool finalPass);

void cudaStart(Bitmap *bitmap, Mesh *objects, int numObjects, LightCuda *lights, int numLights, Options *options);
void checkError(cudaError_t error, const char *file, int line, void **nullObject = NULL);

#endif // CUDAKERNEL_H
