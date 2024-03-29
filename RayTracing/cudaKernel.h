#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "Utilities/doublecolor.h"
#include "MatrixManipulation/double3d.h"

#define CHECK_ERROR(err) checkError(err, __FILE__, __LINE__)
#define CHECK_ERROR_FREE(err, nullObject) checkError(err, __FILE__, __LINE__, (void**)nullObject);

#define CHUNK 102

static const int EYE = 0;
static const int REFLECT = 0x1;
static const int INTERNAL_REFRACT = 0x2;
static const int EXTERNAL_REFRACT = 0x4;
static const float rhoAIR = 1.0;

struct Float3D
{
    float x;
    float y;
    float z;

    __device__ Float3D(){};

    /*__device__ Float3D()
    {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }*/

    __device__ Float3D(float nX, float nY, float nZ)
    {
        x = nX;
        y = nY;
        z = nZ;
    }

    __device__ Float3D(Double3D *from)
    {
        x = (float)from->x;
        y = (float)from->y;
        z = (float)from->z;
    }

    __device__ Float3D minus(Float3D *t1)
    {
        return Float3D(x - t1->x, y - t1->y, z - t1->z);
    }

    __device__ Float3D plus(Float3D *t1)
    {
        return Float3D(x + t1->x, y + t1->y, z + t1->z);
    }

    __device__ Float3D cross(Float3D *t1)
    {
        return Float3D((y)*(t1->z)-(t1->y)*(z), (z)*(t1->x)-(t1->z)*(x), (x)*(t1->y)-(t1->x)*(y));
    }

    __device__ Float3D sDiv(float s)
    {
        return Float3D(x/s, y/s, z/s);
    }

    __device__ Float3D sMult(float s)
    {
        return Float3D(s*x, s*y, s*z);
    }

    __device__ Float3D getUnit()
    {
        float s = sqrt(x*x+y*y+z*z);
        if (s > 0)
            return Float3D( x/s, y/s, z/s);
        return Float3D();
    }

    __device__ float dot(Float3D *t1)
    {
        return (x)*(t1->x) + (y)*(t1->y) + (z)*(t1->z);
    }

    __device__ float distanceTo(Float3D *point)
    {
        Float3D newVect = this->minus(point);
        return (float)sqrt(newVect.x * newVect.x + newVect.y * newVect.y + newVect.z * newVect.z);
    }

    __device__ void unitize()
    {
        float s = sqrt(x*x+y*y+z*z);
        if (s > 0)
        {
            x = x/s;
            y = y/s;
            z = z/s;
        }
    }
};

struct FloatColor
{
    float r;
    float g;
    float b;
    float a;

    __device__ FloatColor(){};

    __device__ FloatColor(float nR, float nG, float nB, float nA)
    {
        r = nR;
        g = nG;
        b = nB;
        a = nA;
    }

    __device__ FloatColor(DoubleColor *from)
    {
        r = (float)from->r;
        g = (float)from->g;
        b = (float)from->b;
        a = (float)from->a;
    }

    __device__ void plus(FloatColor other)
    {
        r = r + other.r;
        g = g + other.g;
        b = b + other.b;
    }

    __device__ void scale(float scaleValue)
    {
        r *= scaleValue;
        g *= scaleValue;
        b *= scaleValue;
    }
};

struct Bitmap
{
    unsigned char *data;
    float pixelWidth, pixelHeight;
    int width, height;
    Float3D firstPixel;
};

struct BoundingSphere
{
    Float3D center;
    float radius;
    float radiusSq;

    __device__ BoundingSphere()
    {
        center = Float3D(0.0, 0.0, 0.0);
        radius = 0;
        radiusSq = 0;
    }

    BoundingSphere(Float3D center, float radius)
    {
        this->center = center;
        this->radius = radius;
        radiusSq = radius * radius;
    }
};

struct Material
{
    FloatColor ka;
    FloatColor kd;
    FloatColor ks;
    FloatColor reflectivity;
    FloatColor refractivity;
    float refractiveIndex;
    float shiny;
};

struct Surface
{
    int numVerts, material;
    Float3D *vertArray;
    Float3D *viewNormArray;

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
    Float3D viewCenter;
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
    Float3D Rd;
    Float3D Ro;
    int flags;

    __device__ Ray(){};

    /*__device__ Ray()
    {
        Rd = Float3D();
        Ro = Float3D();
        flags = 0;
    }*/

    __device__ Ray(Float3D dir, Float3D origin)
    {
        Rd = dir;
        Ro = origin;
        flags = 0;
    }

    __device__ Ray(Float3D dir, Float3D origin, int type)
    {
        Rd = dir;
        Ro = origin;
        flags = type;
    }
};

struct LightCuda
{
    FloatColor ambient;
    FloatColor diffuse;
    FloatColor specular;
    Float3D viewPosition;
};

struct Options
{
    bool spheresOnly;
    bool reflections;
    bool refractions;
    bool shadows;
    int maxRecursiveDepth;
};

struct Intersect
{
    int materialIndex;
    float distance;
    bool backFacing;
    Mesh *theObj;
    Float3D point;
    Float3D normal;

    __device__ Intersect()
    {
        theObj = NULL;
        distance = 100000000.0;
    }

    __device__ Intersect(int matIndex, bool backFacing, Mesh *obj, Float3D point, Float3D normal, float distance)
    {
        materialIndex = matIndex;
        this->backFacing = backFacing;
        theObj = obj;
        this->point = point;
        this->normal = normal;
        this->distance = distance;
    }
};

__device__ bool intersectSphere(Ray *ray, const float radiusSq, const Float3D viewCenter, float &t);
__device__ bool intersectTriangle(Ray *ray, Float3D *tri, float *t, bool *backFacing, float *v, float *u);
__global__ void baseKrnl(Ray *rays, Bitmap bitmap);
__global__ void initIntersectKrnl(int numIntrs, Intersect *intrs);
__global__ void intersectSphereKrnl(Ray *rays, int numRays, Mesh *objects, int numObjects, bool spheresOnly, Intersect *intrs, bool *hits);
__global__ void intersectTriangleKrnl(Ray *rays, int numRays, Intersect *intrs, bool *hits, Mesh *theObj, Float3D *verts, Float3D *norms, int numVerts, int mat);
__global__ void shadeKrnl(Ray *rays, int numRays, Intersect *intrs, unsigned char *layer, LightCuda *lights, int numLights, Options options);
__global__ void reflectKrnl(Ray *rays, int numRays, Intersect *intrs);
__global__ void composeKrnl(Bitmap bitmap, unsigned char *layer, bool finalPass);

void cudaStart(Bitmap *bitmap, Mesh *objects, int numObjects, LightCuda *lights, int numLights, Options *options);
void checkError(cudaError_t error, const char *file, int line, void **nullObject = NULL);

#endif // CUDAKERNEL_H
