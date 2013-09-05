#include "cudaKernel.h"
#include "stdio.h"

__device__ int numObjects;
__device__ Mesh *objects;
__device__ int numLights;
__device__ LightCuda * lights;
__device__ Options *options;

__global__ void kernel(Bitmap bitmap, Mesh *d_objects, int d_numObjects, LightCuda *d_lights, int d_numLights, Options d_options)
{
    //Map from threadIdx & blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < (bitmap.width * bitmap.height))
    {
        numObjects = d_numObjects;
        objects = d_objects;
        numLights = d_numLights;
        lights = d_lights;
        options = &d_options;

        Double3D point(bitmap.firstPixel);
        point.x += (offset % bitmap.width) * bitmap.pixelWidth;
        point.y += ((offset - x) / bitmap.width) * bitmap.pixelHeight;
        Ray ray(point.getUnit(), Double3D(), EYE);

        DoubleColor rgb = trace(ray, 0);

        bitmap.data[offset*3 + 0] = (int) (rgb.r * 255);
        bitmap.data[offset*3 + 1] = (int) (rgb.g * 255);
        bitmap.data[offset*3 + 2] = (int) (rgb.b * 255);
    }
}

__device__ DoubleColor trace(Ray ray, int numRecurs)
{
    double t = 0.0;
    double intersectDist = 0.0;
    double minDist = 100000000.0;
    int minMatIndex = 0;
    bool minBackfacing = false;
    Mesh *minObj = NULL;
    Double3D minIntPt;
    Double3D minNormal;
    Double3D intersectPt;
    Double3D normal;
    Double3D origin;

    for (int obj = 0; obj < numObjects; obj++)
    {
        if (ray.intersectSphere(&objects[obj], &t))
        {
            if (abs(t) < 0.00001)
                continue;
            if (options->spheresOnly)
            {
                intersectPt = Double3D((ray.Ro.x+(ray.Rd.x*t)), (ray.Ro.y+(ray.Rd.y*t)), (ray.Ro.z+(ray.Rd.z*t)));
                normal = (intersectPt.minus(objects[obj].viewCenter).sDiv(objects[obj].boundingSphere.radius));
                normal.unitize();
                intersectDist = origin.distanceTo(intersectPt);
                if (intersectDist < minDist)
                {
                    minDist = intersectDist;
                    minObj = &objects[obj];
                    minIntPt = Double3D(intersectPt);
                    minNormal = Double3D(normal);
                }
            }
        }
    }
    if (minObj != NULL)
    {
        return shade(minObj, minIntPt, minNormal, minMatIndex, minBackfacing, ray, numRecurs);
    }
    else
    {
        return DoubleColor(0.0, 0.0, 0.0, 1.0);
    }
}

__device__ DoubleColor shade(Mesh *theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, Ray ray, int numRecurs)
{
    DoubleColor shadeColor;
    DoubleColor reflColor;
    DoubleColor refrColor;
    DoubleColor Ka;
    DoubleColor Kd;
    DoubleColor Ks;
    Double3D inv_normal = normal.sMult(-1.0);
    Double3D trueNormal;
    bool reflections;
    bool refractions;
    double reflectivity;
    double refractivity;

    if (ray.flags == INTERNAL_REFRACT)
        reflections = false;
    else
        reflections = options->reflections;
    refractions = options->refractions;
    reflectivity = theObj->materials[materialIndex].reflectivity.r;
    refractivity = theObj->materials[materialIndex].refractivity.r;

    Ka = theObj->materials[materialIndex].ka;
    Kd = theObj->materials[materialIndex].kd;
    Ks = theObj->materials[materialIndex].ks;

    Double3D R;
    Double3D L;
    Double3D V;

    shadeColor.plus(lights[0].ambient);

    V = Double3D(0.0, 0.0, 0.0).minus(point);
    V.unitize();

    if (ray.flags == EYE && backFacing && !options->cull)
        trueNormal = inv_normal;
    else if (ray.flags == INTERNAL_REFRACT && backFacing)
        trueNormal = inv_normal;
    else
        trueNormal = normal;

    LightCuda *curLight;
    for (int i = 0; i < numLights; i++)
    {
        bool obstructed = false;
        curLight = &lights[i];

        if(options->shadows)
        {
            Double3D Rd(curLight->viewPosition.minus(point));
            Rd.unitize();
            Ray shadowRay = Ray(Double3D(Rd), Double3D(point));
            if (traceLightRay(shadowRay))
                obstructed = true;
        }
        if (obstructed)
            continue;

        L = curLight->viewPosition.minus(point);
        L.unitize();
        double LdotN = L.dot(trueNormal);
        LdotN = max(0.0, LdotN);
        DoubleColor diffComponent(0.0, 0.0, 0.0, 1.0);
        if (LdotN > 0.0)
            diffComponent.plus(DoubleColor(curLight->diffuse.r*Kd.r*LdotN, curLight->diffuse.g*Kd.g*LdotN, curLight->diffuse.b*Kd.b*LdotN, 1.0));
        shadeColor.plus(diffComponent);

        Double3D Pr = trueNormal.sMult(LdotN);
        Double3D sub = Pr.sMult(2.0);
        R = L.sMult(-1.0).plus(sub);
        R.unitize();
        double RdotV = R.dot(V);
        RdotV = max(0.0, RdotV);
        double cosPhiPower = 0.0;
        if (RdotV > 0.0)
            cosPhiPower = pow(RdotV, theObj->materials[materialIndex].shiny);
        DoubleColor specComponent(curLight->specular.r*Ks.r*cosPhiPower, curLight->specular.g*Ks.g*cosPhiPower, curLight->specular.b*Ks.b*cosPhiPower, 1.0);
        shadeColor.plus(specComponent);
    }
    if (numRecurs >= options->maxRecursiveDepth)
        return shadeColor;

    /*if (refractions)
    {
        double rhoNew, rhoOld;
        Double3D norm;
        if (ray.flags == INTERNAL_REFRACT)
        {
            rhoOld = theObj->materials[materialIndex].refractiveIndex;
            rhoNew = rhoAIR;
            norm = Double3D(inv_normal);
        }
        else
        {
            rhoNew = theObj->materials[materialIndex].refractiveIndex;
            rhoOld = rhoAIR;
            norm = Double3D(normal);
        }
        double rhoOldSq = rhoOld * rhoOld;
        double rhoNewSq = rhoNew * rhoNew;
        Double3D d = ray.Rd;
        double dDotn = d.dot(norm);
        Double3D term1 = d.minus(norm.sMult(dDotn)).sMult(rhoOld);
        term1 = term1.sDiv(rhoNew);
        double sqrtOp = 1.0 - ((rhoOldSq*(1.0 - dDotn * dDotn))/rhoNewSq);
        if (sqrtOp < 0.0)
        {
            reflectivity = reflectivity + refractivity;
            reflections = true;
            refractions = false;
        }
        else
        {
            double root = sqrt(sqrtOp);
            Double3D term2 = norm.sMult(root);
            Double3D t = term1.minus(term2);
            t.unitize();
            Ray newRay = Ray(Double3D(), Double3D());
            if (ray.flags == INTERNAL_REFRACT)
                newRay = Ray(t, point, EXTERNAL_REFRACT);
            else
                newRay = Ray(t, point, INTERNAL_REFRACT);
            refrColor = trace(newRay, numRecurs+1);
        }
    }

    if (reflections)
    {
        Double3D Pr = trueNormal.sMult(ray.Rd.dot(trueNormal));
        Double3D sub = Pr.sMult(2.0);
        Double3D refVect = ray.Rd.minus(sub);
        refVect.unitize();

        Ray reflRay = Ray(refVect, point, REFLECT);
        reflColor = trace(reflRay, numRecurs+1);
    }

    DoubleColor rtnColor;
    double shadeWeight;

    if (reflections && !refractions)
    {
        shadeWeight = 1.0 - reflectivity;
        reflColor.scale(reflectivity);
        shadeColor.scale(shadeWeight);
        rtnColor.plus(shadeColor);
        rtnColor.plus(reflColor);
        return rtnColor;
    }
    else if (reflections && refractions)
    {
        shadeWeight = 1.0 - (reflectivity + refractivity);
        reflColor.scale(refractivity);
        reflColor.scale(reflectivity);
        shadeColor.scale(shadeWeight);
        rtnColor.plus(refrColor);
        rtnColor.plus(shadeColor);
        rtnColor.plus(reflColor);
        return rtnColor;
    }
    else if (!reflections && refractions)
    {
        shadeWeight = 1.0 - refractivity;
        reflColor.scale(refractivity);
        shadeColor.scale(shadeWeight);
        rtnColor.plus(refrColor);
        rtnColor.plus(shadeColor);
        return rtnColor;
    }
    else
        return shadeColor;*/
}

__device__ bool traceLightRay(Ray ray)
{
    double t = 0.0;
    for (int obj = 0; obj < numObjects; obj++)
    {
        if (ray.intersectSphere(&objects[obj], &t))
        {
            if (abs(t) < 0.0001)
                return false;
            else
                return true;
        }
    }
    return false;
}

void cudaStart(Bitmap *bitmap, Mesh *objects, int numObjects, LightCuda *lights, int numLights, Options *options)
{
    unsigned char *d_bitmap;
    unsigned char *h_bitmap;
    Mesh *d_objects;
    Mesh *h_objects;
    LightCuda *d_lights;

    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, bitmap->width * bitmap->height * 3));
    h_bitmap = (unsigned char*)malloc(sizeof(unsigned char) * (bitmap->width * bitmap->height * 3));

    bitmap->data = d_bitmap;

    h_objects = (Mesh *)malloc(sizeof(Mesh) *  numObjects);
    memcpy(h_objects, objects, sizeof(Mesh) * numObjects);

    for (int x = 0; x < numObjects; x++)
    {
        CHECK_ERROR(cudaMalloc((void **)&h_objects[x].materials, sizeof(Material) * h_objects[x].numMats));
        CHECK_ERROR(cudaMemcpy(h_objects[x].materials, objects[x].materials, sizeof(Material) * h_objects[x].numMats, cudaMemcpyHostToDevice));
    }

    CHECK_ERROR(cudaMalloc((void**)&d_objects, sizeof(Mesh) * numObjects));
    CHECK_ERROR(cudaMemcpy(d_objects, h_objects, sizeof(Mesh) * numObjects, cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc((void**)&d_lights, sizeof(LightCuda) * numLights));
    CHECK_ERROR(cudaMemcpy(d_lights, lights, sizeof(LightCuda) * numLights, cudaMemcpyHostToDevice));

    dim3 blocks((bitmap->width+15)/16, (bitmap->height+15)/16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(*bitmap, d_objects, numObjects, d_lights, numLights, *options);

    CHECK_ERROR(cudaMemcpy(h_bitmap, d_bitmap, bitmap->width * bitmap->height * 3, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_bitmap));

    for (int x = 0; x < numObjects; x++)
        CHECK_ERROR(cudaFree(h_objects[x].materials));
    CHECK_ERROR(cudaFree(d_objects));

    CHECK_ERROR(cudaFree(d_lights));

    bitmap->data = h_bitmap;

    free(h_objects);
}

void checkError(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}
