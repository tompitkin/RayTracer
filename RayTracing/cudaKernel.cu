#include "cudaKernel.h"
#include "stdio.h"

__device__ int numObjects;
__device__ Mesh *objects;
__device__ Options *options;

__global__ void kernel(Bitmap bitmap, Mesh *d_objects, int d_numObjects, Options d_options)
{
    //Map from threadIdx & blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    if (offset < (bitmap.width * bitmap.height))
    {
        numObjects = d_numObjects;
        objects = d_objects;
        options = &d_options;

        Double3D point(bitmap.firstPixel);
        point.x += (offset % bitmap.width) * bitmap.pixelWidth;
        point.y += ((offset - x) / bitmap.width) * bitmap.pixelHeight;
        Ray ray(point.getUnit(), Double3D(), EYE);

        DoubleColor rgb = trace(ray, 0);

        bitmap.data[offset*3 + 0] = rgb.r;
        bitmap.data[offset*3 + 1] = rgb.g;
        bitmap.data[offset*3 + 2] = rgb.b;

        objects = NULL;
        options = NULL;
    }
}

__device__ DoubleColor trace(Ray ray, int numRecurs)
{
    double t = 0.0;
    double intersectDist = 0.0;
    double minDist = 100000000.0;
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
        return DoubleColor(255, 0.0, 0.0, 1.0);
    }
    else
    {
        return DoubleColor(0.0, 0.0, 0.0, 1.0);
    }
}

__device__ DoubleColor shade(Mesh *theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, Ray ray, int numRecurs)
{
    DoubleColor ambColor;
    DoubleColor shadeColor;
    DoubleColor reflColor;
    DoubleColor refrColor;
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

    Double3D lightPos;
    Double3D lightViewPos;
    Double3D R;
    Double3D L;
    Double3D V;
    ambColor.r = Ka.r * theScene->lights->lights[0].ambient[0];
    ambColor.g = Ka.g * theScene->lights->lights[0].ambient[1];
    ambColor.b = Ka.b * theScene->lights->lights[0].ambient[2];
    shadeColor.plus(ambColor);

    V = Double3D(0.0, 0.0, 0.0).minus(point);
    V.unitize();

    if (ray.flags == EYE && backFacing && !theScene->cull)
        trueNormal = inv_normal;
    else if (ray.flags == INTERNAL_REFRACT && backFacing)
        trueNormal = inv_normal;
    else
        trueNormal = normal;

    Lights::Light *curLight;
    for (int i = 0; i < 8; i++)
    {
        bool obstructed = false;
        curLight = &theScene->lights->lights[i];
        if (curLight->lightSwitch == 0)
            continue;
        lightPos = Double3D(curLight->position[0], curLight->position[1], curLight->position[2]);
        lightViewPos = lightPos.preMultiplyMatrix(theScene->camera->viewMat);
        if(rayTracer->shadows)
        {
            Double3D Rd(lightViewPos.minus(point));
            Rd.unitize();
            Ray shadowRay = Ray(Double3D(Rd), Double3D(point));
            if (traceLightRay(shadowRay, theObj))
                obstructed = true;
        }
        if (obstructed)
            continue;

        L = lightViewPos.minus(point);
        L.unitize();
        double LdotN = L.dot(trueNormal);
        LdotN = max(0.0, LdotN);
        DoubleColor diffComponent(0.0, 0.0, 0.0, 1.0);
        if (LdotN > 0.0)
            diffComponent.plus(DoubleColor(curLight->diffuse[0]*Kd.r*LdotN, curLight->diffuse[1]*Kd.g*LdotN, curLight->diffuse[2]*Kd.b*LdotN, 1.0));
        shadeColor.plus(diffComponent);

        Double3D Pr = trueNormal.sMult(LdotN);
        Double3D sub = Pr.sMult(2.0);
        R = L.sMult(-1.0).plus(sub);
        R.unitize();
        double RdotV = R.dot(V);
        RdotV = max(0.0, RdotV);
        if (RdotV > 1.0)
            fprintf(stdout, "RdotV: %f\n", RdotV);
        double cosPhiPower = 0.0;
        if (RdotV > 0.0)
            cosPhiPower = pow(RdotV, theObj->materials[materialIndex].shiny);
        DoubleColor specComponent(curLight->specular[0]*Ks.r*cosPhiPower, curLight->specular[1]*Ks.g*cosPhiPower, curLight->specular[2]*Ks.b*cosPhiPower, 1.0);
        shadeColor.plus(specComponent);
    }
    if (numRecurs >= rayTracer->maxRecursiveDepth)
        return shadeColor;

    if (refractions)
    {
        double rhoNew, rhoOld;
        Double3D norm;
        if (ray.flags == INTERNAL_REFRACT)
        {
            rhoOld = theObj->materials[theObj->objNumber].refractiveIndex;
            rhoNew = rhoAIR;
            norm = Double3D(inv_normal);
        }
        else
        {
            rhoNew = theObj->materials[theObj->objNumber].refractiveIndex;
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
        return shadeColor;
}

void cudaStart(Bitmap *bitmap, Mesh *objects, int numObjects, Options *options)
{
    unsigned char *d_bitmap;
    unsigned char *h_bitmap;
    Mesh *d_objects;

    CHECK_ERROR(cudaMalloc((void**)&d_bitmap, bitmap->width * bitmap->height * 3));
    h_bitmap = (unsigned char*)malloc(sizeof(unsigned char) * (bitmap->width * bitmap->height * 3));

    bitmap->data = d_bitmap;

    CHECK_ERROR(cudaMalloc((void**)&d_objects, sizeof(Mesh) * numObjects));
    CHECK_ERROR(cudaMemcpy(d_objects, objects, sizeof(Mesh) * numObjects, cudaMemcpyHostToDevice));

    dim3 blocks((bitmap->width+15)/16, (bitmap->height+15)/16);
    dim3 threads(16, 16);
    kernel<<<blocks, threads>>>(*bitmap, d_objects, numObjects, *options);

    CHECK_ERROR(cudaMemcpy(h_bitmap, d_bitmap, bitmap->width * bitmap->height * 3, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_bitmap));
    CHECK_ERROR(cudaFree(d_objects));

    bitmap->data = h_bitmap;
}

void checkError(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}
