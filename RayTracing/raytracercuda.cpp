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

    doViewTrans();

    int numObjects = theScene->objects.size();
    Mesh objects[numObjects];
    loadObjects(objects);

    int numLights = 0;
    for (int x = 0; x < 8; x++)
    {
        if (theScene->lights->lights[x].lightSwitch == Lights::Light::ON)
            numLights++;
    }
    LightCuda lights[numLights];
    loadLights(lights);

    Options options;
    options.spheresOnly = rayTracer->spheresOnly;
    options.reflections = rayTracer->reflections;
    options.refractions = rayTracer->refractions;
    options.cull = theScene->cull;
    options.shadows = rayTracer->shadows;
    options.maxRecursiveDepth = rayTracer->maxRecursiveDepth;

    cudaStart(&bitmap, objects, numObjects, lights, numLights, &options);

    if (rayTracer->data != nullptr)
    {
        free(rayTracer->data);
        rayTracer->data = nullptr;
    }
    rayTracer->data = bitmap.data;
}

void RayTracerCuda::loadObjects(Mesh *output)
{
    for (int i = 0; i < (int)theScene->objects.size(); i++)
    {
        PMesh *theObj = theScene->objects.at(i).get();
        output[i].boundingSphere = BoundingSphere(theObj->boundingSphere->center, theObj->boundingSphere->radius);

        output[i].viewCenter = theObj->viewCenter;

        output[i].numMats = theObj->numMats;
        output[i].materials = new Material[theObj->numMats];
        for (int j = 0; j < theObj->numMats; j++)
        {
            output[i].materials[j].ka = theObj->materials[j].ka;
            output[i].materials[j].kd = theObj->materials[j].kd;
            output[i].materials[j].ks = theObj->materials[j].ks;
            output[i].materials[j].reflectivity = theObj->materials[j].reflectivity;
            output[i].materials[j].refractivity = theObj->materials[j].refractivity;
            output[i].materials[j].refractiveIndex = theObj->materials[j].refractiveIndex;
            output[i].materials[j].shiny = theObj->materials[j].shiny;
        }
    }
}

void RayTracerCuda::loadLights(LightCuda *output)
{
    int count = 0;
    Lights::Light *curLight;
    for (int i = 0; i < 8; i++)
    {
        if (theScene->lights->lights[i].lightSwitch == Lights::Light::ON)
        {
            curLight = &theScene->lights->lights[i];
            output[count].ambient.r = curLight->ambient[0];
            output[count].ambient.g = curLight->ambient[1];
            output[count].ambient.b = curLight->ambient[2];
            output[count].diffuse.r = curLight->diffuse[0];
            output[count].diffuse.g = curLight->diffuse[1];
            output[count].diffuse.b = curLight->diffuse[2];
            output[count].specular.r = curLight->specular[0];
            output[count].specular.g = curLight->specular[1];
            output[count].specular.b = curLight->specular[2];
            output[count].viewPosition = Double3D(curLight->position[0], curLight->position[1], curLight->position[2]).preMultiplyMatrix(theScene->camera->viewMat);
            count++;
        }
    }
}

void RayTracerCuda::doViewTrans()
{
    vector<double> modelViewInvTranspose;
    for (int obj = 0; obj < (int)theScene->objects.size(); obj++)
    {
        PMesh *thisObj = theScene->objects.at(obj).get();
        vector<double> modelView = MatrixOps::newIdentity();
        modelView = MatrixOps::multMat(modelView, thisObj->modelMat);
        modelView = MatrixOps::multMat(theScene->camera->viewMat, modelView);
        modelViewInvTranspose = MatrixOps::inverseTranspose(modelView);
        Double3D transNorm;
        for (int vert = 0; vert < thisObj->numVerts; vert++)
        {
            thisObj->vertArray.at(vert)->viewPos = thisObj->vertArray.at(vert)->worldPos.preMultiplyMatrix(modelView);
            transNorm = thisObj->vertNormArray.at(vert)->preMultiplyMatrix(modelViewInvTranspose);
            thisObj->viewNormArray.insert(thisObj->viewNormArray.begin()+vert, transNorm);
        }
        thisObj->viewCenter = thisObj->center.preMultiplyMatrix(theScene->camera->viewMat);
        thisObj->calcViewPolyNorms();
    }
}
