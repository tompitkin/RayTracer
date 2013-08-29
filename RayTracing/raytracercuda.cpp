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
    loadObjects(objects, theScene);

    Options options;
    options.spheresOnly = rayTracer->spheresOnly;
    options.reflections = rayTracer->reflections;
    options.refractions = rayTracer->refractions;

    cudaStart(&bitmap, objects, numObjects, &options);

    if (rayTracer->data != nullptr)
    {
        free(rayTracer->data);
        rayTracer->data = nullptr;
    }
    rayTracer->data = bitmap.data;
}

void RayTracerCuda::loadObjects(Mesh *output, Scene *input)
{
    for (int i = 0; i < (int)input->objects.size(); i++)
    {
        PMesh *theObj = input->objects.at(i).get();
        output[i].boundingSphere = BoundingSphere(theObj->boundingSphere->center, theObj->boundingSphere->radius);

        output[i].viewCenter = theObj->viewCenter;
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
