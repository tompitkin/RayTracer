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
    bitmap.firstPixel = make_float3(theScene->camera->windowLeft + bitmap.pixelWidth / 2, theScene->camera->windowBottom + bitmap.pixelHeight / 2, -theScene->camera->near);

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
        output[i].boundingSphere = BoundingSphere(make_float3(theObj->boundingSphere->center.x, theObj->boundingSphere->center.y, theObj->boundingSphere->center.z), theObj->boundingSphere->radius);

        output[i].viewCenter = make_float3(theObj->viewCenter.x, theObj->viewCenter.y, theObj->viewCenter.z);

        output[i].numMats = theObj->numMats;
        output[i].materials = new Material[theObj->numMats];
        for (int j = 0; j < theObj->numMats; j++)
        {
            output[i].materials[j].ka = make_float3(theObj->materials[j].ka.r, theObj->materials[j].ka.g, theObj->materials[j].ka.b);
            output[i].materials[j].kd = make_float3(theObj->materials[j].kd.r, theObj->materials[j].kd.g, theObj->materials[j].kd.b);
            output[i].materials[j].ks = make_float3(theObj->materials[j].ks.r, theObj->materials[j].ks.g, theObj->materials[j].ks.b);
            output[i].materials[j].reflectivity = make_float3(theObj->materials[j].reflectivity.r, theObj->materials[j].reflectivity.g, theObj->materials[j].reflectivity.b);
            output[i].materials[j].refractivity = make_float3(theObj->materials[j].refractivity.r, theObj->materials[j].refractivity.g, theObj->materials[j].refractivity.b);
            output[i].materials[j].refractiveIndex = theObj->materials[j].refractiveIndex;
            output[i].materials[j].shiny = theObj->materials[j].shiny;
        }

        output[i].numSurfs = theObj->numSurf;
        output[i].surfaces = new Surface[theObj->numSurf];

        int surfCount = 0, vertCount = 0;
        for (PMesh::SurfCell *curSurf = theObj->surfHead.get(); curSurf != nullptr; curSurf = curSurf->next.get(), surfCount++, vertCount = 0)
        {
            output[i].surfaces[surfCount].material = curSurf->material;
            output[i].surfaces[surfCount].numVerts = curSurf->numVerts;
            output[i].surfaces[surfCount].vertArray = new float3[curSurf->numVerts];
            output[i].surfaces[surfCount].viewNormArray = new float3[curSurf->numVerts];

            for (PMesh::PolyCell *curPoly = curSurf->polyHead.get(); curPoly != nullptr; curPoly = curPoly->next.get())
            {
                for (PMesh::VertListCell *curVert = curPoly->vert.get(); curVert != nullptr; curVert = curVert->next.get())
                {
                    output[i].surfaces[surfCount].vertArray[vertCount] = make_float3(theObj->vertArray.at(curVert->vert)->viewPos.x, theObj->vertArray.at(curVert->vert)->viewPos.y, theObj->vertArray.at(curVert->vert)->viewPos.z);
                    output[i].surfaces[surfCount].viewNormArray[vertCount++] = make_float3(theObj->viewNormArray.at(curVert->vert).x, theObj->viewNormArray.at(curVert->vert).y, theObj->viewNormArray.at(curVert->vert).z);
                }
            }
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
            output[count].ambient.x = curLight->ambient[0];
            output[count].ambient.y = curLight->ambient[1];
            output[count].ambient.z = curLight->ambient[2];
            output[count].diffuse.x = curLight->diffuse[0];
            output[count].diffuse.y = curLight->diffuse[1];
            output[count].diffuse.z = curLight->diffuse[2];
            output[count].specular.x = curLight->specular[0];
            output[count].specular.y = curLight->specular[1];
            output[count].specular.z = curLight->specular[2];
            Double3D viewPos = Double3D(curLight->position[0], curLight->position[1], curLight->position[2]).preMultiplyMatrix(theScene->camera->viewMat);
            output[count].viewPosition = make_float3(viewPos.x, viewPos.y, viewPos.z);
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
