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
    int matCount = 0;
    for (int i = 0; i < numObjects; i++)
        matCount += theScene->objects.at(i)->numMats;
    int numMaterials[numObjects];
    int numVerts[numObjects];
    BoundingSphere spheres[numObjects];
    Material materials[matCount];
    float4 *verts[numObjects];
    float4 *triangles[numObjects];
    loadObjects(spheres, materials, numMaterials, verts, numVerts);

    printf("numObjects %d\n", numObjects);
    for (int x = 0; x < numObjects; x++)
    {
        printf("numVerts %d\n", numVerts[x]);
        triangles[x] = new float4[numVerts[x]];
        for (int y = 0; y < (numVerts[x] / 3); y++)
        {
            triangles[x][y*3] = verts[x][y*3];
            triangles[x][y*3+1] = make_float4(verts[x][y*3+1].x - verts[x][y*3].x, verts[x][y*3+1].y - verts[x][y*3].y, verts[x][y*3+1].z - verts[x][y*3].z, verts[x][y*3+1].w);
            triangles[x][y*3+1] = make_float4(verts[x][y*3+2].x - verts[x][y*3].x, verts[x][y*3+2].y - verts[x][y*3].y, verts[x][y*3+2].z - verts[x][y*3].z, verts[x][y*3+2].w);
            printf("Index %d\n", (int)triangles[x][y*3].w);
        }
    }

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

    cudaStart(&bitmap, numObjects, spheres, materials, numMaterials, triangles, numVerts, lights, numLights, &options);

    if (rayTracer->data != nullptr)
    {
        free(rayTracer->data);
        rayTracer->data = nullptr;
    }
    rayTracer->data = bitmap.data;

    for (int x = 0; x < numObjects; x++)
    {
        delete [] verts[x];
        delete [] triangles[x];
    }
}

void RayTracerCuda::loadObjects(BoundingSphere *spheres, Material *materials, int *numMaterials, float4 **verts, int *numVerts)
{
    int offset = 0;

    for (int i = 0; i < (int)theScene->objects.size(); i++)
    {
        PMesh *theObj = theScene->objects.at(i).get();
        spheres[i] = BoundingSphere(make_float3(theObj->viewCenter.x, theObj->viewCenter.y, theObj->viewCenter.z), theObj->boundingSphere->radius);

        numMaterials[i] = theObj->numMats;
        int matOffset = 0;
        for (int j = 0; j < i; j++)
            matOffset += numMaterials[j];
        for (int j = 0; j < theObj->numMats; j++, offset++)
        {
            materials[offset].ka = make_float3(theObj->materials[j].ka.r, theObj->materials[j].ka.g, theObj->materials[j].ka.b);
            materials[offset].kd = make_float3(theObj->materials[j].kd.r, theObj->materials[j].kd.g, theObj->materials[j].kd.b);
            materials[offset].ks = make_float3(theObj->materials[j].ks.r, theObj->materials[j].ks.g, theObj->materials[j].ks.b);
            materials[offset].reflectivity = make_float3(theObj->materials[j].reflectivity.r, theObj->materials[j].reflectivity.g, theObj->materials[j].reflectivity.b);
            materials[offset].refractivity = make_float3(theObj->materials[j].refractivity.r, theObj->materials[j].refractivity.g, theObj->materials[j].refractivity.b);
            materials[offset].refractiveIndex = theObj->materials[j].refractiveIndex;
            materials[offset].shiny = theObj->materials[j].shiny;
        }
        numVerts[i] = 0;
        for (PMesh::SurfCell *curSurf = theObj->surfHead.get(); curSurf != nullptr; curSurf = curSurf->next.get())
            numVerts[i] += curSurf->numVerts;

        verts[i] = new float4[numVerts[i]];

        int vertCount = 0;
        for (PMesh::SurfCell *curSurf = theObj->surfHead.get(); curSurf != nullptr; curSurf = curSurf->next.get(), vertCount = 0)
        {
            for (PMesh::PolyCell *curPoly = curSurf->polyHead.get(); curPoly != nullptr; curPoly = curPoly->next.get())
            {
                for (PMesh::VertListCell *curVert = curPoly->vert.get(); curVert != nullptr; curVert = curVert->next.get())
                {
                    verts[i][vertCount] = make_float4(theObj->vertArray.at(curVert->vert)->viewPos.x, theObj->vertArray.at(curVert->vert)->viewPos.y, theObj->vertArray.at(curVert->vert)->viewPos.z, matOffset + curSurf->material);
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
