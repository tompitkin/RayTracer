#include "pmesh.h"
#include "scene.h"

PMesh::PMesh(Scene *aScene)
{
    theScene = aScene;
    objNumber = -1;
    numVerts = 0;
    numNorms = 0;
    numTex = 0;
    numSurf = 0;
    numPolys = 0;
    numMats = 1;
    materials = nullptr;
    active = true;

    firstDraw = true;

    modelMatUniform = new Matrix4Uniform(nullptr, "modelMat");

    useAmbTexUniform = new IntUniform(0, "materialSettings.ambTex");
    useDiffTexUniform = new IntUniform(0, "materialSettings.diffTex");
    useSpecTexUniform = new IntUniform(0, "materialSettings.specTex");
}

PMesh::~PMesh()
{
    theScene = nullptr;
    delete boundingSphere;
    delete modelMatUniform;
    delete useAmbTexUniform;
    delete useDiffTexUniform;
    delete useSpecTexUniform;
    delete file;
    delete []bufferIDs;
    delete []VAOIDs;
    delete []modelMat;
}

Sphere *PMesh::calcBoundingSphere()
{
    double greatest = 0.0;
    double dist;
    for (int i = 0; i < numVerts; i++)
    {
        dist = vertArray.at(i)->worldPos.distanceTo(center);
        if (dist > greatest)
            greatest = dist;
    }
    fprintf(stdout, "Sphere: radius = %f\n",greatest);
    Sphere *retVal = new Sphere(center, greatest, this);
    return retVal;
}

void PMesh::calcPolyNorms()
{
    Double3D *vector1, *vector2, *cross;
    shared_ptr<SurfCell> curSurf = surfHead;
    shared_ptr<PolyCell> curPoly;
    shared_ptr<VertListCell> curVert, temp;

    while (curSurf != nullptr)
    {
        curPoly = curSurf->polyHead;
        while (curPoly != nullptr)
        {
            temp = curVert = curPoly->vert;
            while (temp != nullptr)
            {
                temp->norm = temp->vert;
                temp = temp->next;
            }
            if (curVert != nullptr)
            {
                if (curVert->next != nullptr)
                {
                    if (curVert->next->next != nullptr)
                    {
                        vector1 = new Double3D(vertArray.at(curVert->next->vert)->worldPos);
                        vector1 = new Double3D(vector1->minus(vertArray.at(curVert->vert)->worldPos));
                        vector2 = new Double3D(vertArray.at(curVert->next->next->vert)->worldPos);
                        vector2 = new Double3D(vector2->minus(vertArray.at(curVert->vert)->worldPos));
                        cross = new Double3D(vector1->cross(vector2));
                        curPoly->normal = new Double3D(cross->getUnit());
                    }
                }
            }
            else
                curPoly->normal = new Double3D();
            curPoly = curPoly->next;
        }
        curSurf = curSurf->next;
    }
    fprintf(stdout, "Polygon Normals calculated\n");
}

void PMesh::calcVertNorms()
{
    Double3D norm;
    shared_ptr<PolyListCell> curPolyLC;
    vertNormArray.resize(this->numVerts);
    this->numNorms = this->numVerts;
    shared_ptr<SurfCell> curSurf = surfHead;
    while (curSurf != nullptr)
    {
        shared_ptr<PolyCell> curPoly = curSurf->polyHead;
        while (curPoly != nullptr)
        {
            shared_ptr<VertListCell> curVertLC = curPoly->vert;
            while (curVertLC != nullptr)
            {
                curPolyLC = vertArray.at(curVertLC->vert)->polys;
                if (curPolyLC != nullptr)
                {
                    norm = new Double3D();
                    while (curPolyLC != nullptr)
                    {
                        if (curPolyLC->poly != nullptr)
                            norm = norm.plus(curPolyLC->poly->normal);
                        curPolyLC = curPolyLC->next;
                    }

                    vertNormArray.at(curVertLC->vert) = shared_ptr<Double3D>(new Double3D(norm.getUnit()));
                    curVertLC->norm = curVertLC->vert;
                }
                else
                {
                    vertNormArray.at(curVertLC->vert) = shared_ptr<Double3D>(new Double3D(curPoly->normal));
                    curVertLC->norm = curVertLC->vert;
                }
                curVertLC = curVertLC->next;
            }
            curPoly = curPoly->next;
        }
        curSurf = curSurf->next;
    }
    fprintf(stdout, "Vertex Normals calculated\n");
}

void PMesh::updateUniforms()
{
    if (glIsProgram(theShader->progID))
        glUseProgram(theShader->progID);
    else
    {
        fprintf(stderr, "updateUniforms: %s is NOT a program\n", theShader->baseName.c_str());
        return;
    }
    for (int i = 0; i < (int)theShader->uniformList.size(); i++)
    {
        Uniform *thisOne = theShader->uniformList.at(i);
        if (thisOne->needsUpdate)
            thisOne->update(theShader->progID);
        thisOne->needsUpdate=false;
    }
}

void PMesh::draw(Camera *camera)
{
    SurfCell *curSurf;
    fprintf(stdout, "Drawing %s\n", qPrintable(this->objName));

    if (firstDraw)
    {
        bufferIDs = new GLuint[numSurf*3];
        VAOIDs = new GLuint[numSurf];
        glGenBuffers(numSurf*3, bufferIDs);
        glGenVertexArrays(numSurf, VAOIDs);
        useAmbTexUniform->update(theShader->progID);
        useDiffTexUniform->update(theShader->progID);
        useSpecTexUniform->update(theShader->progID);
    }
    if (active)
    {
        int previousMat = -1;
        if (modelMatUniform->theBuffer == nullptr)
            modelMatUniform->theBuffer = new GLfloat[16];
        copy(modelMat, modelMat+16, modelMatUniform->theBuffer);
        GL::checkGLErrors(" updateMatrix4Uniform(before): ");
        modelMatUniform->update(theShader->progID);
        double *modelViewMat = MatrixOps::newIdentity();
        modelViewMat = MatrixOps::multMat(modelViewMat, camera->viewMat);
        modelViewMat = MatrixOps::multMat(modelViewMat, modelMat);
        //MatrixOps::inverseTranspose(modelViewMat);
    }
}

PMesh::VertCell::VertCell()
{
    polys = nullptr;
}

PMesh::VertCell::VertCell(const VertCell *from)
{
    worldPos = Double3D(from->worldPos);
}

PMesh::VertCell::~VertCell()
{
    polys.reset();
}


PMesh::PolyCell::PolyCell()
{
    numVerts = 0;
    vert = nullptr;
    culled = false;
    next = nullptr;
}

PMesh::PolyCell::~PolyCell()
{
    vert.reset();
    next.reset();
}


PMesh::VertListCell::VertListCell()
{
    vert = -1;
    norm = -1;
    tex = -1;
    tan = -1;
    bitan = -1;
    next = nullptr;
}

PMesh::VertListCell::~VertListCell()
{
    next.reset();
}


PMesh::PolyListCell::PolyListCell()
{
    poly = nullptr;
    next = nullptr;
}

PMesh::PolyListCell::~PolyListCell()
{
    poly.reset();
    next.reset();
}


PMesh::SurfCell::SurfCell(QString name)
{
    this->name = name;
    numPoly = 0;
    polyHead = nullptr;
    material = 0;
    numVerts = 0;
}

PMesh::SurfCell::~SurfCell()
{
    polyHead.reset();
    delete vertexBuffer;
    delete normalBuffer;
    delete texCoordBuffer;
    delete tangentBuffer;
    delete bitangentBuffer;
}


PMesh::MaterialCell::MaterialCell()
{
    materialName = "default";
    ka = DoubleColor(0.2, 0.2, 0.2, 1.0);
    kd = DoubleColor(0.8, 0.0, 0.0, 1.0);
    ks = DoubleColor(1.0, 1.0, 1.0, 1.0);
    mapKa = -1;
    mapKd = -1;
    mapKs = -1;
    mapD = "";
    emmColor = DoubleColor(0.0, 0.0, 0.0, 1.0);
    shiny = 90.0;
    reflectivity = DoubleColor(0.3, 0.0, 0.0, 1.0);
    refractivity = DoubleColor(0.3, 0.0, 0.0, 1.0);
    refractiveIndex = 1.5;
    transmissionFilter = DoubleColor(1.0, 1.0, 1.0, 1.0);
    lineColor = DoubleColor(1.0, 1.0, 1.0, 1.0);
    doubleSided = false;
}

QString PMesh::MaterialCell::toString()
{
    return materialName;
}
