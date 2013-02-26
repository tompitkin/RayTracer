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
    surfHead = nullptr;
    vertArray = nullptr;
    texVertArray = nullptr;
    vertNormArray = nullptr;
    materials = nullptr;
    active = true;

    center = new Double3D();
    firstDraw = true;

    modelMatUniform = new Matrix4Uniform(nullptr, "modelMat");

    useAmbTexUniform = new IntUniform(0, "materialSettings.ambTex");
    useDiffTexUniform = new IntUniform(0, "materialSettings.diffTex");
    useSpecTexUniform = new IntUniform(0, "materialSettings.specTex");
}

PMesh::~PMesh()
{
}

Sphere *PMesh::calcBoundingSphere()
{
    double greatest = 0.0;
    double dist;
    for (int i = 0; i < numVerts; i++)
    {
        dist = vertArray->at(i)->worldPos->distanceTo(center);
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
    SurfCell *curSurf = surfHead;
    PolyCell *curPoly;
    VertListCell *curVert, *temp;

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
                        vector1 = new Double3D(vertArray->at(curVert->next->vert)->worldPos);
                        vector1 = new Double3D(vector1->minus(vertArray->at(curVert->vert)->worldPos));
                        vector2 = new Double3D(vertArray->at(curVert->next->next->vert)->worldPos);
                        vector2 = new Double3D(vector2->minus(vertArray->at(curVert->vert)->worldPos));
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
    Double3D *norm;
    PolyListCell *curPolyLC;
    vertNormArray = new vector<Double3D*>();
    vertNormArray->resize(this->numVerts);
    this->numNorms = this->numVerts;
    SurfCell *curSurf = surfHead;
    while (curSurf != nullptr)
    {
        PolyCell *curPoly = curSurf->polyHead;
        while (curPoly != nullptr)
        {
            VertListCell *curVertLC = curPoly->vert;
            while (curVertLC != nullptr)
            {
                curPolyLC = vertArray->at(curVertLC->vert)->polys;
                if (curPolyLC != nullptr)
                {
                    norm = new Double3D();
                    while (curPolyLC != nullptr)
                    {
                        if (curPolyLC->poly != nullptr)
                            norm = norm->plus(curPolyLC->poly->normal);
                        curPolyLC = curPolyLC->next;
                    }

                    vertNormArray->at(curVertLC->vert) = new Double3D(norm->getUnit());
                    curVertLC->norm = curVertLC->vert;
                }
                else
                {
                    vertNormArray->at(curVertLC->vert) = new Double3D(curPoly->normal);
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
    worldPos = new Double3D();
    viewPos = new Double3D();
    screenPos = new Double3D();
    polys = nullptr;
}

PMesh::VertCell::VertCell(PMesh::VertCell *from)
{
    worldPos = new Double3D(from->worldPos);
}


PMesh::PolyCell::PolyCell()
{
    numVerts = 0;
    vert = nullptr;
    normal = new Double3D();
    culled = false;
    parentSurf = nullptr;
    next = nullptr;
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


PMesh::PolyListCell::PolyListCell()
{
    poly = nullptr;
    next = nullptr;
}


PMesh::SurfCell::SurfCell(QString name)
{
    this->name = name;
    numPoly = 0;
    polyHead = nullptr;
    material = 0;
    next = nullptr;
    numVerts = 0;
}


PMesh::MaterialCell::MaterialCell()
{
    materialName = "default";
    ka = new DoubleColor(0.2, 0.2, 0.2, 1.0);
    kd = new DoubleColor(0.8, 0.0, 0.0, 1.0);
    ks = new DoubleColor(1.0, 1.0, 1.0, 1.0);
    mapKa = -1;
    mapKd = -1;
    mapKs = -1;
    mapD = "";
    emmColor = new DoubleColor(0.0, 0.0, 0.0, 1.0);
    shiny = 90.0;
    reflectivity = new DoubleColor(0.3, 0.0, 0.0, 1.0);
    refractivity = new DoubleColor(0.3, 0.0, 0.0, 1.0);
    refractiveIndex = 1.5;
    transmissionFilter = new DoubleColor(1.0, 1.0, 1.0, 1.0);
    lineColor = new DoubleColor(1.0, 1.0, 1.0, 1.0);
    doubleSided = false;
}

QString PMesh::MaterialCell::toString()
{
    return materialName;
}
