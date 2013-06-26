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
    bufferIDs = nullptr;
    VAOIDs = nullptr;

    firstDraw = true;

    normalMatUniform = new Matrix3Uniform(nullptr, "normalMat");
    modelMatUniform = new Matrix4Uniform(nullptr, "modelMat");

    ambUniform = new Vec4Uniform(nullptr, "materialSettings.ambient");
    diffUniform = new Vec4Uniform(nullptr, "materialSettings.diffuse");
    specUniform = new Vec4Uniform(nullptr, "materialSettings.specular");
    shinyUniform = new FloatUniform(0.0f, "materialSettings.shininess");

    useAmbTexUniform = new IntUniform(0, "materialSettings.ambTex");
    useDiffTexUniform = new IntUniform(0, "materialSettings.diffTex");
    useSpecTexUniform = new IntUniform(0, "materialSettings.specTex");
}

PMesh::~PMesh()
{
    for (int i = 0; i < (int)vertUsedArray.size(); i++)
        vertUsedArray[i].reset();
    for (int i = 0; i < (int)vertArray.size(); i++)
        vertArray[i].reset();
    for (int i = 0; i < (int)texVertArray.size(); i++)
        texVertArray[i].reset();
    for (int i = 0; i < (int)vertNormArray.size(); i++)
        vertNormArray[i].reset();
    theScene = nullptr;
    theShader.reset();
    //surfHead->polyHead.reset();
    surfHead.reset();
    delete boundingSphere;
    delete normalMatUniform;
    delete modelMatUniform;
    delete useAmbTexUniform;
    delete useDiffTexUniform;
    delete useSpecTexUniform;
    delete ambUniform;
    delete diffUniform;
    delete specUniform;
    delete shinyUniform;
    delete file;
    if (bufferIDs != nullptr)
        delete []bufferIDs;
    if (VAOIDs != nullptr)
        delete []VAOIDs;
    delete []materials;
}

Sphere *PMesh::calcBoundingSphere()
{
    double greatest = 0.0;
    double dist;
    for (int i = 0; i < numVerts; i++)
    {
        dist = (vertArray.at(i))->worldPos.distanceTo(center);
        if (dist > greatest)
            greatest = dist;
    }
    //fprintf(stdout, "Sphere: radius = %f\n",greatest);
    Sphere *retVal = new Sphere(center, greatest, this);
    return retVal;
}

void PMesh::calcPolyNorms()
{
    Double3D vector1, vector2, cross;
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
                        vector1 = Double3D(vertArray.at(curVert->next->vert)->worldPos);
                        vector1 = Double3D(vector1.minus(vertArray.at(curVert->vert)->worldPos));
                        vector2 = Double3D(vertArray.at(curVert->next->next->vert)->worldPos);
                        vector2 = Double3D(vector2.minus(vertArray.at(curVert->vert)->worldPos));
                        cross = Double3D(vector1.cross(vector2));
                        curPoly->normal = Double3D(cross.getUnit());
                    }
                }
            }
            else
                curPoly->normal = new Double3D();
            curPoly = curPoly->next;
        }
        curSurf = curSurf->next;
    }
    //fprintf(stdout, "Polygon Normals calculated\n");
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
                    norm = Double3D();
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
    //fprintf(stdout, "Vertex Normals calculated\n");
}

void PMesh::calcViewPolyNorms()
{
    PMesh::SurfCell *curSurf;
    PMesh::PolyCell *curPoly;
    PMesh::VertListCell *curVert;

    Double3D p1;
    Double3D p2;
    Double3D p3;
    Double3D v1;
    Double3D v2;
    Double3D norm;

    curSurf = surfHead.get();

    while (curSurf != nullptr)
    {
        curPoly = curSurf->polyHead.get();
        while (curPoly != nullptr)
        {
            curVert = curPoly->vert.get();

            p1 = vertArray.at(curVert->vert)->viewPos;
            curVert = curVert->next.get();
            p2 = vertArray.at(curVert->vert)->viewPos;
            curVert = curVert->next.get();
            p3 = vertArray.at(curVert->vert)->viewPos;

            v1 = p2.minus(p1);
            v2 = p3.minus(p2);

            norm = v1.cross(v2);
            norm.unitize();

            curPoly->viewNorm = norm;
            curPoly = curPoly->next.get();
        }
        curSurf = curSurf->next.get();
    }
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
        shared_ptr<Uniform> thisOne = theShader->uniformList.at(i);
        if (thisOne->needsUpdate)
            thisOne->update(theShader->progID);
        thisOne->needsUpdate=false;
    }
}

void PMesh::draw(Camera *camera)
{
    shared_ptr<SurfCell> curSurf;
    //fprintf(stdout, "Drawing %s\n", qPrintable(this->objName));

    if (firstDraw)
    {
        if (bufferIDs != nullptr)
        {
            delete []bufferIDs;
            bufferIDs = nullptr;
        }
        if (VAOIDs != nullptr)
        {
            delete []VAOIDs;
            VAOIDs = nullptr;
        }
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
        copy(modelMat.begin(), modelMat.begin()+16, modelMatUniform->theBuffer);
        GL::checkGLErrors(" updateMatrix4Uniform(before): ");
        modelMatUniform->update(theShader->progID);
        vector<double> modelViewMat = MatrixOps::newIdentity();
        modelViewMat = MatrixOps::multMat(modelViewMat, camera->viewMat);
        modelViewMat = MatrixOps::multMat(modelViewMat, modelMat);
        //MatrixOps::inverseTranspose(modelViewMat);
        if (normalMatUniform->theBuffer == nullptr)
            normalMatUniform->theBuffer = new GLfloat[9]();
        normalMatUniform->theBuffer[0] = (float)modelViewMat[0];
        normalMatUniform->theBuffer[1] = (float)modelViewMat[1];
        normalMatUniform->theBuffer[2] = (float)modelViewMat[2];
        normalMatUniform->theBuffer[3] = (float)modelViewMat[4];
        normalMatUniform->theBuffer[4] = (float)modelViewMat[5];
        normalMatUniform->theBuffer[5] = (float)modelViewMat[6];
        normalMatUniform->theBuffer[6] = (float)modelViewMat[8];
        normalMatUniform->theBuffer[7] = (float)modelViewMat[9];
        normalMatUniform->theBuffer[8] = (float)modelViewMat[10];
        normalMatUniform->update(theScene->shaderProg.get()->progID);

        int surfCount = 0;
        curSurf = surfHead;
        while (curSurf != nullptr)
        {
            if (firstDraw)
            {
                /*
                 *      #########################################
                 *      ##############TEXTURE STUFF##############
                 *      #########################################
                 */

                curSurf->buffers[0] = bufferIDs[surfCount*3];
                curSurf->buffers[1] = bufferIDs[surfCount*3+1];
                if (curSurf->texCoordBuffer != nullptr)
                    curSurf->buffers[2] = bufferIDs[surfCount*3+2];
                curSurf->vaoID = VAOIDs[surfCount];

                glBindVertexArray(curSurf->vaoID);
                glBindBuffer(GL_ARRAY_BUFFER, curSurf->buffers[0]);
                glBufferData(GL_ARRAY_BUFFER, curSurf->numVerts*3*(sizeof(GLfloat)), curSurf->vertexBuffer, GL_STATIC_DRAW);
                glEnableVertexAttribArray(0);
                GL::checkGLErrors("Enabled Vertex VBO");
                glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0L);

                glBindBuffer(GL_ARRAY_BUFFER, curSurf->buffers[1]);
                glBufferData(GL_ARRAY_BUFFER, curSurf->numVerts*3*(sizeof(GLfloat)), curSurf->normalBuffer, GL_STATIC_DRAW);
                glEnableVertexAttribArray(1);
                GL::checkGLErrors("Enabled Normal VBO");
                glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, 0L);
            }

            if (curSurf->material != previousMat)
            {
                if (ambUniform->theBuffer != nullptr)
                    delete []ambUniform->theBuffer;
                ambUniform->theBuffer = materials[curSurf->material].ka.toFloatv();
                ambUniform->update(theScene->shaderProg.get()->progID);

                if (diffUniform->theBuffer != nullptr)
                    delete []diffUniform->theBuffer;
                diffUniform->theBuffer = materials[curSurf->material].kd.toFloatv();
                diffUniform->update(theScene->shaderProg.get()->progID);

                if (specUniform->theBuffer != nullptr)
                    delete []specUniform->theBuffer;
                specUniform->theBuffer = materials[curSurf->material].ks.toFloatv();
                specUniform->update(theScene->shaderProg.get()->progID);

                shinyUniform->theFloat = materials[curSurf->material].shiny;
                shinyUniform->update(theScene->shaderProg.get()->progID);
            }

            previousMat = curSurf->material;

            glBindVertexArray(curSurf->vaoID);
            GL::checkGLErrors("Bound VAO");

            glDrawArrays(GL_TRIANGLES, 0, 3*curSurf->numPoly);
            GL::checkGLErrors(("After Draw"));

            curSurf = curSurf->next;
            surfCount++;
        }
        firstDraw = false;
    }
}

void PMesh::translate(double x, double y, double z)
{
    vector<double> trans = MatrixOps::makeTranslation(x, y, z);
    modelMat = MatrixOps::multMat(trans, modelMat);
    center = center.preMultiplyMatrix(trans);
}

PMesh::VertCell::VertCell()
{
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
    culled = false;
}

PMesh::PolyCell::~PolyCell()
{
    vert.reset();
    parentSurf.reset();
    next.reset();
}


PMesh::VertListCell::VertListCell()
{
    vert = -1;
    norm = -1;
    tex = -1;
    tan = -1;
    bitan = -1;
}

PMesh::VertListCell::~VertListCell()
{
    next.reset();
}


PMesh::PolyListCell::PolyListCell()
{
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
    material = 0;
    numVerts = 0;
}

PMesh::SurfCell::~SurfCell()
{
    next.reset();
    polyHead.reset();
    delete []vertexBuffer;
    delete []normalBuffer;
    delete []texCoordBuffer;
    delete []tangentBuffer;
    delete []bitangentBuffer;
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
