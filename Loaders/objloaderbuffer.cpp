#include "objloaderbuffer.h"
#include "scene.h"

ObjLoaderBuffer::ObjLoaderBuffer(Scene *aScene):PMesh(aScene)
{
}

bool ObjLoaderBuffer::load(QString filePath)
{
    fileType = ObjectTypes::TYPE_OBJ;
    file = new QFile(filePath);
    if (!file->open(QIODevice::ReadOnly | QIODevice::Text))
            return false;
    QFileInfo pathInfo(filePath);
    objName = pathInfo.fileName();
    readVerts();
    readTexVerts();

    if (readMaterials() == -1)
    {
        fprintf(stdout, "Not all materials could be loaded for: %s\n", qPrintable(filePath));
        fprintf(stdout, "Surfaces with no materials will be assigned a default material\n");
    }

    readSurfaces();
    countPolyVerts();
    calcPolyNorms();
    calcVertNorms();
    boundingSphere = calcBoundingSphere();
    active = true;
    modelMat = MatrixOps::newIdentity();
    loadBuffers();

    file->close();
    return true;
}

void ObjLoaderBuffer::readVerts()
{
    int vertNo = 0;
    double xSum = 0.0, ySum = 0.0, zSum = 0.0;
    numVerts = countVerts();
    if (numVerts > 0)
    {
        vertArray = new vector<VertCell*>();
        vertUsedArray = new vector<SurfCell*>();
        for (int i = 0; i < numVerts; i++)
        {
            vertArray->push_back(new VertCell());
            vertUsedArray->push_back(nullptr);
        }
    }

    file->seek(0);
    QTextStream in(file);
    while (!in.atEnd())
    {
        QString line = in.readLine();
        if (line.length() > 0)
        {
            if (line[0] == 'v' && line[1] == ' ')
            {
                QStringList list = line.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
                list.removeAt(0);
                vertArray->at(vertNo)->worldPos->x = list[0].toDouble();
                xSum += vertArray->at(vertNo)->worldPos->x;
                vertArray->at(vertNo)->worldPos->y = list[1].toDouble();
                ySum += vertArray->at(vertNo)->worldPos->y;
                vertArray->at(vertNo)->worldPos->z = list[2].toDouble();
                zSum += vertArray->at(vertNo)->worldPos->z;
                vertNo++;
            }
        }
    }
    center->x = xSum/(double)numVerts;
    center->y = ySum/(double)numVerts;
    center->z = zSum/(double)numVerts;
}

void ObjLoaderBuffer::readTexVerts()
{
    int texNo = 0;
    numTex = countTexVerts();
    if (numTex > 0)
    {
        texVertArray = new vector<Double3D*>();
        for (int i = 0; i < numTex; i++)
            texVertArray->push_back(new Double3D());
    }

    file->seek(0);
    QTextStream in(file);
    while (!in.atEnd())
    {
        QString line = in.readLine();
        if (line.length() > 0)
        {
            if (line[0] == 'v' && line[1] == 't')
            {
                QStringList list = line.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
                list.removeAt(0);
                texVertArray->at(texNo)->x = list[0].toDouble();
                texVertArray->at(texNo)->y = list[1].toDouble();
                if (texVertArray->size() > 2)
                    texVertArray->at(texNo)->z = list[2].toDouble();
                texNo++;
            }
        }
    }
}

void ObjLoaderBuffer::readSurfaces()
{
    int curMat = 0;
    SurfCell *curSurf;
    QString matName;
    bool inSmooth = false;
    file->seek(0);
    QTextStream in(file);
    curSurf = surfHead;
    while (!in.atEnd())
    {
        QString line = in.readLine();
        if (line.length() > 0)
        {
            QStringList tokens = line.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
            switch(line.at(0).toLatin1())
            {
            case '#':
            case '!':
            case '$':
            case '\n':
            case 'v':
                break;
            case 'u':
            {
                matName = tokens[1];
                bool found = false;
                int i = 0;
                while (!found && i < numMats)
                {
                    if (materials[i].materialName.toUpper().compare(matName.toUpper()) == 0)
                    {
                        curMat = i;
                        found = true;
                    }
                    i++;
                }
                if (!found)
                {
                    fprintf(stdout, "Group %s material %s not found - using default\n", qPrintable(curSurf->name), qPrintable(matName));
                    curSurf->material = 0;
                }
                else
                    curSurf->material = curMat;
                break;
            }
            case 's':
                if (tokens.size() > 1)
                {
                    if (tokens[1].toUpper().compare("OFF") == 0)
                        inSmooth = false;
                    else
                        inSmooth = true;
                }
                break;
            case 'f':
                if (curSurf == nullptr)
                {
                    fprintf(stdout, "ReadSurface: No active surface available\n");
                    fprintf(stdout, "Creating a default surface\n");
                    surfHead = new SurfCell("default");
                    curSurf = surfHead;
                }
                addPolyToSurf(curSurf, line, inSmooth);
                curSurf->numPoly++;
                numPolys++;
                break;
            case 'g':
            {
                QString name;
                if (tokens.size() == 1)
                    name = ("Group"+numSurf);
                else
                    name = tokens[1];
                if (surfHead == nullptr)
                {
                    surfHead = new SurfCell(name);
                    curSurf = surfHead;
                }
                else
                {
                    curSurf->next = new SurfCell(name);
                    curSurf = curSurf->next;
                }
                numSurf++;
                break;
            }
            default:
                break;
            }
        }
    }
}

void ObjLoaderBuffer::addPolyToSurf(SurfCell *curSurf, QString line, bool inSmooth)
{
    int curIndex = 0;
    PolyCell *curPoly;
    PolyListCell *curVertPoly;
    VertListCell *curVert;
    QStringList tokens = line.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
    if (curSurf->polyHead == nullptr)
    {
        curSurf->polyHead = new PolyCell();
        curPoly = curSurf->polyHead;
    }
    else
    {
        curPoly = curSurf->polyHead;
        while (curPoly->next != nullptr)
            curPoly = curPoly->next;
        curPoly->next = new PolyCell();
        curPoly = curPoly->next;
    }
    curPoly->numVerts = 0;

    curPoly->parentSurf = curSurf;
    for (int i = 1; i < tokens.size(); i++)
    {
        QStringList vertTokens = tokens[i].split("/", QString::SkipEmptyParts);
        curIndex = vertTokens[0].toInt();
        if (curPoly->vert == nullptr)
        {
            curPoly->vert = new VertListCell();
            curVert = curPoly->vert;
        }
        else
        {
            curVert = curPoly->vert;
            while (curVert->next != nullptr)
                curVert = curVert->next;
            curVert->next = new VertListCell();
            curVert = curVert->next;
        }
        if (!inSmooth)
        {
            if (curIndex <= (int)vertUsedArray->size() && vertUsedArray->at(curIndex - 1)!= nullptr)
            {
                vertArray->insert(vertArray->begin()+numVerts, new VertCell(vertArray->at(curIndex-1)));
                if (texVertArray != nullptr)
                    texVertArray->insert(texVertArray->begin()+numVerts, new Double3D(texVertArray->at(curIndex-1)));
                vertUsedArray->insert(vertUsedArray->begin()+numVerts++, curSurf);
                curIndex = numVerts;
            }
        }
        else if(vertUsedArray->at(curIndex - 1) != nullptr && vertUsedArray->at(curIndex-1)!=curSurf)
        {
            int copyIndex = findCopyInSurf(curSurf, vertArray->at(curIndex-1)->worldPos);
            if (copyIndex == -1)
            {
                vertArray->insert(vertArray->begin()+numVerts, new VertCell(vertArray->at(curIndex-1)));
                if (texVertArray != nullptr)
                    texVertArray->insert(texVertArray->begin()+numVerts, new Double3D(texVertArray->at(curIndex-1)));
                vertUsedArray->insert(vertUsedArray->begin()+numVerts++, curSurf);
                curIndex = numVerts;
            }
            else
                curIndex = copyIndex+1;
        }

        curVert->vert = curIndex-1;
        curVert->tex = curIndex-1;
        vertUsedArray->at(curIndex-1) = curSurf;

        if (inSmooth)
        {
            if (vertArray->at(curIndex-1)->polys == nullptr)
            {
                vertArray->at(curIndex-1)->polys = new PolyListCell();
                curVertPoly = vertArray->at(curIndex-1)->polys;
            }
            else
            {
                curVertPoly = vertArray->at(curIndex-1)->polys;
                while (curVertPoly->next != nullptr)
                    curVertPoly = curVertPoly->next;
                curVertPoly->next = new PolyListCell();
                curVertPoly = curVertPoly->next;
            }
            curVertPoly->poly = curPoly;
        }
        else
            vertArray->at(curIndex-1)->polys = nullptr;

        curIndex = tokens[i].indexOf('/');
        if (curIndex > -1)
        {
            if (tokens[i].at(curIndex+1).toLatin1() != '/')
            {
                curIndex = vertTokens[1].toInt();
                curVert->tex = curIndex-1;
            }
        }
    }
}

void ObjLoaderBuffer::countPolyVerts()
{
    SurfCell *curSurf;
    PolyCell *curPoly;
    VertListCell *curVert;
    curSurf = surfHead;
    while (curSurf != nullptr)
    {
        curPoly = curSurf->polyHead;
        while (curPoly != nullptr)
        {
            curVert = curPoly->vert;
            while (curVert != nullptr)
            {
                curPoly->numVerts++;
                curVert = curVert->next;
            }
            curPoly = curPoly->next;
        }
        curSurf = curSurf->next;
    }
}

void ObjLoaderBuffer::loadBuffers()
{
    SurfCell *curSurf;
    PolyCell *curPoly;
    VertListCell *curVertLC;

    curSurf = surfHead;
    int vertCount;
    while (curSurf != nullptr)
    {
        int vertsPerPrim;
        vertCount = 0;
        curPoly = curSurf->polyHead;
        vertsPerPrim = curPoly->numVerts;
        while (curPoly != nullptr)
        {
            vertCount += curPoly->numVerts;
            if (curPoly->numVerts != vertsPerPrim)
            {
                fprintf(stderr, "Surface %s: Unequal number of vertices\n", qPrintable(curSurf->name));
                fprintf(stderr, "   First prim had %d Cur Prim has %d\n", curPoly->numVerts, vertsPerPrim);
                return;
            }
            curPoly = curPoly->next;
        }
        curSurf->numVerts = vertCount;
        GLfloat *vertices = new GLfloat[vertCount*3];
        int vInd = 0;
        GLfloat *normals = new GLfloat[vertCount *3];
        int nInd = 0;
        GLfloat *texCoords = new GLfloat[vertCount *2];
        int tind = 0;
        curPoly = curSurf->polyHead;
        while (curPoly != nullptr)
        {
            curVertLC = curPoly->vert;
            while (curVertLC != nullptr)
            {
                VertCell *curVert = vertArray->at(curVertLC->vert);
                vertices[vInd++] = curVert->worldPos->x;
                vertices[vInd++] = curVert->worldPos->y;
                vertices[vInd++] = curVert->worldPos->z;
                if (texVertArray != nullptr)
                {
                    Double3D *curTexCoord = texVertArray->at(curVertLC->tex);
                    texCoords[tind++] = curTexCoord->x;
                    texCoords[tind++] = curTexCoord->y;
                }
                normals[nInd++] = vertNormArray->at(curVertLC->vert)->x;
                normals[nInd++] = vertNormArray->at(curVertLC->vert)->y;
                normals[nInd++] = vertNormArray->at(curVertLC->vert)->z;
                curVertLC = curVertLC->next;
            }
            curPoly = curPoly->next;
        }
        curSurf->vertexBuffer = vertices;
        curSurf->normalBuffer = normals;
        curSurf->texCoordBuffer = texCoords;
        curSurf = curSurf->next;
    }
    theShader = theScene->shaderProg;
}

int ObjLoaderBuffer::countVerts()
{
    int vertCount = 0;
    file->seek(0);
    QTextStream in(file);
    while (!in.atEnd())
    {
        QString line = in.readLine();
        if (line.length() > 0)
            if (line[0] == 'v' && line[1] == ' ')
                vertCount++;
    }
    return vertCount;
}

int ObjLoaderBuffer::countTexVerts()
{
    int texVertCount = 0;
    file->seek(0);
    QTextStream in(file);
    while (!in.atEnd())
    {
        QString line = in.readLine();
        if (line.length() > 0)
            if (line[0] == 'v' && line[1] == 't')
                texVertCount++;
    }
    return texVertCount;
}

int ObjLoaderBuffer::readMaterials()
{
    int matNo = -1;
    QString fileNameList;
    file->seek(0);
    QTextStream in(file);
    while (!in.atEnd())
    {
        QString line = in.readLine();
        if (line.length() > 6)
        {
            QStringList list = line.split(" ", QString::SkipEmptyParts);
            if (list.at(0) == "mtllib")
            {
                mtlFileName = list.at(1);
                fileNameList.append(list.at(1));
                for (int i = 2; i < list.length(); i++)
                    fileNameList.append(" "+list[i]);
            }
        }
    }
    QStringList fileNames = fileNameList.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);

    fprintf(stdout, "The following material libraries were found:\n");
    for (int i = 0; i < fileNames.size(); i++)
        fprintf(stdout, "   * %s\n", qPrintable(fileNames[i]));

    numMats += countMaterials(fileNameList);
    materials = new MaterialCell[numMats]();
    for (int i = 0; i < fileNames.size(); i++)
    {
        QFile mtlFile(file->fileName().mid(0, (file->fileName().length()-objName.length()))+fileNames[i]);
        if (!mtlFile.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            fprintf(stderr, "Error opening material library for reading\n");
            return -1;
        }
        QTextStream in(&mtlFile);
        while (!in.atEnd())
        {
            QString line = in.readLine();
            if (line.length() > 0)
            {
                QStringList tokens = line.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
                switch(line.at(0).toLatin1())
                {
                case '#':
                case '!':
                case '$':
                case '\n':
                    break;
                case 'n':
                    matNo++;
                    materials[matNo].materialName = tokens[1];
                    break;
                case 'K':
                    switch(line.at(1).toLatin1())
                    {
                    case 'a':
                        materials[matNo].ka->r = tokens[1].toDouble();
                        materials[matNo].ka->g = tokens[2].toDouble();
                        materials[matNo].ka->b = tokens[3].toDouble();
                        if (tokens.size() > 4)
                            materials[matNo].ka->a = tokens[4].toDouble();
                        break;
                    case 'd':
                        materials[matNo].kd->r = tokens[1].toDouble();
                        materials[matNo].kd->g = tokens[2].toDouble();
                        materials[matNo].kd->b = tokens[3].toDouble();
                        if (tokens.size() > 4)
                            materials[matNo].kd->a = tokens[4].toDouble();
                        break;
                    case 's':
                        materials[matNo].ks->r = tokens[1].toDouble();
                        materials[matNo].ks->g = tokens[2].toDouble();
                        materials[matNo].ks->b = tokens[3].toDouble();
                        if (tokens.size() > 4)
                            materials[matNo].ks->a = tokens[4].toDouble();
                        break;
                    default:
                        break;
                    }
                    break;
                case 'I':
                    switch(line.at(1).toLatin1())
                    {
                    case 'r':
                    {
                        int count = 0;
                        do
                        {
                            count++;
                            materials[matNo].reflectivity->r = tokens[count].toDouble();
                        }
                        while(count < tokens.size() && count < 3);
                        if (count == 1)
                        {
                            materials[matNo].reflectivity->g = materials[matNo].reflectivity->r;
                            materials[matNo].reflectivity->b = materials[matNo].reflectivity->r;
                        }
                        else
                            fprintf(stderr, "Error reading reflectivity: count=%d\n", count);
                        break;
                    }
                    case 't':
                    {
                        int cnt = 0;
                        do
                        {
                            cnt++;
                            materials[matNo].refractivity->r = tokens[cnt].toDouble();
                        }
                        while(cnt < tokens.size() && cnt < 3);
                        if (cnt == 1)
                        {
                            materials[matNo].refractivity->g = materials[matNo].refractivity->r;
                            materials[matNo].refractivity->b = materials[matNo].refractivity->r;
                        }
                        else
                            fprintf(stderr, "Error reading refractivity: count=%d\n", cnt);
                        break;
                    }
                    }
                    break;
                case 'm':
                    /*
                     *      #########################################
                     *      ##############TEXTURE STUFF##############
                     *      #########################################
                     */
                case 'e':
                    materials[matNo].emmColor->r = tokens[1].toDouble();
                    materials[matNo].emmColor->g = tokens[2].toDouble();
                    materials[matNo].emmColor->b = tokens[3].toDouble();
                    break;
                case 'T':
                    materials[matNo].transmissionFilter->r = tokens[1].toDouble();
                    materials[matNo].transmissionFilter->g = tokens[2].toDouble();
                    materials[matNo].transmissionFilter->b = tokens[3].toDouble();
                    break;
                case 'N':
                    switch (line.at(1).toLatin1())
                    {
                    case 's':
                        materials[matNo].shiny = tokens[1].toDouble();
                        break;
                    case 'i':
                        materials[matNo].refractiveIndex = tokens[1].toDouble();
                        break;
                    default:
                        break;
                    }
                    break;
                case 'L':
                    materials[matNo].lineColor->r = tokens[1].toDouble();
                    materials[matNo].lineColor->g = tokens[2].toDouble();
                    materials[matNo].lineColor->b = tokens[3].toDouble();
                    if (tokens.size() > 4)
                        materials[matNo].lineColor->a = tokens[4].toDouble();
                    break;
                case 'd':
                    if (line.at(1) == 's')
                        if (tokens[1].toInt() == 1)
                            materials[matNo].doubleSided = true;
                    break;
                default:
                    break;
                }
            }
        }
        mtlFile.close();
    }
    return 0;
}

int ObjLoaderBuffer::countMaterials(QString fileNameList)
{
    int matCount = 0;
    QStringList fileNames = fileNameList.split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
    for (int i = 0; i < fileNames.size(); i++)
    {
        QFile mtlFile(file->fileName().mid(0, (file->fileName().length()-objName.length()))+fileNames[i]);
        if (!mtlFile.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            fprintf(stderr, "Error opening material library for counting\n");
            return 0;
        }
        QTextStream in(&mtlFile);
        while (!in.atEnd())
        {
            QString line = in.readLine();
            if (line.length() > 6)
                if (line.mid(0, 6).toLower().compare("newmtl") == 0)
                    matCount++;
        }
        mtlFile.close();
    }
    fprintf(stdout, "%d materials found\n", matCount);
    return matCount;
}

int ObjLoaderBuffer::findCopyInSurf(PMesh::SurfCell *curSurf, Double3D *findme)
{
    int i = 0;
    while (i < numVerts && (vertUsedArray->at(i) != curSurf || vertUsedArray->at(i) == nullptr || vertArray->at(i)->worldPos != findme))
        i++;
    if (i == numVerts)
        return -1;
    else return i;
}
