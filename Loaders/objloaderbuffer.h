#ifndef OBJLOADERBUFFER_H
#define OBJLOADERBUFFER_H

#include <QFileInfo>
#include <QTextStream>
#include <QStringList>
#include <QRegularExpression>
#include "pmesh.h"
#include "Loaders/objecttypes.h"
#include "MatrixManipulation/matrixops.h"

class Scene;

class ObjLoaderBuffer : public PMesh
{
public:
    ObjLoaderBuffer(Scene *aScene);

    bool load(QString filePath);
    int countVerts();
    int countTexVerts();
    int readMaterials();
    int countMaterials(QString fileNameList);
    int findCopyInSurf(SurfCell *curSurf, Double3D *findme);
    void readVerts();
    void readTexVerts();
    void readSurfaces();
    void addPolyToSurf(SurfCell *curSurf, QString line, bool inSmooth);
    void countPolyVerts();
    void loadBuffers();

    QString mtlFileName = "";
};

#endif // OBJLOADERBUFFER_H
