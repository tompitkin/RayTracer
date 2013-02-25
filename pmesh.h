#ifndef PMESH_H
#define PMESH_H

#include <QString>
#include <QFile>
#include <string>
#include "GL/glew.h"
#include "MatrixManipulation/double3d.h"
#include "Utilities/doublecolor.h"
#include "sphere.h"
#include "shaderprogram.h"

using namespace std;

class Scene;

class PMesh
{
public:
    class PolyListCell;
    class SurfCell;

    class VertCell
    {
    public:
        VertCell();
        VertCell(VertCell *from);

        PolyListCell *polys;
        Double3D *worldPos;
        Double3D *viewPos;
        Double3D *screenPos;
    };

    class VertListCell
    {
    public:
        VertListCell();

        VertListCell *next;
        int vert;
        int norm;
        int tex;
        int tan;
        int bitan;
    };

    class PolyCell
    {
    public:
        PolyCell();

        VertListCell *vert;
        SurfCell *parentSurf;
        PolyCell *next;
        Double3D *normal;
        Double3D *tangent;
        Double3D *biTangent;
        Double3D *viewNorm;
        bool culled;
        int numVerts;
    };

    class PolyListCell
    {
    public:
        PolyListCell();

        PolyCell *poly;
        PolyListCell *next;
    };

    class SurfCell
    {
    public:
        SurfCell(QString name);

        PolyCell *polyHead;
        SurfCell *next;
        QString name;
        GLfloat *vertexBuffer = nullptr;
        GLfloat *normalBuffer = nullptr;
        GLfloat *texCoordBuffer = nullptr;
        GLfloat *tangentBuffer = nullptr;
        GLfloat *bitangentBuffer = nullptr;
        int buffers[5];
        int vaoID;
        int numPoly;
        int material;
        int numVerts;
    };

    class MaterialCell
    {
    public:
        MaterialCell();

        QString toString();

        QString materialName;
        QString mapD;
        DoubleColor *ka;
        DoubleColor *kd;
        DoubleColor *ks;
        DoubleColor *emmColor;
        DoubleColor *reflectivity;
        DoubleColor *refractivity;
        DoubleColor *transmissionFilter;
        DoubleColor *lineColor;
        bool doubleSided;
        double refractiveIndex;
        double shiny;
        int mapKa;
        int mapKd;
        int mapKs;
    };

    PMesh(Scene *aScene);

    virtual bool load(QString filePath) = 0;
    Sphere *calcBoundingSphere();
    void calcPolyNorms();
    void calcVertNorms();
    void updateUniforms();

    Scene *theScene;
    QFile *file;
    QString objName;
    vector<VertCell*> *vertArray;
    vector<SurfCell*> *vertUsedArray;
    vector<Double3D*> *texVertArray;
    vector<Double3D*> *vertNormArray;
    SurfCell *surfHead;
    MaterialCell *materials;
    Double3D *center;
    Sphere *boundingSphere;
    ShaderProgram *theShader;
    double *modelMat;
    bool active;
    int objNumber;
    int fileType;
    int numVerts;
    int numNorms;
    int numTex;
    int numSurf;
    int numPolys;
    int numMats;
};

#endif // PMESH_H
