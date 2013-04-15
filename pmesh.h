#ifndef PMESH_H
#define PMESH_H

#include <QString>
#include <QFile>
#include <string>
#include "GL/glew.h"
#include "MatrixManipulation/double3d.h"
#include "Utilities/doublecolor.h"
#include "Utilities/gl.h"
#include "sphere.h"
#include "shaderprogram.h"
#include "camera.h"
#include "Uniforms/intuniform.h"
#include "Uniforms/matrix4uniform.h"
#include "Uniforms/matrix3uniform.h"
#include "Uniforms/vec4uniform.h"
#include "Uniforms/floatuniform.h"

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
        VertCell(const VertCell *from);
        virtual ~VertCell();

        shared_ptr<PolyListCell> polys;
        Double3D worldPos;
        Double3D viewPos;
        Double3D screenPos;
    };

    class VertListCell
    {
    public:
        VertListCell();
        virtual ~VertListCell();

        shared_ptr<VertListCell> next;
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
        virtual ~PolyCell();

        shared_ptr<VertListCell> vert;
        weak_ptr<SurfCell> parentSurf;
        shared_ptr<PolyCell> next;
        Double3D normal;
        Double3D tangent;
        Double3D biTangent;
        Double3D viewNorm;
        bool culled;
        int numVerts;
    };

    class PolyListCell
    {
    public:
        PolyListCell();
        virtual ~PolyListCell();

        shared_ptr<PolyCell> poly;
        shared_ptr<PolyListCell> next;
    };

    class SurfCell
    {
    public:
        SurfCell(QString name);
        virtual ~SurfCell();

        shared_ptr<PolyCell> polyHead;
        shared_ptr<SurfCell> next;
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
        DoubleColor ka;
        DoubleColor kd;
        DoubleColor ks;
        DoubleColor emmColor;
        DoubleColor reflectivity;
        DoubleColor refractivity;
        DoubleColor transmissionFilter;
        DoubleColor lineColor;
        bool doubleSided;
        double refractiveIndex;
        double shiny;
        int mapKa;
        int mapKd;
        int mapKs;
    };

    PMesh(Scene *aScene);
    virtual ~PMesh();

    virtual bool load(QString filePath) = 0;
    Sphere *calcBoundingSphere();
    void calcPolyNorms();
    void calcVertNorms();
    void calcViewPolyNorms();
    void updateUniforms();
    void draw(Camera *camera);
    void translate(double x, double y, double z);

    Scene *theScene;
    QFile *file;
    QString objName;
    vector<shared_ptr<VertCell>> vertArray;
    vector<shared_ptr<SurfCell>> vertUsedArray;
    vector<shared_ptr<Double3D>> texVertArray;
    vector<shared_ptr<Double3D>> vertNormArray;
    vector<Double3D> viewNormArray;
    shared_ptr<SurfCell> surfHead;
    MaterialCell *materials;
    Double3D center;
    Double3D viewCenter;
    Sphere *boundingSphere;
    shared_ptr<ShaderProgram> theShader;
    GLuint *bufferIDs;
    GLuint *VAOIDs;
    Vec4Uniform *ambUniform;
    Vec4Uniform *diffUniform;
    Vec4Uniform *specUniform;
    FloatUniform *shinyUniform;
    IntUniform *useAmbTexUniform;
    IntUniform *useDiffTexUniform;
    IntUniform *useSpecTexUniform;
    Matrix3Uniform *normalMatUniform;
    Matrix4Uniform *modelMatUniform;
    vector<double> modelMat;
    bool active;
    bool firstDraw;
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
