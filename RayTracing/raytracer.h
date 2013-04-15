#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <memory>
#include "pmesh.h"
#include "MatrixManipulation/double3d.h"
#include "Utilities/doublecolor.h"

class Scene;

class RayTracer
{
public:

    class HitRecord
    {
    public:
        HitRecord();
        HitRecord(double newT, double newU, double newV, Double3D newIntersect, Double3D newNormal, bool newBackFacing);

        PMesh::PolyCell *thisPoly = nullptr;
        Double3D intersectPoint;
        Double3D normal;
        double t;
        double u;
        double v;
        bool backfacing;
    };

    class Ray
    {
    public:
        Ray(Double3D dir, Double3D origin);
        Ray(Double3D dir, Double3D origin, int type);

        bool intersectSphere(shared_ptr<PMesh> theObj, double *t);
        bool intersectTriangle(PMesh *theObj, PMesh::PolyCell *thePoly, RayTracer::HitRecord *hrec, bool cull);

        Double3D Rd;
        Double3D Ro;
        int flags;
    };

    RayTracer(Scene *theScene);

    DoubleColor trace(Ray ray, int numRecurs);
    DoubleColor shade(shared_ptr<PMesh> theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, Ray ray, int numRecurs);
    GLbyte *castRays();
    void render();
    void calcBounds();
    void doViewTrans();
    void writeBMP(const char *fname, int w,int h,unsigned char *img);
    double heightOfPixel();
    double widthOfPixel();
    bool traceLightRay(Ray ray, shared_ptr<PMesh> fromObj);

    static const int EYE = 0;
    static const int REFLECT = 0x1;
    static const int INTERNAL_REFRACT = 0x2;
    static const int EXTERNAL_REFRACT = 0x4;
    constexpr static const double rhoAIR = 1.0;
    Scene *theScene;
    shared_ptr<ShaderProgram> rayTracerShaderProg;
    Double3D firstPixel;
    DoubleColor Ka;
    DoubleColor Kd;
    DoubleColor Ks;
    GLuint buffer;
    GLuint tex;
    bool spheresOnly = false;
    bool reflections = false;
    bool refractions = false;
    bool shadows = false;
    bool checkerBackground = false;
    int maxRecursiveDepth = 0;
    double checkerSize = 1000.0;
    double shadeWeight = 0.0;
    double height;
    double width;
    double farTop;
    double farBottom;
    double farLeft;
    double farRight;
};

#endif // RAYTRACER_H
