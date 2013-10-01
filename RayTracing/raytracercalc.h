#ifndef RAYTRACERCALC_H
#define RAYTRACERCALC_H

#include <QThread>
#include "pmesh.h"
#include "Utilities/doublecolor.h"
#include "MatrixManipulation/double3d.h"

class Scene;
class RayTracer;

class RayTracerCalc : public QThread
{
    Q_OBJECT
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

        bool intersectSphere(PMesh *theObj, double *t);
        bool intersectTriangle(PMesh *theObj, PMesh::PolyCell *thePoly, RayTracerCalc::HitRecord *hrec, bool cull);

        Double3D Rd;
        Double3D Ro;
        int flags;
    };

    RayTracerCalc(Scene *theScene, RayTracer *rayTracer);
    virtual ~RayTracerCalc();

    DoubleColor trace(Ray ray, int numRecurs);
    DoubleColor shade(PMesh *theObj, Double3D point, Double3D normal, int materialIndex, bool backFacing, Ray ray, int numRecurs);
    unsigned char *castRays();
    void run();
    void calcBounds();
    void doViewTrans();
    double heightOfPixel();
    double widthOfPixel();
    bool traceLightRay(Ray ray);

    static const int EYE = 0;
    static const int REFLECT = 0x1;
    static const int INTERNAL_REFRACT = 0x2;
    static const int EXTERNAL_REFRACT = 0x4;
    constexpr static const double rhoAIR = 1.0;
    Scene *theScene;
    RayTracer *rayTracer;
    DoubleColor Ka;
    DoubleColor Kd;
    DoubleColor Ks;
    Double3D firstPixel;
    double shadeWeight = 0.0;
    double farTop;
    double farBottom;
    double farLeft;
    double farRight;
    double height;
    double width;
    bool cancelRayTrace;

signals:
    void percentageComplete(int);
};

#endif // RAYTRACERCALC_H
