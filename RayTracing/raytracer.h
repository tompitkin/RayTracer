#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <QThreadPool>
#include <QHBoxLayout>
#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <memory>
#include "pmesh.h"
#include "MatrixManipulation/double3d.h"
#include "Utilities/doublecolor.h"
#include "RayTracing/raytracercalc.h"

class Scene;

class RayTracer : public QObject
{
public:

    class PopUp : public QDialog
    {
    public:
        PopUp();

        QHBoxLayout layout;
        QLabel text;
        QProgressBar bar;
    };

    RayTracer(Scene *theScene);
    virtual ~RayTracer();

    void calc();
    void render();
    void writeBMP(const char *fname, int w,int h,unsigned char *img);

    static const int EYE = 0;
    static const int REFLECT = 0x1;
    static const int INTERNAL_REFRACT = 0x2;
    static const int EXTERNAL_REFRACT = 0x4;
    constexpr static const double rhoAIR = 1.0;
    Scene *theScene;
    RayTracerCalc *rayCalc;
    PopUp *popup;
    shared_ptr<ShaderProgram> rayTracerShaderProg;
    GLuint buffer;
    GLuint tex;
    GLbyte *data = nullptr;
    bool spheresOnly = false;
    bool reflections = false;
    bool refractions = false;
    bool shadows = false;
    bool checkerBackground = false;
    int maxRecursiveDepth = 0;
    double checkerSize = 1000.0;
};

#endif // RAYTRACER_H
