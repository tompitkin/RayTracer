#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>
#include "MatrixManipulation/matrixops.h"
#include "MatrixManipulation/double3d.h"
#include "Uniforms/matrix4uniform.h"
#include "shaderprogram.h"

class Camera
{
public:
    Camera();
    virtual ~Camera();

    void updateCamera();
    void setViewport(double left, double right, double top, double bottom);
    void setFrustum(double left, double right, double bottom, double top, double nr, double fr);
    void frustumToPerspective();
    vector<double> makeViewingTransform();
    vector<double> makePersepctiveTransform();
    vector<double> makeInverseCamera();
    double getWindowHeight();
    double getWindowWidth();

    static const int FRUSTUM_MODE = 0;
    static const int PERSPECTIVE_MODE = 1;
    static const int ORTHOGRAPHIC_MODE = 2;
    Double3D eye = Double3D(0.0, 0.0, 10.0);
    Double3D center = Double3D(0.0, 0.0, 0.0);
    Double3D up = Double3D(0.0, 1.0, 0.0);
    double near = 9.0;
    double far = 10000.0;
    double fov;
    double windowWidth;
    double windowHeight;
    double windowRight = 5.0;
    double windowLeft = -5.0;
    double windowTop = 5.0;
    double windowBottom = -5.0;
    double viewportWidth;
    double viewportHeight;
    double viewportRight = 600.0;
    double viewportLeft = 0.0;
    double viewportTop = 600.0;
    double viewportBottom = 0.0;
    double aspectRatio;
    vector<double> viewMat;
    vector<double> projMat;
    vector<double> inverseCamera;
    shared_ptr<Matrix4Uniform> viewMatUniform;
    shared_ptr<Matrix4Uniform> projMatUniform;
    shared_ptr<Matrix4Uniform> invCamUniform;
    bool frustumChanged = false;
    bool perspectiveChanged = false;
    bool cameraMoved = false;
    int viewingMode;
};

#endif // CAMERA_H
