#ifndef SPHERE_H
#define SPHERE_H

#include "MatrixManipulation/double3d.h"

class PMesh;

class Sphere
{
public:
    Sphere(Double3D cent, double rad, PMesh *ownerObj);

    Double3D center;
    PMesh *theObj;
    double radius;
    double radiusSq;
};

#endif // SPHERE_H
