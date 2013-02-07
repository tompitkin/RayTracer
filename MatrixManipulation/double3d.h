#ifndef DOUBLE3D_H
#define DOUBLE3D_H

#include <math.h>

class Double3D
{
public:
    Double3D();
    Double3D(double nX, double nY, double nZ);
    Double3D minus(Double3D t1);
    Double3D cross(Double3D t1);
    void unitize();

    double x;
    double y;
    double z;
};

#endif // DOUBLE3D_H