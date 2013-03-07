#ifndef DOUBLE3D_H
#define DOUBLE3D_H

#include <math.h>
#include <vector>

using namespace std;

class Double3D
{
public:
    Double3D();
    Double3D(double nX, double nY, double nZ);
    Double3D(const Double3D *from);

    Double3D minus(Double3D t1);
    Double3D cross(Double3D t1);
    Double3D plus(Double3D t1);
    Double3D sDiv(double s);
    Double3D sMult(double s);
    Double3D preMultiplyMatrix(vector<double> m);
    Double3D getUnit();
    double dot(Double3D t1);
    float distanceTo(Double3D point);
    void unitize();

    double x;
    double y;
    double z;
};

#endif // DOUBLE3D_H
