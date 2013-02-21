#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <cstring>

class MatrixOps
{
public:
    static double *newIdentity();
    static double *multMat(double *m1, double *m2);
};

#endif // MATRIXOPS_H
