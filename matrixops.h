#ifndef MATRIXOPS_H
#define MATRIXOPS_H

class MatrixOps
{
public:
    static double *newIdentity();
    static double *multMat(double *m1, double *m2);
};

#endif // MATRIXOPS_H
