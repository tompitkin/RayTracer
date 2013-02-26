#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <string>
#include <cstring>

using namespace std;

class MatrixOps
{
public:
    static double *newIdentity();
    static double *multMat(const double *m1, const double *m2);
    static double *inverseTranspose(const double *m);
    static double *convertToMatrix1D(const double **m);
    static double **convertToMatrix2D(const double *m);
    static double **transpose(const double **a);
    static double **inverse(const double **a, int size);
    static double **upperTriangle(const double **matrix, int size);
    static double **adjoint(const double **a, int size);
    static double determinant(const double **matrix, int size);
};

#endif // MATRIXOPS_H
