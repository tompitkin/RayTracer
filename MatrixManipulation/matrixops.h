#ifndef MATRIXOPS_H
#define MATRIXOPS_H

#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdio.h>

using namespace std;

class MatrixOps
{
public:
    static vector<double> newIdentity();
    static vector<double> multMat(const vector<double> m1, const vector<double> m2);
    static vector<double> inverseTranspose(const vector<double> m);
    static vector<double> convertToMatrix1D(const vector<vector<double> > m);
    static vector<vector<double>> convertToMatrix2D(const vector<double> m);
    static vector<vector<double>> transpose(const vector<vector<double> > a);
    static vector<vector<double>> inverse(const vector<vector<double> > a);
    static vector<vector<double>> upperTriangle(const vector<vector<double> > matrix);
    static vector<vector<double>> adjoint(const vector<vector<double> > a);
    static vector<double> makeTranslation(double x, double y, double z);
    static double determinant(const vector<vector<double> > matrix);
};

#endif // MATRIXOPS_H
