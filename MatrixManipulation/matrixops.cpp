#include <cstring>
#include <cmath>
#include <stdio.h>
#include "matrixops.h"

static int iDF;

double *MatrixOps::newIdentity()
{
    double *array = new double[16];
    memset(array, 0.0, sizeof(double)*16);
    array[0] = 1.0;
    array[5] = 1.0;
    array[10] = 1.0;
    array[15] = 1.0;
    return array;
}

double *MatrixOps::multMat(const double *m1, const double *m2)
{
    double *prod = new double[16];
    prod[0] = m1[0] * m2[0] + m1[4] * m2[1] + m1[8] * m2[2] + m1[12] * m2[3];
    prod[1] = m1[1] * m2[0] + m1[5] * m2[1] + m1[9] * m2[2] + m1[13] * m2[3];
    prod[2] = m1[2] * m2[0] + m1[6] * m2[1] + m1[10] * m2[2] + m1[14] * m2[3];
    prod[3] = m1[3] * m2[0] + m1[7] * m2[1] + m1[11] * m2[2] + m1[15] * m2[3];

    prod[4] = m1[0] * m2[4] + m1[4] * m2[5] + m1[8] * m2[6] + m1[12] * m2[7];
    prod[5] = m1[1] * m2[4] + m1[5] * m2[5] + m1[9] * m2[6] + m1[13] * m2[7];
    prod[6] = m1[2] * m2[4] + m1[6] * m2[5] + m1[10] * m2[6] + m1[14] * m2[7];
    prod[7] = m1[3] * m2[4] + m1[7] * m2[5] + m1[11] * m2[6] + m1[15] * m2[7];

    prod[8] = m1[0] * m2[8] + m1[4] * m2[9] + m1[8] * m2[10] + m1[12] * m2[11];
    prod[9] = m1[1] * m2[8] + m1[5] * m2[9] + m1[9] * m2[10] + m1[13] * m2[11];
    prod[10] = m1[2] * m2[8] + m1[6] * m2[9] + m1[10] * m2[10] + m1[14] * m2[11];
    prod[11] = m1[3] * m2[8] + m1[7] * m2[9] + m1[11] * m2[10] + m1[15] * m2[11];

    prod[12] = m1[0] * m2[12] + m1[4] * m2[13] + m1[8] * m2[14] + m1[12] * m2[15];
    prod[13] = m1[1] * m2[12] + m1[5] * m2[13] + m1[9] * m2[14] + m1[13] * m2[15];
    prod[14] = m1[2] * m2[12] + m1[6] * m2[13] + m1[10] * m2[14] + m1[14] * m2[15];
    prod[15] = m1[3] * m2[12] + m1[7] * m2[13] + m1[11] * m2[14] + m1[15] * m2[15];
    return prod;
}

double *MatrixOps::inverseTranspose(const double *m)
{
    double **temp = convertToMatrix2D(m);
    double **result = transpose((const double**)inverse((const double**)temp, 4));
    return convertToMatrix1D((const double**)result);
}

double *MatrixOps::convertToMatrix1D(const double **m)
{
    int num = 4;
    double *matrix = new double[num*num];
    for (int row = 0; row < num; row++)
        for (int col = 0; col < num; col++)
            matrix[(col*num)+row] = m[row][col];
    return matrix;
}

double MatrixOps::determinant(const double **matrix, int size)
{

    double det = 1;
    double **m = upperTriangle(matrix, size);
    for (int i = 0; i < size; i++)
        det = det * m[i][i];
    det = det * iDF;
    return det;
}

double **MatrixOps::convertToMatrix2D(const double *m)
{
    int num = sqrt(16);
    double **matrix = new double*[num];
    for (int i = 0; i < num; i++)
        matrix[i] = new double[num];
    for (int col = 0; col < num; col++)
        for (int row = 0; row < num; row++)
            matrix[row][col] = m[col*num+row];
    return matrix;
}

double **MatrixOps::transpose(const double **a)
{
    int tms = 4;
    double **m = new double*[tms];
    for (int x = 0; x < tms; x++)
        m[x] = new double[tms];
    for (int i = 0; i < tms; i++)
        for (int j = 0; j < tms; j++)
            m[i][j] = a[j][i];
    return m;
}

double **MatrixOps::inverse(const double **a, int size)
{
    double **m = new double*[size];
    for (int x = 0; x < size; x++)
        m[x] = new double[size];
    double **mm = adjoint(a, size);
    double det = determinant(a, size);
    double dd = 0;
    if (det == 0)
        fprintf(stderr, "MatrixOps.inverse(): Determinant Equals 0, Not Invertible.\n");
    else
        dd = 1/det;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            m[i][j] = dd * mm[i][j];
    return m;
}

double **MatrixOps::upperTriangle(const double **matrix, int size)
{
    double **m = new double*[size];
    copy(matrix, matrix+size, (const double**)m);
    for (int x = 0; x < size; x++)
    {
        m[x] = new double[size];
        copy(matrix[x], matrix[x]+size, m[x]);
    }
    double f1 = 0;
    double temp = 0;
    int v = 1;

    iDF=1;

    for (int col = 0; col < size-1; col++)
    {
        for (int row = col+1; row < size; row++)
        {
            v = 1;

            while (m[col][col]==0)
            {
                if (col+v >= size)
                {
                    iDF=0;
                    break;
                }
                else
                {
                    for (int c=0; c < size; c++)
                    {
                        temp = m[col][c];
                        m[col][c] = m[col+v][c];
                        m[col+v][c] = temp;
                    }
                    v++;
                    iDF = iDF * -1;
                }
            }
            if (m[col][col] != 0)
            {
                f1 = (-1) * m[row][col] / m[col][col];
                for (int i = 0; i < size; i++)
                    m[row][i] = f1*m[col][i] + m[row][i];
            }
        }
    }
    return m;
}

double **MatrixOps::adjoint(const double **a, int size)
{
    double **m = new double*[size];
    for (int x = 0; x < size; x++)
        m[x] = new double[size];
    int ii, jj, ia, ja;
    double det;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            ia = ja = 0;
            double **ap = new double*[size-1];
            for (int y = 0; y < (size-1); y++)
                ap[y] = new double[size-1];
            for (ii = 0; ii < size; ii++)
            {
                for (jj = 0; jj < size; jj++)
                {
                    if ((ii != i) && (jj != j))
                    {
                        ap[ia][ja] = a[ii][jj];
                        ja++;
                    }
                }
                if ((ii != i) && (jj != j))
                    ia++;
                ja = 0;
            }
            det = determinant((const double**)ap, size-1);
            m[i][j] = pow(-1, i+j) * det;
        }
    }
    m = transpose((const double**)m);
    return m;
}
