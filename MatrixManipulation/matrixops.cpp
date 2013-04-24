#include "matrixops.h"

static int iDF;

vector<double> MatrixOps::newIdentity()
{
    vector<double> array(16, 0);
    array[0] = 1.0;
    array[5] = 1.0;
    array[10] = 1.0;
    array[15] = 1.0;
    return array;
}

vector<double> MatrixOps::multMat(const vector<double> m1, const vector<double> m2)
{
    vector<double> prod(16, 0);
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

vector<double> MatrixOps::inverseTranspose(const vector<double> m)
{
    vector<vector<double>> temp = convertToMatrix2D(m);
    vector<vector<double>> result = transpose(inverse(temp));
    return convertToMatrix1D(result);
}

vector<double> MatrixOps::convertToMatrix1D(const vector<vector<double>> m)
{
    int num = m.size();
    vector<double> matrix(num*num, 0);
    for (int row = 0; row < num; row++)
        for (int col = 0; col < num; col++)
            matrix[(col*num)+row] = m[row][col];
    return matrix;
}

double MatrixOps::determinant(const vector<vector<double>> matrix)
{
    int size = matrix.size();
    double det = 1;
    vector<vector<double>> m = upperTriangle(matrix);
    for (int i = 0; i < size; i++)
        det = det * m[i][i];
    det = det * iDF;
    return det;
}

vector<vector<double>> MatrixOps::convertToMatrix2D(const vector<double> m)
{
    int num = sqrt(m.size());
    vector<vector<double>> matrix(num, vector<double>(4, 0));
    for (int col = 0; col < num; col++)
        for (int row = 0; row < num; row++)
            matrix[row][col] = m[col*num+row];
    return matrix;
}

vector<vector<double>> MatrixOps::transpose(const vector<vector<double>> a)
{
    int tms = a.size();
    vector<vector<double>> m(tms, vector<double>(tms, 0));
    for (int i = 0; i < tms; i++)
        for (int j = 0; j < tms; j++)
            m[i][j] = a[j][i];
    return m;
}

vector<vector<double>> MatrixOps::inverse(const vector<vector<double>> a)
{
    int size = a.size();
    vector<vector<double>> m(size, vector<double>(size, 0));
    vector<vector<double>> mm = adjoint(a);
    double det = determinant(a);
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

vector<vector<double>> MatrixOps::upperTriangle(const vector<vector<double>> matrix)
{
    int size = matrix.size();
    vector<vector<double>> m = matrix;
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

vector<vector<double>> MatrixOps::adjoint(const vector<vector<double>> a)
{
    int size = a.size();
    vector<vector<double>> m(size, vector<double>(size, 0));
    int ii, jj, ia, ja;
    double det;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            ia = ja = 0;
            vector<vector<double>> ap(size-1, vector<double>(size-1, 0));
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
            det = determinant(ap);
            m[i][j] = pow(-1, i+j) * det;
        }
    }
    m = transpose(m);
    return m;
    }

    vector<double> MatrixOps::makeTranslation(double x, double y, double z)
    {
        vector<double> array(16, 0);
        array[0] = 1.0;
        array[5] = 1.0;
        array[10] = 1.0;
        array[12] = x;
        array[13] = y;
        array[14] = z;
        array[15] = 1.0;
        return array;
    }
