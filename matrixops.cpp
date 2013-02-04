#include "matrixops.h"

double *MatrixOps::newIdentity()
{
    double *array = new double[16];
    array[0] = 1.0;
    array[5] = 1.0;
    array[10] = 1.0;
    array[15] = 1.0;
    return array;
}
