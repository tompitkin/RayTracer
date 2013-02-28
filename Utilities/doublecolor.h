#ifndef DOUBLECOLOR_H
#define DOUBLECOLOR_H

class DoubleColor
{
public:
    DoubleColor();
    DoubleColor(double nR, double nG, double nB, double nA);

    float *toFloatv();

    double r;
    double g;
    double b;
    double a;
};

#endif // DOUBLECOLOR_H
