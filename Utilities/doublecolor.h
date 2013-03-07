#ifndef DOUBLECOLOR_H
#define DOUBLECOLOR_H

class DoubleColor
{
public:
    DoubleColor();
    DoubleColor(double nR, double nG, double nB, double nA);

    void plus(DoubleColor other);
    void scale(double scaleValue);
    float *toFloatv();

    double r;
    double g;
    double b;
    double a;
};

#endif // DOUBLECOLOR_H
