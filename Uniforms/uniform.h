#ifndef UNIFORM_H
#define UNIFORM_H

#include <string>

using namespace std;

class Uniform
{
public:
    Uniform();

    virtual void update() = 0;

    string shaderVarName;
    bool off = false;
    bool needsUpdate;
};

#endif // UNIFORM_H
