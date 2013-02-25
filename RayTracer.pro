#-------------------------------------------------
#
# Project created by QtCreator 2013-02-01T10:27:31
#
#-------------------------------------------------

QT       += core gui opengl

QMAKE_CXXFLAGS += -std=c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RayTracer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    scene.cpp \
    pmesh.cpp \
    Loaders/objloaderbuffer.cpp \
    camera.cpp \
    MatrixManipulation/matrixops.cpp \
    MatrixManipulation/double3d.cpp \
    Uniforms/uniform.cpp \
    Uniforms/matrix4uniform.cpp \
    Uniforms/intuniform.cpp \
    Uniforms/vec4uniform.cpp \
    Uniforms/floatuniform.cpp \
    Uniforms/vec3uniform.cpp \
    lights.cpp \
    shaderprogram.cpp \
    Utilities/gl.cpp \
    axes.cpp \
    shaderutils.cpp \

HEADERS  += mainwindow.h \
    scene.h \
    pmesh.h \
    Loaders/objecttypes.h \
    Loaders/objloaderbuffer.h \
    camera.h \
    MatrixManipulation/matrixops.h \
    MatrixManipulation/double3d.h \
    Uniforms/uniform.h \
    Uniforms/matrix4uniform.h \
    Uniforms/intuniform.h \
    Uniforms/vec4uniform.h \
    Uniforms/floatuniform.h \
    Uniforms/vec3uniform.h \
    lights.h \
    shaderprogram.h \
    Utilities/gl.h \
    axes.h \
    shaderutils.h \

OTHER_FILES += Shaders/axes.frag \
    Shaders/axes.vert \

FORMS    += mainwindow.ui

LIBS += -lGLEW -lGLU

HEADERS += \
    Utilities/doublecolor.h

SOURCES += \
    Utilities/doublecolor.cpp

HEADERS += \
    sphere.h

SOURCES += \
    sphere.cpp

