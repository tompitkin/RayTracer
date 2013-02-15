#-------------------------------------------------
#
# Project created by QtCreator 2013-02-01T10:27:31
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RayTracer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    scene.cpp \
    pmesh.cpp \
    objloaderbuffer.cpp \
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

HEADERS  += mainwindow.h \
    scene.h \
    pmesh.h \
    objloaderbuffer.h \
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

FORMS    += mainwindow.ui

LIBS += -lGLEW -lGLU

QMAKE_CXXFLAGS += -std=c++11
