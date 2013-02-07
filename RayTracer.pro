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
    lights.cpp

HEADERS  += mainwindow.h \
    scene.h \
    pmesh.h \
    objloaderbuffer.h \
    camera.h \
    MatrixManipulation/matrixops.h \
    MatrixManipulation/double3d.h \
    Uniforms/uniform.h \
    Uniforms/matrix4uniform.h \
    lights.h

FORMS    += mainwindow.ui

LIBS += -lGLEW

QMAKE_CXXFLAGS += -std=c++11