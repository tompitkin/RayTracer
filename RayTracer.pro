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
    matrixops.cpp

HEADERS  += mainwindow.h \
    scene.h \
    pmesh.h \
    objloaderbuffer.h \
    camera.h \
    matrixops.h

FORMS    += mainwindow.ui

LIBS += -lGLEW

QMAKE_CXXFLAGS += -std=c++11
