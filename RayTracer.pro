#-------------------------------------------------
#
# Project created by QtCreator 2013-02-01T10:27:31
#
#-------------------------------------------------

QT       += core gui opengl

QMAKE_CXXFLAGS += -std=c++11 -fopenmp

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
    Uniforms/uniform.cpp \
    Uniforms/matrix4uniform.cpp \
    Uniforms/intuniform.cpp \
    Uniforms/vec4uniform.cpp \
    Uniforms/floatuniform.cpp \
    Uniforms/vec3uniform.cpp \
    Uniforms/matrix3uniform.cpp \
    lights.cpp \
    shaderprogram.cpp \
    Utilities/gl.cpp \
    axes.cpp \
    shaderutils.cpp \
    sphere.cpp \
    RayTracing/raytracer.cpp \
    RayTracing/raytracercalc.cpp \
    RayTracing/cudaKernel.cu \
    RayTracing/raytracercuda.cpp

SOURCES -= RayTracing/cudaKernel.cu

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
    Uniforms/matrix3uniform.h \
    lights.h \
    shaderprogram.h \
    Utilities/gl.h \
    Utilities/doublecolor.h \
    axes.h \
    shaderutils.h \
    sphere.h \
    RayTracing/raytracer.h \
    RayTracing/raytracercalc.h \
    RayTracing/raytracercuda.h \
    RayTracing/cudaKernel.h \
    RayTracing/cutil_math.h

OTHER_FILES += Shaders/axes.frag \
    Shaders/axes.vert \
    Shaders/multi.frag \
    Shaders/multi.vert \
    Shaders/raytrace.frag \
    Shaders/raytrace.vert \

FORMS    += mainwindow.ui

LIBS += -lGLEW -lGLU #-fopenmp

# project build directories
DESTDIR     = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj

# Cuda sources
CUDA_SOURCES += RayTracing/cudaKernel.cu

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include \
    $$PWD
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bits Operating system
# libs used in your code
LIBS += -lcudart -lcuda              # Yeah! I've a new device. Adjust with your compute capability
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -G -g

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -arch=sm_20 -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
