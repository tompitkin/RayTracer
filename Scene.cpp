/*
 * Scene.cpp
 *
 *  Created on: Jan 27, 2013
 *      Author: tom
 */

#include "Scene.h"

Scene::Scene(int* argc, char** argv, int x, int y) {
	init(argc, argv);

	glutInitWindowSize(x, y);
	glutCreateWindow("Ray Tracer");

	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
		return;
	}
	fprintf(stdout, "Driver Version String: %s\n", glGetString(GL_VERSION));
	if (glewIsSupported("GL_VERSION_3_0"))
		fprintf(stdout, "GL3 is available\n");
	else
	{
		fprintf(stdout, "GL3 is NOT available\n");
		return;
	}
	glutDisplayFunc(&display);
	glutMainLoop();
}

Scene::~Scene() {
	// TODO Auto-generated destructor stub
}

void Scene::init(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Scene::display()
{
    glutSwapBuffers();
}

