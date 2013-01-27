/*
 * main.cpp
 *
 *  Created on: Jan 26, 2013
 *      Author: tom
 */

#include <stdlib.h>
#include <GL/glut.h>

static void render(void)
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(400, 300);
	glutCreateWindow("Hello World");
	glutDisplayFunc(&render);
	glutMainLoop();
}




