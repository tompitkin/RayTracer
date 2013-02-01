/*
 * Scene.h
 *
 *  Created on: Jan 27, 2013
 *      Author: tom
 */

#ifndef SCENE_H_
#define SCENE_H_

#include <stdio.h>
#include <GL/glew.h>
#include <GL/glut.h>

class Scene {
public:
	Scene(int* argc, char** argv, int width, int height);
	virtual ~Scene();

	void init(int* argc, char** argv);
	static void display();

private:
	int width;
	int height;
};

#endif /* SCENE_H_ */
