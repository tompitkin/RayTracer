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
	Scene(int* argc, char** argv, int x, int y);
	virtual ~Scene();

	void init(int* argc, char** argv);
	static void display();
};

#endif /* SCENE_H_ */
