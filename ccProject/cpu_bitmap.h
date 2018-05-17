/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/
#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__
#include "glad.h"
#include "glfw3.h"
#include "bmp.h"
struct CPUBitmap {
	unsigned char    *pixels;
	int     x, y;
	CPUBitmap(int width, int height) {
		pixels = new unsigned char[width * height * 4];
		x = width;
		y = height;
	}
	~CPUBitmap() {
		delete[] pixels;
	}
	unsigned char* get_ptr(void) const { return pixels; }
	long image_size(void) const { return x * y * 4; }

	void displayimage() {
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		GLFWwindow* window = glfwCreateWindow(this->x,this->y, "Cuda RayTracing", NULL, NULL);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			return;
		}
		glfwMakeContextCurrent(window);
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			return;
		}


		static const char* quad_shader_vs =
			"#version 330 core\n"								//Shader语言版本
			"\n"
			"layout (location = 0) in vec2 in_position;\n"		//第一个输入值为位置
			"layout (location = 1) in vec2 in_tex_coord;\n"		//第二个输入值为纹理坐标
			"\n"
			"out vec2 tex_coord;\n"								//输出纹理坐标
			"\n"
			"void main(void)\n"									//main函数
			"{\n"
			"    gl_Position = vec4(in_position, 0.5, 1.0);\n"	//2维位置转换为3维位置，齐次化
			"    tex_coord = in_tex_coord;\n"					//直接将纹理坐标传递到后面着色器去
			"}\n"
			;
		static const char* quad_shader_fs =
			"#version 330 core\n"						//Shader语言版本
			"\n"
			"in vec2 tex_coord;\n"						//输入值纹理坐标
			"\n"
			"layout (location = 0) out vec4 color;\n"	//输出值像素颜色
			"\n"
			"uniform sampler2D tex;\n"					//uniform变量，采样器
			"\n"
			"void main(void)\n"							//main函数
			"{\n"
			"    color = texture(tex, tex_coord);\n"	//取纹理，输出颜色
			"}\n"
			;
		// 顶点着色器
		int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &quad_shader_vs, NULL);
		glCompileShader(vertexShader);
		int success;
		char infoLog[512];
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
			return;
		}
		// 片段着色器
		int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &quad_shader_fs, NULL);
		glCompileShader(fragmentShader);
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
			return;
		}
		GLuint base_prog = glCreateProgram();	//创建Shader程序
		glAttachShader(base_prog, vertexShader);
		glAttachShader(base_prog, fragmentShader);
		glLinkProgram(base_prog);
		// check for linking errors
		glGetProgramiv(base_prog, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(base_prog, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			return;
		}
		// 删除着色器
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		GLuint quad_vbo,vao;
		glGenBuffers(1, &quad_vbo);										//取得VBO编号
		glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);						//绑定VBO
		static const GLfloat quad_data[] =								//生成数据
		{
			1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,		//四个顶点
			0.0f, 0.0f,1.0f, 0.0f,1.0f, 1.0f,0.0f, 1.0f				//四个纹理坐标
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad_data), quad_data, GL_STATIC_DRAW);	//指定数据
		glGenVertexArrays(1, &vao);						 //取得VAO编号
		glBindVertexArray(vao);							 //绑定VAO
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);			  //指定前面是顶点
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void *)(8 * sizeof(float)));//指定后面是纹理坐标
		glEnableVertexAttribArray(0);					  //启用VAO，对应到Shader第一个in字段
		glEnableVertexAttribArray(1);					  //启用VAO，对应到Shader第二个in字段

		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		// set the texture wrapping parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		// set texture filtering parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, this->x, this->y, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->pixels);

		while (!glfwWindowShouldClose(window))
		{
			glClearColor(0.2, 0.6, 0.5, 1.0);
			glClear(GL_COLOR_BUFFER_BIT);
			glUseProgram(base_prog);				//使用这个Shader程序
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture);
			glBindVertexArray(vao);					//使用这个VAO
			glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
		glfwTerminate();
		return;
	}
	void savetobmp(const char* filename)
	{
		WriteBmp(this->x, this->y, this->pixels,4, filename);
	}
};
#endif  // __CPU_BITMAP_H__