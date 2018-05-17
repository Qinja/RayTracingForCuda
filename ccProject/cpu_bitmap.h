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
			"#version 330 core\n"								//Shader���԰汾
			"\n"
			"layout (location = 0) in vec2 in_position;\n"		//��һ������ֵΪλ��
			"layout (location = 1) in vec2 in_tex_coord;\n"		//�ڶ�������ֵΪ��������
			"\n"
			"out vec2 tex_coord;\n"								//�����������
			"\n"
			"void main(void)\n"									//main����
			"{\n"
			"    gl_Position = vec4(in_position, 0.5, 1.0);\n"	//2άλ��ת��Ϊ3άλ�ã���λ�
			"    tex_coord = in_tex_coord;\n"					//ֱ�ӽ��������괫�ݵ�������ɫ��ȥ
			"}\n"
			;
		static const char* quad_shader_fs =
			"#version 330 core\n"						//Shader���԰汾
			"\n"
			"in vec2 tex_coord;\n"						//����ֵ��������
			"\n"
			"layout (location = 0) out vec4 color;\n"	//���ֵ������ɫ
			"\n"
			"uniform sampler2D tex;\n"					//uniform������������
			"\n"
			"void main(void)\n"							//main����
			"{\n"
			"    color = texture(tex, tex_coord);\n"	//ȡ���������ɫ
			"}\n"
			;
		// ������ɫ��
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
		// Ƭ����ɫ��
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
		GLuint base_prog = glCreateProgram();	//����Shader����
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
		// ɾ����ɫ��
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		GLuint quad_vbo,vao;
		glGenBuffers(1, &quad_vbo);										//ȡ��VBO���
		glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);						//��VBO
		static const GLfloat quad_data[] =								//��������
		{
			1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,		//�ĸ�����
			0.0f, 0.0f,1.0f, 0.0f,1.0f, 1.0f,0.0f, 1.0f				//�ĸ���������
		};
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad_data), quad_data, GL_STATIC_DRAW);	//ָ������
		glGenVertexArrays(1, &vao);						 //ȡ��VAO���
		glBindVertexArray(vao);							 //��VAO
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);			  //ָ��ǰ���Ƕ���
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void *)(8 * sizeof(float)));//ָ����������������
		glEnableVertexAttribArray(0);					  //����VAO����Ӧ��Shader��һ��in�ֶ�
		glEnableVertexAttribArray(1);					  //����VAO����Ӧ��Shader�ڶ���in�ֶ�

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
			glUseProgram(base_prog);				//ʹ�����Shader����
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture);
			glBindVertexArray(vao);					//ʹ�����VAO
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