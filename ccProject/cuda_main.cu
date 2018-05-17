#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <Windows.h>
#include "cpu_bitmap.h"
#include "cudaErrorYoN.h"

using namespace std;
#define DIMX 1600
#define DIMY 800
//#define rnd( x ) (x * rand() / RAND_MAX)
//#define INF     2e10f
//#define SPHERES 30

#pragma region vec3
struct vec3
{
	__device__ inline vec3() { }
	__device__ inline vec3(float e0, float e1, float e2) { x = e0; y = e1; z = e2; }
	float x;
	float y;
	float z;
	__device__ inline vec3& operator+=(const vec3 &v) { x += v.x; y += v.y; z += v.z; }
	__device__ inline vec3& operator-=(const vec3 &v) { x -= v.x; y -= v.y; z -= v.z; }
	__device__ inline vec3& operator/=(float n) { x /= n; y /= n; z /= n; }
	__device__ inline float length() const { return sqrt(x*x + y * y + z * z); }
	__device__ inline float squared_length() const { return (x*x + y * y + z * z); }
	__device__ inline vec3 normalize() const { float t = length(); return vec3(x / t, y / t, z / t); }
};
__device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) { return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z); }
__device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) { return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z); }
__device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) { return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z); }
__device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) { return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z); }
__device__ inline vec3 operator*(const vec3 &v1, const float t) { return vec3(v1.x * t, v1.y * t, v1.z * t); }
__device__ inline vec3 operator*(const float t, const vec3 &v1) { return vec3(v1.x * t, v1.y * t, v1.z * t); }
__device__ inline vec3 operator/(const vec3 &v1, const float t) { return vec3(v1.x / t, v1.y / t, v1.z / t); }
__device__ inline float dot(const vec3 &v1, const vec3 &v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
__device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
	return vec3((v1.y * v2.z - v1.z * v2.y),
		(v1.z * v2.x - v1.x * v2.z),
		(v1.x * v2.y - v1.y * v2.x));
}
#pragma endregion

#pragma region ray
struct ray
{
	__device__ inline ray() { }
	__device__ inline ray(const vec3& a, const vec3& b) :origin(a), direction(b) {}
	__device__ vec3 get_point_by_t(float t)const { return origin + t * direction; }
	vec3 origin;
	vec3 direction;
};
#pragma endregion

#pragma region hittable
struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
};

class hitable
{
public:
	__device__ hitable(){}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec)const = 0;
};
#pragma endregion

#pragma region sphere
class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 cen, float r) :center(cen), radius(r) {}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec)const;
	vec3 center;
	float radius;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec)const
{
	vec3 oc = r.origin - center;
	float a = dot(r.direction, r.direction);
	float b = dot(oc, r.direction);
	float c = dot(oc, oc) - radius * radius;
	float dis = b * b - a *c;
	if (dis > 0)
	{
		float tmp = (-b - sqrt(dis)) / a;
		if (tmp<t_max &&tmp>t_min)
		{
			rec.t = tmp;
			rec.p = r.get_point_by_t(tmp);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
		tmp = (-b + sqrt(dis)) / a;
		if (tmp<t_max &&tmp>t_min)
		{
			rec.t = tmp;
			rec.p = r.get_point_by_t(tmp);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
	}
	return false;
}
#pragma endregion

#pragma region hitable_list
class hitable_list : public hitable
{
public:
	__device__ hitable_list(){}
	__device__ hitable_list(hitable **l, int n) :list(l), list_size(n) {}
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec)const;
	hitable **list;
	int list_size;
};
__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec)const
{
	hit_record tmp_rec;
	bool hit_anything = false;
	double cloest_so_far = t_max;
	for (int i = 0; i < list_size; i++)
	{
		if (list[i]->hit(r, t_min, cloest_so_far, tmp_rec))
		{
			hit_anything = true;
			cloest_so_far = tmp_rec.t;
			rec = tmp_rec;
		}
	}
	return hit_anything;
}

#pragma endregion

__device__ vec3 get_color(const ray& r, hitable *world)
{
	hit_record rec;
	if (world->hit(r, 0.0, FLT_MAX, rec))
	{
		return 0.5 * (rec.normal + vec3(1, 1, 1));
	}
	else
	{
		vec3 u = r.direction.normalize();
		float t = 0.5 * (u.y + 1);
		return (1 - t)*vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1);
	}
}
__global__ void kernel(unsigned char *ptr)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;	//横坐标
	int y = threadIdx.y + blockIdx.y * blockDim.y;	//纵坐标
	int offset = x + y * blockDim.x * gridDim.x;	//横数第几个点


	vec3 bottom_left(-2, -1, -1);
	vec3 horizontal(4, 0, 0);
	vec3 vertial(0, 2, 0);
	vec3 origin(0, 0, 0);
	float u = float(x) / DIMX;
	float v = float(y) / DIMY;
	hitable * list[2];



	sphere b(vec3(0, -100.5, -1), 100);
	list[1] = &b;




	sphere a(vec3(0, 0, -1), 0.5);
	list[0] = &a;

	//hitable * list[2];
	//list[0] = new sphere(vec3(0, 0, -1), 0.5);
	//list[1] = new sphere(vec3(0, -100.5, -1), 100);



	//hitable * world = new hitable_list(list, 2);
	hitable_list world(list, 2);
	ray r(origin, bottom_left + u * horizontal + v * vertial);
	//vec3 color((float)x / DIMX, (float)y / DIMY, 0.2);		//默认颜色
	vec3 color = get_color(r, &world);


	ptr[offset * 4 + 0] = (int)(color.x * 255);
	ptr[offset * 4 + 1] = (int)(color.y * 255);
	ptr[offset * 4 + 2] = (int)(color.z * 255);
	ptr[offset * 4 + 3] = 255;
}

int main(void)
{
	// 记录起始时间
	cudaEvent_t     start, stop;
	cudaErrorYoN(cudaEventCreate(&start), 4);
	cudaErrorYoN(cudaEventCreate(&stop), 4);
	cudaErrorYoN(cudaEventRecord(start, 0), 4);
	CPUBitmap bitmap(DIMX, DIMY);
	unsigned char   *dev_bitmap;
	// 在GPU上分配内存以计算输出位图
	cudaErrorYoN(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()), 1);



	//// 为 Sphere数据集分配内存
	//cudaErrorYoN(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES), 1);
	//// 分配临时内存，在CPU上对其初始化，并复制到GPU内存上，然后释放内存
	//Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	//// 为SPHERES个圆分配位置，颜色，半径信息
	//for (int i = 0; i<SPHERES; i++)
	//{
	//	temp_s[i].r = rnd(1.0f);
	//	temp_s[i].g = rnd(1.0f);
	//	temp_s[i].b = rnd(1.0f);
	//	temp_s[i].x = rnd(1000.0f) - 500;
	//	temp_s[i].y = rnd(1000.0f) - 500;
	//	temp_s[i].z = rnd(1000.0f) - 500;
	//	temp_s[i].radius = rnd(100.0f) + 20;
	//}
	//cudaErrorYoN(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice), 2);
	//free(temp_s);


	// generate a bitmap
	dim3    grids(DIMX / 16, DIMY / 16);
	dim3    threads(16, 16);
	kernel <<<grids, threads >>>(dev_bitmap);
	// 将位图从GPU上复制到主机上
	cudaErrorYoN(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost), 2);
	// 记录结束时间
	cudaErrorYoN(cudaEventRecord(stop, 0), 4);
	cudaErrorYoN(cudaEventSynchronize(stop), 4);
	// 显示运行时间
	float   elapsedTime;
	cudaErrorYoN(cudaEventElapsedTime(&elapsedTime, start, stop), 4); // 计算两个事件之间的时间
	printf("Time to generate:  %3.1f ms\n", elapsedTime);
	// 销毁事件
	cudaErrorYoN(cudaEventDestroy(start), 4);
	cudaErrorYoN(cudaEventDestroy(stop), 4);
	// 释放内存
	cudaErrorYoN(cudaFree(dev_bitmap), 3);
	//cudaErrorYoN(cudaFree(s), 3);
	bitmap.displayimage();
	// 显示位图
	bitmap.savetobmp("output/output.bmp");
}