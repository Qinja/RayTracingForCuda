#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <Windows.h>
#include <curand_kernel.h>
#include "cpu_bitmap.h"
#include "cudaErrorYoN.h"

using namespace std;
#define SEED_BASE 12345657890
#define DIMX 32
#define DIMY 16
#define SPP 1
#define MAX_DEPTH 1

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

#pragma region mathutil
struct random
{
	__device__ random(long id)
	{
		curand_init(SEED_BASE + id, 0, 0, &state);
	}
	__device__ float drand48()
	{
		return 0.5;
		return curand_uniform(&state);
	}
	__device__ vec3 random_in_unit_sphere()
	{
		vec3 p;
		do
		{
			p = 2.0 * vec3(drand48(), drand48(), drand48()) - vec3(1, 1, 1);
		} while (dot(p, p) >= 1.0);
		return p;
	}
private:
	curandState state;
};

#pragma endregion

#pragma region ray
struct ray
{
	__device__ inline ray() { }
	__device__ inline ray(const vec3& a, const vec3& b) :origin(a), direction(b) {}
	__device__ inline vec3 get_point_by_t(float t)const { return origin + t * direction; }
	vec3 origin;
	vec3 direction;
};
#pragma endregion

#pragma region camera
struct camera {
public:
	
	__device__ inline camera()
	{
		left_bottom = vec3(-2, -1, -1);
		horizontal = vec3(4, 0, 0);
		vertical = vec3(0, 2, 0);
		origin = vec3(0, 0, 0);
	}
	__device__ inline ray get_ray(float u, float v)
	{
		return ray(origin, left_bottom + u * horizontal + v * vertical);
	}
	vec3 origin;
	vec3 left_bottom;
	vec3 horizontal;
	vec3 vertical;
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

#pragma region __DeviceCode__
__device__ vec3 get_color(const ray& r, hitable *world,unsigned int maxDepth, random rnd)
{
	hit_record tmprec;
	vec3 tmpc(1, 1, 1);
	ray tmpr = r;
	for (int d = 0; d < maxDepth; d++)
	{
		if (world->hit(tmpr, 0.1, FLT_MAX, tmprec))
		{
			tmpc = tmpc * 0.5;
			vec3 target = tmprec.p + tmprec.normal +rnd.random_in_unit_sphere();
			tmpr = ray(tmprec.p, target - tmprec.p);
		}
		else
		{
			vec3 unitdir = tmpr.direction.normalize();
			float k = 0.5 *(unitdir.y + 1.0);
			tmpc = tmpc * ((1.0 - k)*vec3(1.0, 1.0, 1.0) + k * vec3(0.4, 0.6, 1.0));
			break;
		}
	}
	return tmpc;
}
__global__ void kernel(unsigned char *ptr, hitable_list **hl_ptr, camera **cam, unsigned int spp)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;	//横坐标
	int y = threadIdx.y + blockIdx.y * blockDim.y;	//纵坐标
	int offset = x + y * blockDim.x * gridDim.x;	//横数第几个点
	random rnd(offset);

	vec3 color(0, 0, 0);
	for (int s = 0; s < spp; s++)
	{
		float u = float(x+ rnd.drand48()) / DIMX;
		float v = float(y+ rnd.drand48()) / DIMY;
		ray r = (*cam)->get_ray(u, v);
		color += get_color(r, *hl_ptr, MAX_DEPTH, rnd);
	}
	color /= float(spp);
	//vec3 color((float)x / DIMX, (float)y / DIMY, 0.2);		//默认颜色
	ptr[offset * 4 + 0] = (int)(color.x * 255);
	ptr[offset * 4 + 1] = (int)(color.y * 255);
	ptr[offset * 4 + 2] = (int)(color.z * 255);
	ptr[offset * 4 + 3] = 255;
}
//分配内存
__global__ void AllocateOnDevice(hitable **h, hitable_list **list, camera **cam)
{
	*h = new sphere[2];
	h[0] = new sphere(vec3(0, 0, -1), 0.5);
	h[1] = new sphere(vec3(0, -100.5, -1), 100);
	*list = new hitable_list(h, 2);
	*cam = new camera();
}
// 释放内存
__global__ void DeleteOnDevice(hitable **h, hitable_list **list, camera **cam)
{
	delete[](*h);
	h = nullptr;
	delete[](*list);
	list = nullptr;
	delete[](*cam);
	cam = nullptr;
}
#pragma endregion

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

	dim3 grids(DIMX / 16, DIMY / 16);
	dim3 threads(16, 16);
	hitable **h_ptr = nullptr;
	hitable_list **hl_ptr = nullptr;
	camera **cam = nullptr;
	cudaErrorYoN(cudaMalloc((void **)&h_ptr, sizeof(hitable **)), 1);
	cudaErrorYoN(cudaMalloc((void **)&hl_ptr, sizeof(hitable_list **)), 1);
	cudaErrorYoN(cudaMalloc((void **)&cam, sizeof(camera **)), 1);

	AllocateOnDevice <<<1,1>>> (h_ptr, hl_ptr, cam);
	kernel <<<grids, threads >>>(dev_bitmap, hl_ptr, cam,SPP);
	cudaDeviceSynchronize();
	cudaGetLastError();
	DeleteOnDevice << <1, 1 >> >(h_ptr, hl_ptr, cam);

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