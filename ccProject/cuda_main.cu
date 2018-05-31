#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "bmp.h"

#define SEED_BASE 12345657890
#define DIMX 1600
#define DIMY 800
#define SPP 16
#define MAX_DEPTH 2

#pragma region cudaCheck
static unsigned int cudaCallCount = 0;
__host__ void cudaCheck(cudaError_t cudaStatus)
{
	cudaCallCount++;
	// 判断cuda函数是否错误
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "cuda failed! CallCount: [" << cudaCallCount 
			<< "]Error Code: [" << cudaStatus << "]" << std::endl;
		system("pause");
		exit(1);
	}
}
#pragma endregion

#pragma region host_CPUBitmap
struct CPUBitmap {
	unsigned char *pixels;
	int x, y;
	__host__ __device__ CPUBitmap(int width, int height) {
		pixels = new unsigned char[width * height * 3];
		x = width;
		y = height;
	}
	__host__ __device__ ~CPUBitmap() {
		delete[] pixels;
	}
	__host__ __device__ unsigned char* get_ptr(void) const { return pixels; }
	__host__ __device__ long image_size(void) const { return x * y * 3; }
	__host__ __device__ void savetobmp(const char* filename)
	{
		WriteBmp(this->x, this->y, this->pixels, 3, filename);
	}
};
#pragma endregion

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
	__device__ random(int seed)
	{
		threadseed = SEED_BASE + seed* seed;
	}
	__device__ double drand48()
	{
		threadseed = (0x5DEECE66DLL * threadseed + 0xB16) & 0xFFFFFFFFFFFFLL;
		unsigned int x = threadseed >> 16;
		return  ((double)x / (double)0x100000000LL);
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
	unsigned long long threadseed;
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
class camera {
public:
	__device__ inline camera()
	{
		leftbottom = vec3(-2, -1, -1);
		hori = vec3(4, 0, 0);
		ver = vec3(0, 2, 0);
		origin = vec3(0, 0, 0);
	}
	__device__ inline ray get_ray (float u, float v)const
	{
		//vec3 a = origin;		//代码1
		////vec3 b = vertical;		//代码2
		//vec3 d = ver;			//代码3
		//vec3 e = hori;			//代码4
		//return ray();			//代码5
		return ray(origin, leftbottom + u * hori + v * ver);		//代码6
	}
private:
	vec3 origin;
	vec3 leftbottom;
	vec3 hori;
	vec3 ver;
};
#pragma endregion

#pragma region hit_record
struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
};
#pragma endregion

#pragma region sphere
class sphere  {
public:
	__device__ inline sphere() {}
	__device__ inline sphere(vec3 cen, float r) :center(cen), radius(r) {}
	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)const;
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
class hitable_list
{
public:
	__device__ inline hitable_list(){}
	__device__ inline hitable_list(sphere *l, int n) :list(l), list_size(n) {}
	__device__ inline bool hit(const ray& r, float t_min, float t_max, hit_record& rec)const;
	sphere *list;
	int list_size;
};
__device__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec)const
{
	hit_record tmp_rec;
	bool hit_anything = false;
	double cloest_so_far = t_max;
	for (int i = 0; i < list_size; i++)
	{
		if (list[i].hit(r, t_min, cloest_so_far, tmp_rec))
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
__device__ vec3 get_color(const ray& r,const hitable_list *world,unsigned int maxDepth, random rnd)
{
	hit_record tmprec;
	vec3 tmpc(1, 1, 1);
	ray tmpr = r;
	for (int d = 0; d < maxDepth; d++)
	{
		if (world->hit(tmpr, 0.1, FLT_MAX, tmprec))
		{
			tmpc = tmpc * 0.5;
			vec3 target = tmprec.p + tmprec.normal + rnd.random_in_unit_sphere();
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
__global__ void kernel(unsigned char *img,const hitable_list *hit_list,const camera *cam)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;	//横坐标
	int y = threadIdx.y + blockIdx.y * blockDim.y;	//纵坐标
	int z = threadIdx.z + blockIdx.z * blockDim.z;	//深度坐标
	long offset = x + y * blockDim.x * gridDim.x 
		+ z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;	//先深度，后纵向，再横向第几个点
	random rnd(offset);
	vec3 color(0, 0, 0);
	float u = float(x + rnd.drand48()) / DIMX;
	float v = float(y + rnd.drand48()) / DIMY;
	ray r = cam->get_ray(u, v);
	color += get_color(r, hit_list, MAX_DEPTH, rnd);
	img[offset * 3 + 0] = (int)(color.x * 255);
	img[offset * 3 + 1] = (int)(color.y * 255);
	img[offset * 3 + 2] = (int)(color.z * 255);
}
__global__ void mix(unsigned char *img,int spp)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;	//横坐标
	int y = threadIdx.y + blockIdx.y * blockDim.y;	//纵坐标
	int offset = x + y * blockDim.x * gridDim.x;	//横数第几个点
	long frame = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
	for (int i = 1; i < spp; i++)
	{
		img[offset * 3 + 0] += img[frame * i + offset * 3 + 0];
		img[offset * 3 + 1] += img[frame * i + offset * 3 + 1];
		img[offset * 3 + 2] += img[frame * i + offset * 3 + 2];
	}
	//img[offset * 3 + 0] /= spp;
	//img[offset * 3 + 1] /= spp;
	//img[offset * 3 + 2] /= spp;
}
//分配内存
__global__ void AllocateOnDevice(sphere* sph,hitable_list* hit_list, camera* cam)
{
	sph[0] = sphere(vec3(0, 0, -1), 0.5);
	sph[1] = sphere(vec3(0, -100.5, -1), 100);
	*hit_list = hitable_list(sph, 2);
	*cam = camera();
}
// 释放内存
__global__ void DeleteOnDevice( hitable_list *hit_list, camera *cam)
{
	sphere* s = hit_list->list;
	delete s;
	hit_list->list = nullptr;
	//delete hit_list;
	delete cam;
}
#pragma endregion

int main(void)
{
	// 记录起始时间
	cudaEvent_t start, stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));

	CPUBitmap bitmap(DIMX, DIMY);
	unsigned char *dev_bitmap;
	// 在GPU上分配内存以计算输出位图
	cudaCheck(cudaMalloc(&dev_bitmap, bitmap.image_size()*SPP));

	hitable_list *dev_hitlist = nullptr;
	camera *dev_cam = nullptr;
	sphere *dev_sphere = nullptr;
	cudaCheck(cudaMalloc(&dev_hitlist, sizeof(hitable_list)));
	cudaCheck(cudaMalloc(&dev_cam, sizeof(camera)));
	cudaCheck(cudaMalloc(&dev_sphere,2 * sizeof(sphere)));
	cudaMemcpy()

	printf("Starting AllocateOnDevice\n");
	AllocateOnDevice <<<1,1>>> (dev_sphere, dev_hitlist, dev_cam);


	dim3 grids3d(DIMX / 16, DIMY / 16, SPP / 16);
	dim3 blocks3d(16, 16, 16);
	printf("Starting kernel\n");
	kernel <<<grids3d, blocks3d >>>(dev_bitmap, dev_hitlist, dev_cam);


	cudaCheck(cudaDeviceSynchronize());
	//cudaCheck(cudaGetLastError());
	dim3 grids2d(DIMX / 16, DIMY / 16);
	dim3 blocks2d(16, 16);
	printf("Starting mix\n");
	mix << < grids2d, blocks2d >> > (dev_bitmap, SPP);

	printf("Starting DeleteOnDevice\n");
	DeleteOnDevice << <1, 1 >> >(dev_hitlist, dev_cam);

	// 将位图从GPU上复制到主机上
	cudaCheck(cudaMemcpy(bitmap.get_ptr(), dev_bitmap
		, bitmap.image_size(), cudaMemcpyDeviceToHost));

	// 记录结束时间
	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	// 显示运行时间
	float elapsedTime;
	cudaCheck(cudaEventElapsedTime(&elapsedTime, start, stop)); // 计算两个事件之间的时间
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	// 销毁事件
	cudaCheck(cudaEventDestroy(start));
	cudaCheck(cudaEventDestroy(stop));

	// 释放内存
	cudaCheck(cudaFree(dev_bitmap));

	// 显示位图
	bitmap.savetobmp("output/output.bmp");
	system("output\\output.bmp");
	system("pause");

}