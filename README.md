# RayTracingForCuda
学习了RayTracing in one weekend之后，尝试用OpenMP去优化多线程，但是毕竟CPU运算，效果并不特别明显。

看了一下Cuda编程，感觉还是可以改动的，不用费很大的事情。Cuda的编程实例参考：
[Github:muzichao](https://github.com/muzichao/Learning/tree/master/CUDA-learning/ray%20tracing)

这个工程会以RayTracing为参照，完成这个之后再往Next 做。RayTracing实例参考：
[Github:petershirley](https://github.com/petershirley/raytracinginoneweekend)

1. 图一，输出颜色

	给cuda添加一个char[]内存区，指定每一个Grid有W/16，H/16个块，每个块有16，16个线程。
	并行计算时获取当前点的屏幕坐标，随便输出个颜色。如下图。
	
	得到颜色之后，把图片从GPU中拷贝出来，至于要保存到图片，还是要显示到glfw都行。
	glfw在这个工程中用的不多。我就是随便试一下。

![图1](/ccProject/output/1.输出颜色.bmp)

2. 图二，随便画个天空

	建立若干个需要用到的类vec3，ray。
	在Cuda核心函数里面先试一下，写个__device__ vec3 getcolor(const& ray)随便输出一下天空。

![图2](/ccProject/output/2.天空.bmp)

3. 图三，画个球

	在Cuda核心函数里面随便试一下与球相交为红色。

![图3](/ccProject/output/3.天空+球.bmp)

4. 图四，给球加法线

	改进相交使得输出法线信息。

![图4](/ccProject/output/4.天空+球+法线可视化.bmp)

5. 图五，添加hitableList实现多物体

	添加hitable类，添加sphere和hitable_list。
	具有抽象继承的类数组要在设备端做，不能在CPU做。
	所以这里是一个比较大的坑。不好调试也不好理解。
	参考资料：[CSDN:Pxy](https://blog.csdn.net/u010445006/article/details/78219033)

![图4](/ccProject/output/5.hitableList.bmp)