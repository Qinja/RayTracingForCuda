/**
*@brief 判断cuda函数是否错误
*@param[in] cudaStatus: cuda函数调用的返回值
type: 调用的cuda函数类型
*@param[out]
*@return
*@pre
*@post
*@code
cudaErrorYoN( cudaFree( s ), 3);
*@endcode
*@since 2014.1.18
*@author lichao
*/
#include "cudaErrorYoN.h"
void cudaErrorYoN(cudaError_t cudaStatus, int type)
{
	// 判断cuda函数是否错误
	if (cudaStatus != cudaSuccess)
	{
		cout << "cuda failed!" << endl;
		switch (type)
		{
		case 1:
			cout << "cudaMalloc failed!" << endl;
			break;
		case 2:
			cout << "cudaMemcpy failed!" << endl;
			break;
		case 3:
			cout << "cudaFree failed!" << endl;
			break;
		default:
			cout << "others failed!" << endl;
			break;
		}
	}
	else
	{
		//cout << "cuda successed!" << endl;
		switch (type)
		{
		case 1:
			//cout << "cudaMalloc!" << endl;
			break;
		case 2:
			//cout << "cudaMemcpy!" << endl;
			break;
		case 3:
			//cout << "cudaFree!" << endl;
			break;
		default:
			//cout << "others!" << endl;
			break;
		}
	}
}