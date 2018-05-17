#ifndef CUDAERRORYON_h
#define CUDAERRORYON_h
#include "cuda_runtime.h"
#include <iostream>
using namespace std;
void cudaErrorYoN(cudaError_t cudaStatus, int type);
#endif