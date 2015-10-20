#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include "../CommonFiles/readfromfile.h"
#include<cmath>
using namespace std;

#ifndef ATOMMATCHING_H
#define ATOMMATCHING_H

__global__ void atomMatching(float* c, const float *a,
		const float *b,const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda(float* c, const float *a, const float *b,
		const int NA, const int NB, const int NMax, float threshold, string outputFileName);

cudaError_t atomMatchingBreakDown (float* c, const float *a, const float *b,
		const int NA, const int NB, const int NMax, float threshold, string outputFileName);

#endif
