#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include "../CommonFiles/commonFunction.h"
#include "../CommonFiles/TestFunction.h"
#define ARRAY_SIZE 3
#include<cmath>
using namespace std;

#ifndef CALCLATESIMILARITY_H
#define CALCLATESIMILARITY_H

__global__ void calculateSimilarity(float* c, float *a, const int NA,
		const int NB, const int NMax);
cudaError_t calculateSimilarityWithCuda(float* c, const float *a,
		const int NA, const int NB, const int NMax, string fileName);


#endif
