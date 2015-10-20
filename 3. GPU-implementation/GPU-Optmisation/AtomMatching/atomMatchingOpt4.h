#include"commonFunction.h"
#include "testFunction.h"

#ifndef ATOMMATCHINGOPT4_H
#define ATOMMATCHINGOPT4_H

__global__ void atomMatchingOpt4(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda4(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);


#endif
