#include"commonFunction.h"
#include "testFunction.h"

#ifndef ATOMMATCHINGOPT3_H
#define ATOMMATCHINGOPT3_H

__global__ void atomMatchingOpt3(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda3(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);


#endif
