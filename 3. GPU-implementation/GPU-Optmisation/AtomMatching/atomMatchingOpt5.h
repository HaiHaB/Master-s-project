#include"commonFunction.h"
#include "testFunction.h"

#ifndef ATOMMATCHINGOPT5_H
#define ATOMMATCHINGOPT5_H

__global__ void atomMatchingOpt5(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda5(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);


#endif
