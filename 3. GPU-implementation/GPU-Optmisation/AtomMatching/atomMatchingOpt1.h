#include"commonFunction.h"
#include "testFunction.h"

#ifndef ATOMMATCHINGOPT1_H
#define ATOMMATCHINGOPT1_H

__global__ void atomMatchingOpt1(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda1(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);


#endif
