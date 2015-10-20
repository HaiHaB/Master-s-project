#include"commonFunction.h"
#include "testFunction.h"

#ifndef ATOMMATCHINGOPT6_H
#define ATOMMATCHINGOPT6_H

__global__ void atomMatchingOpt6(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda6(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);


#endif
