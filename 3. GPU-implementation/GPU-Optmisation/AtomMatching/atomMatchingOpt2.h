#include"commonFunction.h"
#include "testFunction.h"

#ifndef ATOMMATCHINGOPT2_H
#define ATOMMATCHINGOPT2_H

__global__ void atomMatchingOpt2(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda2(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);


#endif
