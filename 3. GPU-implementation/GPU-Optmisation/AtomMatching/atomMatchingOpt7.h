#include"commonFunction.h"
#include "testFunction.h"

#ifndef ATOMMATCHINGOPT7_H
#define ATOMMATCHINGOPT7_H

__global__ void atomMatchingOpt7(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda7(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);


#endif
