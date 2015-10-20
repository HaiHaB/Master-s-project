#include"commonFunction.h"
#include "testFunction.h"

#ifndef CALCULATESIMILARITY1_H
#define CALCULATESIMILARITY1_H

__global__ void calculateSimilarity1(float* c, float *a, const int NA,
		const int NB, const int NMax);
cudaError_t calculateSimilarityWithCuda1(float* c, const float *a,
				const int NA, const int NB, const int NMax, string fileName);


#endif
