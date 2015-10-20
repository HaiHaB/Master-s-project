#include"commonFunction.h"
#include "testFunction.h"

#ifndef CALCULATESIMILARITY3_H
#define CALCULATESIMILARITY3_H

__global__ void calculateSimilarity3(float* c, float *a, const int NA,
		const int NB, const int NMax);
cudaError_t calculateSimilarityWithCuda3(float* c, const float *a,
				const int NA, const int NB, const int NMax, string fileName);


#endif
