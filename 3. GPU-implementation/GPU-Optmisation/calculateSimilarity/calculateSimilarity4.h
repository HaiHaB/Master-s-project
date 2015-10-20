#include"commonFunction.h"
#include "testFunction.h"

#ifndef CALCULATESIMILARITY4_H
#define CALCULATESIMILARITY4_H

__global__ void calculateSimilarity4(float* c, float *a, const int NA,
		const int NB, const int NMax);
cudaError_t calculateSimilarityWithCuda4(float* c, const float *a,
				const int NA, const int NB, const int NMax, string fileName);


#endif
