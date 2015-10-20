#include"commonFunction.h"
#include "testFunction.h"

#ifndef CALCULATESIMILARITY2_H
#define CALCULATESIMILARITY2_H

__global__ void calculateSimilarity2(float* c, float *a, const int NA,
		const int NB, const int NMax);
cudaError_t calculateSimilarityWithCuda2(float* c, const float *a,
				const int NA, const int NB, const int NMax, string fileName);


#endif
