#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include "commonFunction.h"
#include "TestFunction.h"
#include "calculateSimilarity1.h"
#include "calculateSimilarity2.h"
#include "calculateSimilarity3.h"
#include "calculateSimilarity4.h"

#define ARRAY_SIZE 3
#include<cmath>
using namespace std;

__global__ void calculateSimilarity1(float* c, float *a, const int NA,
		const int NB, const int NMax);
cudaError_t calculateSimilarityWithCuda1(float* c, const float *a,
				const int NA, const int NB, const int NMax, string fileName);


int main()
{
	const int NA = 7;
	const int NB = 7;
	const int NMax = 1;
	cudaError_t cudaStatus;

	float A[NA*NB*NMax] = {4};
	float C[NMax];

	cudaStatus = calculateSimilarityWithCuda1(C, A,NA,NB, NMax, "../calculateSimilarityTimeResult/calculateSimilarity1.txt");
	//print out C for correctness checking
	printf("C[] array is %.2f\n", C[0]);

	printf("\ncalculateSimilarity1\n");
	testFunction(A, NA, NB, NMax, 500, 4000, 500,"../calculateSimilarityTimeResult/calculateSimilarity1.txt", &calculateSimilarityWithCuda1);
	testFunction(A, NA, NB, NMax,  1000,10000,1000,"../calculateSimilarityTimeResult/calculateSimilarity1.txt", &calculateSimilarityWithCuda1);

	printf("\ncalculateSimilarity2\n");
	testFunction(A, NA, NB, NMax, 500, 4000, 500,"../calculateSimilarityTimeResult/calculateSimilarity2.txt", &calculateSimilarityWithCuda2);
	testFunction(A, NA, NB, NMax,  1000,10000,1000,"../calculateSimilarityTimeResult/calculateSimilarity2.txt", &calculateSimilarityWithCuda2);

	printf("\ncalculateSimilarity4\n");
	testFunction(A, NA, NB, NMax, 200, 4000, 200,"../calculateSimilarityTimeResult/calculateSimilarity4.txt", &calculateSimilarityWithCuda4);
	testFunction(A, NA, NB, NMax,  1000,10000,1000,"../calculateSimilarityTimeResult/calculateSimilarity4.txt", &calculateSimilarityWithCuda4);

	printf("\ncalculateSimilarity3\n");
	testFunction(A, NA, NB, NMax, 500, 4000, 500,"../calculateSimilarityTimeResult/calculateSimilarity3.txt", &calculateSimilarityWithCuda1);
	testFunction(A, NA, NB, NMax,  1000,10000,1000,"../calculateSimilarityTimeResult/calculateSimilarity3.txt", &calculateSimilarityWithCuda3);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
