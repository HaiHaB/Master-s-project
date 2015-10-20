#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include "commonFunction.h"
#include "TestFunction.h"
#include "calculateSimilarityCPU.h"
#include "calculateSimilarity.h"
#include<cmath>
using namespace std;



int main() {
	// Testing time of varying NB and NA for CPU and GPU
	//	Implementation of calculateSimilarity method for varying NA
	 const int NB = 20;
	 const int NMax = 1;

		for (int NA = 10; NA <70; NA+=10) {
		float *A = new float [NA*NB*NMax];

		//Initilised array A
		for (int i=0; i<NA*NB*NMax; i++) A[i] =2;

		//Test for CPU single thread implementation
		testFunctionCPU (A, NA, NB, NMax, 10, 200, 10, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityCPU(Varying NA, NB =20)x1.txt",	&calculateSimilarity);
		testFunctionCPU (A, NA, NB, NMax, 200, 1000, 200, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityCPU(Varying NA, NB =20)x1.txt",	&calculateSimilarity);
		testFunctionCPU (A, NA, NB, NMax, 1000, 10001, 1000, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityCPU(Varying NA, NB =20)x1.txt",	&calculateSimilarity);

		//Test for GPU implementation
		testFunction (A, NA, NB,  NMax, 10, 200, 10, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityGPU(Varying NA, NB =20)-Amortised(x100).txt", &calculateSimilarityWithCuda);
		testFunction (A, NA, NB,  NMax, 200, 1000, 200, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityGPU(Varying NA, NB =20)-Amortised(x100).txt",  &calculateSimilarityWithCuda);
		testFunction (A, NA, NB,  NMax, 1000, 10001, 1000, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityGPU(Varying NA, NB =20)-Amortised(x100).txt",  &calculateSimilarityWithCuda);

		printf("done with NA = %d, NB = %d \n");
		//free memory
		delete[] A;
	}


	//	Implementation of calculateSimilarity method for varying NB
	 const int NA1 = 20;

		for (int NB1 = 10; NB1 <70; NB1+=10) {
		float *A1 = new float [NA1*NB1*NMax];

		//Initilised array A
		for (int i=0; i<NA1*NB1*NMax; i++) A1[i] =2;

		//Test for CPU single thread implementation
		testFunctionCPU (A1, NA1, NB1, NMax, 10, 200, 10, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityCPU(Varying NB, NA =20)x1.txt",	&calculateSimilarity);
		testFunctionCPU (A1, NA1, NB1, NMax, 200, 1000, 200, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityCPU(Varying NB, NA =20)x1.txt",	&calculateSimilarity);
		testFunctionCPU (A1, NA1, NB1, NMax, 1000, 10001, 1000, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityCPU(Varying NB, NA =20)x1.txt",	&calculateSimilarity);

		//Test for GPU implementation
		testFunction (A1, NA1, NB1,  NMax, 10, 200, 10, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityGPU(Varying NB, NA =20)-Amortised(x100).txt", &calculateSimilarityWithCuda);
		testFunction (A1, NA1, NB1,  NMax, 200, 1000, 200, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityGPU(Varying NB, NA =20)-Amortised(x100).txt",  &calculateSimilarityWithCuda);
		testFunction (A1, NA1, NB1,  NMax, 1000, 10001, 1000, "../Result4CalculateSImilarity(School-Computers)/calculatesimilarityGPU(Varying NB, NA =20)-Amortised(x100).txt",  &calculateSimilarityWithCuda);

		printf("done with NA = %d, NB = %d \n");
		//free memory
		delete[] A1;
	}

	return 0;
}
