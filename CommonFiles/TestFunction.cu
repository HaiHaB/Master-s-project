#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include "commonFunction.h"
#include "TestFunction.h"
#include <iostream>
#include<cmath>
using namespace std;

//To test for atom matching for GPU implementation
void testFunction(float *A, float *B, int NA, int NB, int NMax,  const int Min,
		const int Max, const int Incr, float threshold, string fileName,cudaError_t
		(*atomMatchingWithCuda) (float*,  const float *, const float *,  const int,
				const int , const int, float, string))    {


	for (int q = Min; q < Max; q +=Incr) {
		float* Bq = copy2NewArray(B, NB*NB*NMax, q);
		float *Cq = new float[NB*NA*NMax*q];

		atomMatchingWithCuda(Cq, A,Bq,NA,NB, NMax*q,threshold, fileName);



	//	printf("Memory available is %zu and total is %zu \n", avail, total);
/**
		//print out result for correctness checking
		if (q <Max) {
			printf ("q is %d\t", q);
			for (int j =0; j<NB*NA; j++) {
				printf("%.2f ", Cq[NB*NA*(q-1)+j]);
			}
			printf("\n");
		}
		**/

	 cudaDeviceReset();
/**
	 size_t avail;
	 size_t total;
	 cudaMemGetInfo( &avail, &total );

	 cout << "Memory available is " << avail << " and total is " << total <<
				 "for NA = " << NA << "NB = " << NB <<"NMax = " << NMax*q<<endl;
				 		delete[] Bq; **/
		delete[] Cq;
	}
}



//To test for calculate simlarity score for GPU implementation
void testFunction(float *A, int NA, int NB, int NMax,  const int Min,
		const int Max, const int Incr, string fileName,cudaError_t
		(*calculateSimilarityWithCuda) (float*, const float *,  const int,
				const int , const int, string))    {
	for (int q = Min; q < Max; q +=Incr) {
		float* A20 = copy2NewArray(A, NA*NB*NMax, q);
		float *C20 = new float [NMax*q];
		calculateSimilarityWithCuda(C20, A20,NA,NB, NMax*q, fileName);

		//print out for correctness checking
		printf("C[%d] array is %.2f\n", NMax*q-1, C20[NMax*q-1]);

		cudaDeviceReset();
		delete[] A20;
		delete[] C20;
	}
}
