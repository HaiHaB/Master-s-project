#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include "commonFunction.h"
#include "TestFunction.h"
#include "atomMatching.h"
#include "atomMatchingCPU.h"

#include<cmath>
using namespace std;



int main() {
	// Testing time of varying NB and NA for CPU and GPU
	//	Implementation of atomMatching method for varying NA
	 int NB = 20;
	for (int NA = 10; NA <70; NA+=10) {
	 int NMax = 1;

			float* A = new float [NA*NA];
			for (int i=0; i<NA*NA; i++) A[i] =2;
			float *B= new float[NB*NB*NMax];
			//Initilised array B
			for (int i=0; i<NB*NB*NMax; i++) B[i] =2;

			//Test for GPU implementation
			testFunction (A, B, NA, NB,  NMax, 10, 200, 10,0.5f, "../Result4AtomMatching/atomMatchingGPU(Varying NA, NB =20)x1.txt", &atomMatchingWithCuda);
			testFunction (A, B, NA, NB,  NMax, 200, 1000, 200,0.5f, "../Result4AtomMatching/atomMatchingGPU(Varying NA, NB =20)x1.txt", &atomMatchingWithCuda);
			testFunction (A, B, NA, NB,  NMax, 1000, 10001, 1000,0.5f, "../Result4AtomMatching/atomMatchingGPU(Varying NA, NB =20)x1.txt", &atomMatchingWithCuda);


			//Test for CPU single thread implementation
			testFunctionCPU (A, B, NA, NB,  NMax, 10, 200, 10,0.5f, "../Result4AtomMatching/atomMatchingCPU(Varying NA, NB =20).txt", &atomMatchingCPU);
			testFunctionCPU (A, B, NA, NB,  NMax,200, 1000, 200,0.5f, "../Result4AtomMatching/atomMatchingCPU(Varying NA, NB =20).txt", &atomMatchingCPU);
			testFunctionCPU (A, B, NA, NB,  NMax,1000, 10001, 1000,0.5f, "../Result4AtomMatching/atomMatchingCPU(Varying NA, NB =20).txt", &atomMatchingCPU);

			printf ("done with NA = %d, NB = %d\n", NA, NB );
			//free memory
			delete[] B;
			//free memory
			delete[] A;
}


	//	Implementation of atomMatching method for varying NB
 int NA1 = 20;
 for (int NB1 = 10; NB1 <70; NB1+=10) {

		float* A1 = new float [NA1*NA1];
		for (int i=0; i<NA1*NA1; i++) A1[i] =2;
		float *B1 = new float[NB1*NB1*NMax];
		//Initilised array B
		for (int i=0; i<NB1*NB1*NMax; i++) B[i] =2;

		//Test for GPU implementation
		testFunction (A1, B1, NA1, NB1,  NMax, 10, 200, 10,0.5f, "../Result4AtomMatching/atomMatchingGPU(Varying NB, NA =20)x1.txt", &atomMatchingWithCuda);
		testFunction (A1, B1, NA1, NB1,  NMax, 200, 1000, 200,0.5f, "../Result4AtomMatching/atomMatchingGPU(Varying NB, NA =20)x1.txt", &atomMatchingWithCuda);
		testFunction (A1, B1, NA1, NB1,  NMax, 1000, 10001, 1000,0.5f, "../Result4AtomMatching/atomMatchingGPU(Varying NB, NA =20)x1.txt", &atomMatchingWithCuda);


		//Test for CPU single thread implementation
		testFunctionCPU (A1, B1, NA1, NB1,  NMax, 10, 200, 10,0.5f, "../Result4AtomMatching/atomMatchingCPU(Varying NB, NA =20).txt", &atomMatchingCPU);
		testFunctionCPU (A1, B1, NA1, NB1,  NMax,200, 1000, 200,0.5f, "../Result4AtomMatching/atomMatchingCPU(Varying NB, NA =20).txt", &atomMatchingCPU);
		testFunctionCPU (A1, B1, NA1, NB1,  NMax,1000, 10001, 1000,0.5f, "../Result4AtomMatching/atomMatchingCPU(Varying NB, NA =20).txt", &atomMatchingCPU);

		printf ("done with NA = %d, NB = %d\n", NA1, NB1 );
		//free memory
		delete[] B1;
		//free memory
		delete[] A1;
}


	return 0;
}
