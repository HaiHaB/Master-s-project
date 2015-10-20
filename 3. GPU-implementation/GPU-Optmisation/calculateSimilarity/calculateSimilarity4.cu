#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include "commonFunction.h"
#include "TestFunction.h"
#define ARRAY_SIZE 49
#include<cmath>
using namespace std;

__global__ void calculateSimilarity4(float* c, const float *a, const int NA,
		const int NB, const int NMax);
cudaError_t calculateSimilarityWithCuda4(float* c, const float *a,
		const int NA, const int NB, const int NMax, string fileName);
/**
int main()
{
	const int NA = 7;
	const int NB = 7;
	const int NMax = 1;
	cudaError_t cudaStatus;

	float A[NA*NB*NMax] = {4};
	float C[NMax];

	cudaStatus = calculateSimilarityWithCuda4(C, A,NA,NB, NMax,
				"../calculateSimilarityTimeResult/calculateSimilarity1.txt");
	//print out C for correctness checking
	printf("C[] array is %.2f\n", C[0]);

	testFunction(A, NA, NB, NMax,  10,100,10,
				"../calculateSimilarityTimeResult/calculateSimilarity4.txt",
				&calculateSimilarityWithCuda4);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	return 0;
}
**/
/**
	Algorithm:
	(1) Sort the  elements of the atom match matri into order of decreasing similiarity
			(not necessary because we will need to find max anyway)
	(2) Scan the atom match matrix to find the remaining pair of atoms, one from A
			and one from B, that has the largest calculated value for S(i,j)
	(3) Store the rsulting equivalences as a tuple of the form [A(i) <-> B(j); S9i,j)]
	(4) Remove A(i) and B(j) from further consideration
	(5) Return to step 2 if it is possible to map further atoms in A to atoms in B


	input:
	  array of float c: containing result of total of NA max_element over NA
	  array of float a: containing coordinates NA*NB*NMax elements to find max_element
	  const int NA: number of atoms in molecule A
	  const int NB: number of atoms in each molecule in B
	  const int NMax: number of molecules in B


	output:
	  void
 **/
__global__ void calculateSimilarity4(float* c, float *a, const int NA,
		const int NB, const int NMax){
	float total;
	int position,start;
	int tid= blockIdx.x*blockDim.x+threadIdx.x;

	extern __shared__ float aShared[];
	if(tid<NA*NB*NMax) {
		aShared[tid] = a[tid];
	}
__syncthreads();

	if (tid<NMax) {
		//start is the first element every thread to check
		start = tid*NA*NB;
		// Initialised each thread's total to 0
		total = 0;
		//loop through NA atoms of molecule A
		for (int k =0;k<NA; k++) {
			/**
				 Step 2: Scan the atom match matrix to find the remaining pair of
				         atoms, one from A and one from B, that has the largest
				         calculated value for S(i,j)
			 **/
			// Find the max_element and position of max_element in the array of NA*NB float
			position = 0;
			float max =  aShared[start];

			for (int t = 0; t<NA*NB; t++) {
				if ( aShared[start + t] > max) {
					max =  aShared[start + t];
					position=t;
				}
			}

			/**
				 Step 3: Store the rsulting equivalences as a tuple of the form
				  			[A(i) <-> B(j); S9i,j)]
			 **/
			// Sum the max into total
			total = total + max;
			// Get the position of max_element in 2D array
			int a1 = position/NB; //y axis
			int b1 = position%NB; // x axis


			/**
					Step 4: Remove A(i) and B(j) from further consideration
			 **/
			// Set all the elements in the same row and column of max_element to 0
			// set all elements in the same y axis of max = 0
			for (int i =0; i<NB; i++ )  aShared[start + a1*NB+i] =0;
			// set all elements in the same x axis of max = 0
			for (int j =0; j<NA; j++)  aShared[start + j*NB+b1] =0;
		}
		//The similiarity score is total/NA
		c[tid] = total /NA;
	}
}





// Helper function for using CUDA to add vectors in parallel.
cudaError_t calculateSimilarityWithCuda4(float* c, const float *a,
		const int NA, const int NB, const int NMax, string fileName)
{
	float *dev_a = 0;
	float *dev_c = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float milliseconds;


	cudaStatus = cudaEventCreate(&start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(& start) failed! in scanWithCuda\n");
		goto Error;
	}

	cudaStatus =   cudaEventCreate(&stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(& stop) failed! in scanWithCuda\n");
		goto Error;
	}

	//Start recording time
	cudaStatus = cudaEventRecord(start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(start) failed! in scanWithCuda\n");
		goto Error;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, NMax*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! for dev_c in scanWithCuda\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, NA*NB*NMax * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! for dev_a in scanWithCuda\n");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, NA * NB *NMax* sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for dev_a! in scanWithCuda");
		goto Error;
	}


	// Launch a kernel on the GPU with one thread for each element.

	calculateSimilarity4<<<NMax/1024 +1, 1024, NA*NB*NMax*sizeof(float)>>>(dev_c, dev_a, NA, NB, NMax);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s in scanWithCuda\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! in scanWithCuda\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, NMax*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for dev_c!  in scanWithCuda`\n");
		goto Error;
	}

	cudaStatus =   cudaEventRecord(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(start) failed! in scanWithCuda\n");
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel! in scanWithCuda\n", cudaStatus);
		goto Error;
	}


	cudaStatus = cudaEventElapsedTime(&milliseconds, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventElapsedTime failed! in scanWithCuda\n");
		goto Error;
	}

	printf("elapsed time of scanning matrix of NA = %d, NB = %d, NMax = %d is %.4f milliseconds \n", NA,NB,NMax, milliseconds);
	writeResult2File (NA, NB, NMax,  milliseconds, "milliseconds", fileName);


	Error:
		cudaFree(dev_c);
	cudaFree(dev_a);

	return cudaStatus;
}
