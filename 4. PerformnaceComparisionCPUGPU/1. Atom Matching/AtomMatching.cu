#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include <time.h>
#include "../CommonFiles/readfromfile.h"
#include "../CommonFiles/commonFunction.h"
#include<cmath>
using namespace std;



__global__ void atomMatching(float* c, const float *a,
		const float *b,const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda(float* c, const float *a, const float *b,
		const int NA, const int NB, const int NMax, float threshold, string outputFileName);
cudaError_t atomMatchingBreakDown (float* c, const float *a, const float *b,
		const int NA, const int NB, const int NMax, float threshold, string outputFileName);

/**

cudaError_t atomMatchingBreakDown (float* c, const float *a, const float *b,
		const int NA, const int NB, const int NMax, float threshold, string outputFileName) {

			cudaError_t cudaStatus;

			clock_t begin, end;
			//Start the clock
			begin = clock();

	// NA*NB*NMax < 2,000,000 seem to get the best performace from experiment data
	if (NA*NB*NMax <20000000) {
	 		cudaStatus = atomMatchingWithCuda(c, a, b, NA, NB, NMax, threshold, outputFileName);
		//reset cuda after the run
		cudaDeviceReset();
	}
	else {

			int maxMol, rounds;

		// Find the maximum minimal molecules for each rounds
		maxMol = 20000000/(NA*NB);
		rounds = NMax/maxMol + 1;


		float * btemp = new float[NB*NB*maxMol];
		float * ctemp = new float[NA*NB*maxMol];

		for (int i = 0; i < (rounds-1) ; i ++) {
			// copy the sub array from the orgnial array

			copy(b + i*(NB*NB*maxMol),// starting point to copy is i*NB*NB*maxMol
			    b + (i+1)*NB*NB*maxMol,	// end point to copy
          btemp );								// copy from the begining of btemp

	cudaStatus =	atomMatchingWithCuda(ctemp, a, btemp, NA, NB, maxMol, threshold, outputFileName);
	if (cudaStatus !=cudaSuccess)  {
		printf("Failed at NA = %d, NB = %d, NMax = %d\n", NA, NB, NMax);
		return cudaStatus;
	}
		cudaDeviceReset();
		//copy the result to Ctemp to C
		copy(ctemp,// starting point to copy is begining of ctemp
				ctemp +NA*NB*maxMol,	// end point to copy: NA *NB*NMol elements
				c+i*(NA*NB*maxMol) ); // starting point of float *c to copy to


		}

		//the last round wil only copy the remainder
		int remain = NMax - (rounds -1)*maxMol;

		copy(b + (rounds-1)*(NB*NB*maxMol),// starting point to copy is: last part of array
				b + NB*NB*NMax,	// end point to copy: the end of the array
				btemp );
		cudaStatus = atomMatchingWithCuda(ctemp, a, btemp, NA, NB, remain, threshold, outputFileName);
		cudaDeviceReset();
		//copy the result to Ctemp to C
		copy(ctemp,// starting point to copy is begining of ctemp
				ctemp + NA*NB*remain,	// end point to copy: NA *NB*NMol elements
				c+(rounds-1)*(NA*NB*maxMol) ); // starting point of float *c to copy to


		// free memory
		delete[] btemp;
		delete[] ctemp;
	}

	//end clock
	end = clock();
	//Get elapsed time
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf ("Elasped time of NA= %d, NB= %d, NMax = %d is %.6f seconds.\n", NA, NB, NMax, elapsed_secs );
	//print to a specified file
	writeResult2File (NA, NB, NMax,  elapsed_secs*1000.0, "milliseconds", outputFileName);


	return cudaStatus;
}
**/
/**
int main() {

	const int NA = 2;
	const int NB = 3;
	const int NMax = 1;
	cudaError_t cudaStatus;

	float A[NA*NA] = {2.0,0.0,4.0,1.0};
	float B[NB*NB*NMax] = {1.5,0.5,0.0,  4.2,0.7,1.0,  10.0,12.0,19.0};//{2.0,1.0,0.0,  4.0,0.5,2.0,  3.0,0.0,1.0,  1.5,0.5,0.0,  4.2,0.7,1.0,  10.0,12.0,19.0};
	float C[NB*NA*NMax];


	cudaStatus = compareWithCuda(C, A,B,NA,NB, NMax,0.5f,"CompareMatrixParallelRow.txt");

	for (int i = 0; i< NMax;i++) {
		for (int j =0; j<NB*NA; j++)  printf("%.2f ", C[NB*NA*i+j]);
		cout << endl;
	}

	const int NA1 = 7;
	float A1[NA1*NA1] = {0};
	for (int i = 0; i< NA1;i++) {
		for (int j =0; j<NA1; j++)  printf("%.2f ", A1[NA1*i+j]);
		cout << endl;
	}
	const int NB1 = 7;
	float B1[NB1*NB1*NMax] = {0};
	for (int i = 0; i< NB1;i++) {
		for (int j =0; j<NB1; j++)  printf("%.2f ", B1[NB1*i+j]);

		cout << endl;
	}

	//   printf("%.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", B[0],B[1],B[2],B[3],B[4],B[5],B[6]);
	float C1[NB1*NA1*NMax];

	testFunction (A1, B1, NA1, NB1,  NMax, 500, 4000, 500,0.5f, "CompareMatrixParallelRow.txt");

	//testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "CompareMatrixParallelRow.txt");
	//testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "CompareMatrixParallelRow.txt");
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


**/
/**
Read values from all text file with name containing "Atoms.txt" of a location

input:
  array of float c: containing result of NA*NB*NMax values
  array of float a: containing coordinates of molecule A (NA*NA values)
  array of float b: containing coordinates(NB*NB*NMax) of other molecules to compare with A
	array of bool array: containing NA*NB*NMax*NB 'false' value
  const int NA: number of atoms in molecule A
  const int NB: number of atoms in each molecule in B
  const int NMax: number of molecules in B
  float threshold: acceptable difference for coordinates to be the same

output:
  void
 **/

__global__ void atomMatching(float* c, const float *a,
		const float *b,  bool * array, const int NA, const int NB, const int NMax, float threshold){

	int  tempA, tempB, tid;
	float result;
	tid= blockIdx.x*blockDim.x+threadIdx.x;

	//Total number of threads to compare 1 molecule of NA atoms and NMax molecule
	// of NB atoms is NA*NB*NMax
	if (tid <NB*NA*NMax) {
		//First element in float *a for each thread to look at.
		tempA = (( tid / NB ) % NA ) * NA;
		//First element in float *b for each thread to look at.
		tempB = ( ( tid / (NA*NB) ) *NB + (tid% NB) )*NB;
		result = 0;

		//looping through all elements in float *a and float *b
		for (int k =0; k<NA; k++) {
			for (int t =0; t<NB; t++) {
				//first, check a b's element has not be matched with an a's element previously
				// then, check whether the a's and b's elements are similiry within a threshold
				if (!array[tid*NB+t] && (abs(a[tempA+k]-b[tempB+t])<= threshold)) {
					// increase the count
					result = result  +1;
					//signal that the b's element is now matched with one a's element
					array[(tid*NB+t)%NB] = true;
					//stop the current loop
					break;
				}
			}
		}

		c[tid] = result/(NA+NB-result);
	}
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t atomMatchingWithCuda(float* c, const float *a, const float *b,
		const int NA, const int NB, const int NMax, float threshold,
		string outputFileName)      {


	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	bool *dev_array = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float milliseconds;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//Start timer
	cudaStatus = cudaEventCreate(&start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(& start) failed!\n");
		goto Error;
	}

	cudaStatus =   cudaEventCreate(&stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate(& stop) failed!\n");
		goto Error;
	}

	cudaStatus = cudaEventRecord(start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(start) failed!\n");
		goto Error;
	}


	// Allocate GPU buffers for four vectors (three input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, NA*NB*NMax * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!! for dev_c\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, NA*NA * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! for dev_a\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, NB*NB * NMax *sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! for dev_b\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_array, NA*NB*NMax*NB *sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! for dev_array in compareWithCuda NA = %d NB = %d NMax = %d because of %s\n", NA, NB, NMax, cudaGetErrorString(cudaStatus));
		goto Error;
	}



	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, NA * NA * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for dev_a!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, NB  * NB *NMax *sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! for dev_b");
		goto Error;
	}

	//set memory for dev_array to false
	cudaStatus =  cudaMemset(dev_array,false,NA*NB*NMax*NB);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed for dev_array! in compareWithCuda because of %s \n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	//for (int i = 0; i <100; i++) {

	// Launch a kernel on the GPU with one thread for each element.
	atomMatching<<<(NA*NB*NMax)/1024+1,1024>>>(dev_c, dev_a, dev_b, dev_array, NA, NB, NMax, threshold);
	//}
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s for NA = %d NB = %d NMax= %d\n", cudaGetErrorString(cudaStatus),NA, NB,NMax);
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, NA*NB*NMax * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed for dev_c! NA = %d NB = %d NMax = %d because %s\n", NA, NB, NMax, cudaGetErrorString(cudaStatus));
		goto Error;
	}



	//stop timer
	cudaStatus =   cudaEventRecord(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(start) failed!\n");
	//goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
	}

	//Get elapsed times
	cudaStatus = cudaEventElapsedTime(&milliseconds, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventElapsedTime failed!\n");
		goto Error;
	}


	Error:
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_array);

	//Write result into the specified file
	writeResult2File (NA,NB,NMax,milliseconds,"milliseconds",outputFileName);

	return cudaStatus;
}
