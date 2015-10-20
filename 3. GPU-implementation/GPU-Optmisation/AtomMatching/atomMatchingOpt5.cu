#include "atomMatchingOpt5.h"
using namespace std;

__global__ void atomMatchingOpt5(float* c, const float *a, const float *b,
	const int NA, const int NB, const int NMax, float threshold);
cudaError_t atomMatchingWithCuda5(float* c, const float *a, const float *b,
			const int NA, const int NB, const int NMax, float threshold,
			string outputFileName);

/**

int main() {

    const int NA = 2;
    const int NB = 3;
    const int NMax = 1;
    cudaError_t cudaStatus;

    float A[NA*NA] = {2.0,0.0,4.0,1.0};
    float B[NB*NB*NMax] = {1.5,0.5,0.0,  4.2,0.7,1.0,  10.0,12.0,19.0};//{2.0,1.0,0.0,  4.0,0.5,2.0,  3.0,0.0,1.0,  1.5,0.5,0.0,  4.2,0.7,1.0,  10.0,12.0,19.0};
    float C[NB*NA*NMax];


		cudaStatus = atomMatchingWithCuda5(C, A,B,NA,NB, NMax,0.5f,"atomMatchingWithCuda5.txt");

    for (int i = 0; i< NMax;i++) {
    for (int j =0; j<NB*NA; j++)  printf("%.2f ", C[NB*NA*i+j]);
    cout << endl;
    }

		testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "atomMatchingOpt5.txt",&atomMatchingWithCuda5);
    testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "atomMatchingOpt5.txt",&atomMatchingWithCuda5);
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
Compare every element in each row of a to every element in each row of b.
The match distances is stored in c.
refer to the documentation for explanation

input:
  array of float c: containing result
  array of float a: containing coordinates of molecule A
  array of float b: containing coordinates of other molecules to compare with A
  const int NA: number of atoms in molecule A
  const int NB: number of atoms in each molecule in B
  const int NMax: number of molecules in B
  float threshold: acceptable difference for coordinates to be the same

output:
  void
**/


__global__ void atomMatchingOpt5(float* c, const float *a,
		const float *b,const int NA, const int NB, const int NMax, float threshold){

	int  tempA, tempB, tid;
	float result;
	tid= blockIdx.x*blockDim.x+threadIdx.x;

	//Allocation of bool* array into local memory
	//Note: Due to limited shared memory, kernel call will fall with too large NMax
	extern __shared__ bool array[];
	if(tid <NA*NB*NB*NMax)   {
		array[tid]= false;
	}
	__syncthreads();

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

		c[tid] = result;
	}
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t atomMatchingWithCuda5(float* c, const float *a, const float *b,
		const int NA, const int NB, const int NMax, float threshold,
		string outputFileName)      {

	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;

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

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, NA*NB*NMax * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, NA*NA * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, NB*NB * NMax *sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
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

	// Launch a kernel on the GPU with one thread for each element.
		atomMatchingOpt5<<<128, (NA*NB*NMax)/128+1,NA*NB*NB*NMax*sizeof(bool) >>>(dev_c, dev_a, dev_b, NA, NB, NMax, threshold);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
		fprintf(stderr, "cudaMemcpy failed for dev_c!\n");
		goto Error;
	}


  //stop timer
  cudaStatus =   cudaEventRecord(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord(start) failed!\n");
		goto Error;
	}

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    goto Error;
  }

  //Get elapsed times
  cudaStatus = cudaEventElapsedTime(&milliseconds, start, stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventElapsedTime failed!\n");
		goto Error;
	}

  //Write result into the specified file
    	writeResult2File (NA,NB,NMax,milliseconds,"miilliseconds",outputFileName);


	Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
