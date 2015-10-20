
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include "atomMatchingOpt1.h"
#include "atomMatchingOpt2.h"
#include "atomMatchingOpt3.h"
#include "atomMatchingOpt4.h"
#include "atomMatchingOpt5.h"
#include "atomMatchingOpt6.h"
#include "atomMatchingOpt7.h"
using namespace std;

#include<cmath>
int main() {

    const int NA = 2;
    const int NB = 3;
    const int NMax = 1;
    cudaError_t cudaStatus;

    float A[NA*NA] = {2.0,0.0,4.0,1.0};
    float B[NB*NB*NMax] = {1.5,0.5,0.0,  4.2,0.7,1.0,  10.0,12.0,19.0};
    float C[NB*NA*NMax];

		cudaStatus = atomMatchingWithCuda1(C, A,B,NA,NB, NMax,0.5f,"../MatchingTimeResult/atomMatchingOpt1.txt");

		for (int i = 0; i< NMax;i++) {
		for (int j =0; j<NB*NA; j++)  printf("%.2f ", C[NB*NA*i+j]);
		cout << endl;
		}

    printf("\n atomMatchingOpt1\n");
		testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "../MatchingTimeResult/atomMatchingOpt1.txt",&atomMatchingWithCuda1);
		testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "../MatchingTimeResult/atomMatchingOpt1.txt",&atomMatchingWithCuda1);

    printf("\n atomMatchingOpt3\n");
    testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "../MatchingTimeResult/atomMatchingOpt3.txt",&atomMatchingWithCuda3);
    testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "../MatchingTimeResult/atomMatchingOpt3.txt",&atomMatchingWithCuda3);

    printf("\n atomMatchingOpt4\n");
    testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "../MatchingTimeResult/atomMatchingOpt4.txt",&atomMatchingWithCuda4);
  	testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "../MatchingTimeResult/atomMatchingOpt4.txt",&atomMatchingWithCuda4);

    printf("\n atomMatchingOpt5\n");
    testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "../MatchingTimeResult/atomMatchingOpt5.txt",&atomMatchingWithCuda5);
    testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "../MatchingTimeResult/atomMatchingOpt5.txt",&atomMatchingWithCuda5);

    testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "../MatchingTimeResult/atomMatchingOpt6.txt",&atomMatchingWithCuda6);
    testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "../MatchingTimeResult/atomMatchingOpt6.txt",&atomMatchingWithCuda6);

    testFunction (A, B, NA, NB,  NMax, 500, 4000, 500,0.5f, "../MatchingTimeResult/atomMatchingOpt7.txt",&atomMatchingWithCuda7);
    testFunction (A, B, NA, NB,  NMax, 10000, 100000, 10000,0.5f, "../MatchingTimeResult/atomMatchingOpt7.txt",&atomMatchingWithCuda7);

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
