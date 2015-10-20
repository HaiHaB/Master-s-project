#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include "commonFunction.h"
#include "Sort.h"
#include "SortCPU.h"
#include<cmath>
using namespace std;



int main() {
	// Testing time of varying NB and NA for CPU, CPU Parallel and GPU
	//	implementation of atomMathing

		int NA = 1;
		float* A = new float [NA];
		A[0] = 12;

		string * value = new string [NA];
		value[0] = "Testing";

			
			//Test for CPU single thread implementation
		//	testFunctionCPU (A, value, 1, 10, 100,1000,100, "../Result4Sort(School-Computers)/sortCPU-data.txt", "../Result4Sort(School-Computers)/sortCPU-time.txt", "Testing", &sortCPU);
		//	testFunctionCPU (A, value, 1, 10, 1000,10000,1000, "../Result4Sort(School-Computers)/sortCPU-data.txt", "../Result4Sort(School-Computers)/sortCPU-time.txt", "Testing", &sortCPU);
		//	testFunctionCPU (A, value, 1, 10, 10000,100000,10000, "../Result4Sort(School-Computers)/sortCPU-data.txt", "../Result4Sort(School-Computers)/sortCPU-time.txt", "Testing", &sortCPU);
		//	testFunctionCPU (A, value, 1, 10, 100000,1000001,100000, "../Result4Sort(School-Computers)/sortCPU-data.txt", "../Result4Sort(School-Computers)/sortCPU-time.txt", "Testing", &sortCPU);
		//	testFunctionCPU (A, value, 1, 10, 1000000,10000001,1000000, "../Result4Sort(School-Computers)/sortCPU-data.txt", "../Result4Sort(School-Computers)/sortCPU-time.txt", "Testing", &sortCPU);
			
			
			//Test for GPU implementation
			testFunctionCPU (A, value, 1, 10, 100,1000,100, "../Result4Sort(School-Computers)/sortGPU-data.txt", "../Result4Sort(School-Computers)/sortGPU-time.txt", "Testing", &sort);
			testFunctionCPU (A, value, 1, 10, 1000,10000,1000, "../Result4Sort(School-Computers)/sortGPU-data.txt", "../Result4Sort(School-Computers)/sortGPU-time.txt", "Testing", &sort);
			testFunctionCPU (A, value, 1, 10, 10000,100000,10000, "../Result4Sort(School-Computers)/sortGPU-data.txt", "../Result4Sort(School-Computers)/sortGPU-time.txt", "Testing", &sort);
			testFunctionCPU (A, value, 1, 10, 100000,1000000,100000, "../Result4Sort(School-Computers)/sortGPU-data.txt", "../Result4Sort(School-Computers)/sortGPU-time.txt" , "Testing", &sort);
			testFunctionCPU (A, value, 1, 10, 1000000,10000001,1000000, "../Result4Sort(School-Computers)/sortGPU-data.txt", "../Result4Sort(School-Computers)/sortGPU-time.txt", "Testing", &sort);

			//free memory
			delete[] A;
			delete[] value;

	return 0;
}
