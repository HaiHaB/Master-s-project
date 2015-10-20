#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include <time.h>
#include "commonFunction.h"

#include "atomMatchingCPU.h"
#include<cmath>
using namespace std;

/**
Compare every element in each row of a to every element in each row of b.
The match distances is stored in c.
Time elapsed were recorded in a specified file **/

void atomMatchingCPU(float* c, const float *a,  const float *b,
		const int NA, const int NB, const int NMax, float threshold, string fileName);
/**
int main() {

	const int NA = 2;
	const int NB = 3;
	const int NMax = 1;


	float A[NA*NA] = {2.0,0.0,4.0,1.0};
	float B[NB*NB*NMax] = {1.5,0.5,0.0,  4.2,0.7,1.0,  10.0,12.0,19.0};
  float C[NA*NB*NMax];
	atomMatchingCPU(C, A,  B,NA,NB, NMax,0.5,"CompareMatrixParallelRow.txt");

	for (int i = 0; i< NMax;i++) {
		for (int j =0; j<NB*NA; j++)  printf("%.2f ", C[NB*NA*i+j]);
		cout << endl;
	}

	printf("Start testFuncton\n");

	testFunctionCPU (A, B, NA, NB,  NMax, 10, 4000, 500,0.5f, "CompareMatrixParallelRow.txt", &atomMatchingCPU);
	testFunctionCPU (A, B, NA, NB,  NMax, 10, 1000000, 100000,0.5f, "CompareMatrixParallelRow.txt", &atomMatchingCPU);

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

	float C1[NB1*NA1*NMax];
	atomMatchingCPU(C1, A1,  B1, NA1,NB1, NMax,0.5, "AtomMatchingCPU.csv");
	testFunctionCPU(A1, B1, NA1, NB1,  NMax, 500, 4000, 500,0.5f, "AtomMatchingCPU.csv",
      &atomMatchingCPU);

	return 0;
}

 **/

/**
Compare every element in each row of a to every element in each row of b.
The match distances is stored in c.
Time elapsed were recorded in a specified file
refer to the documentation for explanation

input:
  array of float c: containing result
  array of float a: containing coordinates of molecule A
  array of float b: containing coordinates of other molecules to compare with A
  const int NA: number of atoms in molecule A
  const int NB: number of atoms in each molecule in B
  const int NMax: number of molecules in B
  float threshold: acceptable difference for coordinates to be the same
  string fileName: name of file to write elpsed time to

output:
  void
 **/

void atomMatchingCPU(float* c, const float *a,  const float *b,
		const int NA, const int NB, const int NMax, float threshold, string fileName){
	clock_t begin, end;
	//Start the clock
	begin = clock();


	int tempA, tempB;
	float result;


	//Total number of iteration to compare 1 molecule of NA atoms and NMax molecule
	// of NB atoms is NA*NB*NMax
	for (int tid =0; tid<NB*NA*NMax; tid++ ) {
		//First element in float *a for each iteration to look at.
		tempA = (( tid / NB ) % NA ) * NA;
		//First element in float *b for each iteration to look at.
		tempB = ( ( tid / (NA*NB) ) *NB + (tid% NB) )*NB;
		result = 0;

		//An array of bool of size NB of 'false' values
		bool *array = new bool[NB];
		for (int i =0; i<NB; i++) array[i] = false;

		//looping through all elements in float *a and float *b
		for (int k =0; k<NA; k++) {
			for (int t =0; t<NB; t++) {
				//first, check a b's element has not be matched with an a's element previously
				// then, check whether the a's and b's elements are similiry within a threshold
				if (!array[t] && (abs(a[tempA+k]-b[tempB+t])<= threshold)) {
					// increase the count
					result = result  +1;
					//signal that the b's element is now matched with one a's element
					array[t] = true;
					//stop the current loop
					break;
				}
			}
		}

		c[tid] = result/(NA+NB-result);
		delete[] array;
	}


	//end clock
	end = clock();
	//Get elapsed time
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf ("Elasped time of NA= %d, NB= %d, NMax = %d is %.6f seconds.\n", NA, NB, NMax, elapsed_secs );
	//print to a specified file
	//writeResult2File (NA, NB, NMax,  elapsed_secs, "seconds", fileName);
}
