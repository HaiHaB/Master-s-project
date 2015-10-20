#include <algorithm>
#include <stdio.h>
#include "commonFunction.h"
#include "calculateSimilarityCPU.h"
#include <time.h>
#include<cmath>
using namespace std;

/*Calculate simlarity from float *a and put result to float *c */
void calculateSimilarity(float* c, float *a, const int NA,	const int NB,
		const int NMax, string fileName);
/**
int main()
{
	const int NA = 7;
	const int NB = 7;
	const int NMax = 1;

	float A[NA*NB*NMax] = {4};
	writeResult2File (4, 4.0, "dsdsa", "Testing.txt");
	testFunctionCPU(A, NA, NB, NMax,  1000,10000,1000,"calculateSimilarityCPU.txt", &calculateSimilarity);


			//		void calculateSimilarity(float* c, float *a, const int NA,	const int NB,
			//				const int NMax, string fileName)
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

	The itermolecular similarity is sum[S(i,j) from 1 to NA]/NA


	input:
	  array of float c: containing result of total of NA max_element over NA
	  array of float a: containing coordinates NA*NB*NMax elements to find max_element
	  const int NA: number of atoms in molecule A
	  const int NB: number of atoms in each molecule in B
	  const int NMax: number of molecules in B


	output:
	  void
 **/
void calculateSimilarity(float* c, float *a, const int NA,	const int NB,
		const int NMax, string fileName){

	clock_t begin, end;
	//Start the clock
	begin = clock();
	float total,max;
	int position,start;

//	printf("Before 4 loop\n");
	//To compare 1 molecule of A to NMax molecule B, iterate NMax times
	for (int tid =0; tid < NMax; tid++) {

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
			max =  a[start];
			for (int t = 0; t<NA*NB; t++) {
				if ( a[start + t] > max) {
					max =  a[start + t];
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

//			printf("before ab\n");

			/**
					Step 4: Remove A(i) and B(j) from further consideration
			 **/
			// Set all the elements in the same row and column of max_element to 0
			// set all elements in the same y axis of max = 0
			for (int i =0; i<NB; i++ )  a[start + a1*NB+i] =0;
			// set all elements in the same x axis of max = 0
			for (int j =0; j<NA; j++)  a[start + j*NB+b1] =0;
		}
		//The similiarity score is total/NA
		c[tid] = total /NA;

	}


	//end clock
	end = clock();
	//Get elapsed time
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf ("Elasped time of NA= %d, NB= %d, NMax = %d is %.6f seconds.\n", NA, NB, NMax, elapsed_secs );
	//print to a specified file
	writeResult2File (NA, NB, NMax,  elapsed_secs, "seconds", fileName);

}
