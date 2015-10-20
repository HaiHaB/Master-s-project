#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <iomanip>  // for setw()
#include <time.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "../CommonFiles/readFromFile.h"
#include "../CommonFiles/commonFunction.h"
#include "../CommonFiles/parsedFile.h"
//#include "../CommonFiles/fileCollection.h"
#include "AtomMatching.h"
#include "calculateSimilarity.h"
#include "Sort.h"

using namespace std;

void interactive4User (vector <info> allValue);
void atomMappingGPU(string mol2Compare, vector <info> allValue, int top,  string resultFileName,  string timeFileName);

int main() {

//	string allLocation = "C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\Test2\\";
//	string allLocation = "C:\\Users\\Hai Ha\\Desktop\\Done Processing\\";

	//string allLocation = "/media/A015-9565/HH_SVN_BHAM/Test2/";
	//string allLocation = "/media/A015-9565/Data-processed/";
	string allLocation = "../Data-processed/";
	printf("starting reading data. Please wait for a moment!CH\n");
	//parsedAllFile(allLocation);
	vector <info > allValue =  readAllValue (allLocation);
	interactive4User (allValue);


	return 0;
}

/**A simple interactive program asking the user:
        a. the name and location of the molecule he/she wants to compare.
        b. number of top molecule he/she would like to see
        c. once the comparision is done, whether the user want to compare
        another molecule

input:
  vector <info> allValue: containing all data of all molecules in the datase.
        Reading data of all molecule from memory is slow, so we only do it once
        and pass it the the function to reuse it for every new molecule to be compared
output:
  void
 **/


void interactive4User (vector <info> allValue){
	//check whether to continue
	bool con;
	con = true;
	string mol2Compare, file2Compare;
	int keyIn, numberOfTop;


	while (con) {
		//Ask for the location of the mol file to be compared
		printf ("Please enter the location and file's name of the molecule to be compared.");
		printf("For example (C:/TestFiles/CHEMBL403741.mol)\n" );
		cin >> mol2Compare;
		ifstream myfile (mol2Compare.c_str());

		//Check if the enter file exists
		if (myfile) {

			file2Compare = parsedFile("", mol2Compare, true);
			printf ("How many top molecules would you like to see? \n");
			cin >> numberOfTop;
			printf("Comparsion in progress. Please wait... \n");
			//Compare the provided file name with the whole database.
			atomMappingGPU(file2Compare,allValue, numberOfTop, "../Results/atomMappingGPU-Data.txt","../Results/atomMappingGPU-Time.txt");
		}
		else {
			printf("The name provided does not exist\n");
		}
		printf ("The result is ready to view at: ../Results/atomMappingGPU-Data.txt \n"); 
		//Check whether the user wants to do another search
		printf( "Would you like to try again? (Press 0 to exit)\n");
		// the input has to be an int
		while(!(cin >> keyIn)){
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			cout << "Invalid input.  Please enter an integer: ";
		}

		//if the user does not want to continue, exit the program.
		if (keyIn ==0)   con = false;
	}
	//Terminate the program
	printf("Good Bye!\n");
}


/**
This is a simple function to compare 1 moleclue with all other molecules in the
database. The data result of the attom mapping algorithm will be written into a
file name "resultFileName". The total time taken for the whole matching will be
recorded in the "timeFileNam" file.

input:
  string mol2Compare: the name of parsed file of molecule A
  vector <info> allValue: containing all data of all molecules in the datase.
        Reading data of all molecule from memory is slow, so we only do it once
        and pass it the the function to reuse it for every new molecule to be compared
  int top: number of top molecule the user wants to have
  string resultFileName: name of file containing data result
  string timeFileName:  name of file containing time result for the whole comparision

output: void
 **/


void atomMappingGPU(string mol2Compare, vector <info> allValue, int top,
		string resultFileName,  string timeFileName ) {

	//Start timer
	clock_t begin, end;
	//Start the clock
	begin = clock();


	//Get information about the molecule to be compared
	float * A = getNumber2Array(mol2Compare);
	const int NA = getAtomNumber(mol2Compare);
	string * stringA = getNameList (mol2Compare);
	printf("done getting mol2Compare.info \n");

	//Two vectors to store the name and similarity score of all molecules in the database
	vector <string> Name;
	vector <float> similarityScore;

	//Initially, vectorSize is 0. At the end of the function, vectorSize is the
	//number of all molecules in the database
	int vectorSize =0;

	for (int i =0; i<allValue.size(); i++) {
		//printf("before getting information for each value of i = %d \n", i);
		int NB2 = allValue[i].atomNo;
		int NMax2 = allValue[i].moleculeNo;
		string * Name2 = allValue[i].name;
		float * B2 = allValue[i].bondMatrix;
		//printf("done with getting information for each value of i = %d \n", i);
		
		//Compare atom matrix. The time will be recorded in "compareMatrixGPU-Time.txt"
		//printf("before C2 = new float with NB2 = %d, NA = %d, NMax2 = %d \n", NB2, NA, NMax2);
		float *C2 = new float [NB2*NA*NMax2];
		//printf("after C2 = new float with NB2 = %d, NA = %d, NMax2 = %d \n", NB2, NA, NMax2);
		atomMatchingWithCuda(C2, A,B2,NA,NB2, NMax2,0.5f,
				"../Results/compareMatrixGPU-Time.txt");
		cudaDeviceReset();
		//printf("done with atomMatchingWithCuda of i = %d \n", i);

		//calculate similarity score.The time will be recorded in "calculateSimilarityGPU-Time.txt"
		float* CScan2 = new float[NMax2];
		calculateSimilarityWithCuda(CScan2, C2,NA, NB2, NMax2,
				"../Results/calculateSimilarityGPU-Time.txt");
				cudaDeviceReset();
		//printf("done with calculateSimilarity of i = %d \n", i);

		//Copy name and similirityScore to main vectors of Name and similarityScore
		Name.insert(Name.begin() + vectorSize, Name2, Name2+NMax2);
		similarityScore.insert(similarityScore.begin() + vectorSize, CScan2, CScan2+NMax2);
		vectorSize+= NMax2;
	}


	//Sort and print out top 'top' elemens
	sort(&similarityScore[0], &Name[0], vectorSize, top,
			resultFileName,  "../Results/sortGPU-Time.txt" , stringA[0]);
			cudaDeviceReset();
	printf("done with sort \n");
	//end clock
	end = clock();
	//Get elapsed time
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	//print to a specified file
	writeResult2File (vectorSize , NA, top, elapsed_secs, "seconds", timeFileName);

}
