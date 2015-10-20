#include <algorithm>
#include <stdio.h>     // can use printf in kernel function
#include <string>
#include <fstream>
#include <iostream>
#include <time.h>
#include<cmath>
#include <iomanip>      // std::setw
#include "commonFunction.h"
#include <vector>
#include <sstream>      // std::ostringstre

#include <dirent.h>

using namespace std;

//To test function for atom matching for CPU implementation
void testFunctionCPU (float *A, float *B, int NA, int NB, int NMax, const int Min,
		const int Max, const int Incr,float threshold, string outputFileName,
		void (*atomMatching)(float*, const float*, const float*,
				const int, const int, const int, float, string));

//To test function for calculateSimilarity for CPU implementation
void testFunctionCPU (float *A, int NA, int NB, int NMax, const int Min,const int Max,
		const int Incr, string outputFileName,	void (*calculateSimilarity)(float*, float*,
				const int, const int, const int, string));

//To test for sort function in both CPU and GPU implementation
void testFunctionCPU (float *A, string *value, int size, int top, const int Min,
		const int Max,const int Incr, string outputFileName, string outPutTime,
		string molecule2Compare,void (*sort)(float*, string*, int, int,
				string, string, string));

//Copy a string array of size 'size' n times
string* copy2NewArray(string* A, int size, int times);

//Copy a float array of size 'size' n times
float* copy2NewArray(float* A, int size, int times);

//Write result of sorting to file
void writeResult2File (int N, float milliseconds, string unit, string outputFileName);

//Write result of atomMathing and calculateSimilarity to file
void writeResult2File (int NA, int NB, int NMax, float milliseconds, string unit,
		string outputFileName);

//Write similarity Socre to a file
void writeTopSimilarityScore2File (float* keys, int * position, string* values,
		int size, int top, string outputFileName, string molecule2Compare);

//Get Name of all file in a directory
vector <string> getNameFromDirectory (string directory);

//Copy a string from one location 'fileNameFrom' to another location 'fileNameTo'
static void copyFile(const std::string& fileNameFrom, const std::string& fileNameTo);

//Move all file if their names contain a 'subsringYes' but not 'subStringNo' from
//one location to another one
void moveAllFileContaining(string subStringYes, string subStringNo,
		string fromLocation, string toLocation);

//Check whether a file is empty
bool isEmptyFile(string fileLocation);

//Convert a string from a number
string convertNumber2String(int Number);

//Delete all files if their name contain a 'subStringYes'
void deleteAllFileWithName (string directory, string subStringYes);





/**Check whether a file is empty
		return true if the file is empty;
					false if the file is not empty
 **/
bool isEmptyFile(string fileLocation) {
	ifstream infile;

	infile.open ((fileLocation).c_str(), ios::binary);// open your file

	infile.seekg (0, infile.end);// put the "cursor" at the end of the file
	int length = infile.tellg(); // find the position of the cursor
	infile.close();

	if ( length == 0 )return true;
	else return false;

}



/**
  Delete all file that contains a substring "substringYes" in a folder (directory)

  input:
    string: folder path
    string: a substring that the file name should contain

  output: void
 **/
void deleteAllFileWithName (string directory, string subStringYes) {
	string temp = "";
	//get the list of File Name in the directory
	vector<string> nameList = getNameFromDirectory (directory);

	//for all files in the directory
	for (int i =0; i <nameList.size(); i++) {

		temp = nameList[i];
		//If the File has subStringYes,delete the file
		if (temp.find(subStringYes)!=std::string::npos)
			const int  result2 = remove((directory +temp).c_str());
	}
}

/**
  Move a file contain a substring (substringYes) and don't contai another substring
  (subStringNo) from one location (fromLocation) to another location (toLocation)

  input:
    string: a substring that the file name should contain
    string: a substring that the file name should not contain
    string: location for files to be copied from
    string: location for files to be copied to

  output: void
 **/
void moveAllFileContaining(string subStringYes, string subStringNo,
		string fromLocation, string toLocation) {

	// delele all file containing "subStringYes" in the specified location
	//    deleteAllFileWithName (toLocation, subStringYes);

	//get the list of File Name in the directory
	vector<string> nameList = getNameFromDirectory (fromLocation);

	string temp;
	//for all files in the directory
	for (int i =0; i <nameList.size(); i++) {
		temp = nameList[i];

		//If the File has subStringYes, and don't contain a subStringNo
		// We dont want to move any repeated files with contain .(1)
		if (temp.find(subStringYes.c_str())!=std::string::npos &&
				temp.find(subStringNo.c_str())==std::string::npos) {
			// copy the to the specified file direction
			copyFile(fromLocation+temp, toLocation+temp);
			//delete the files after copying over
			remove((fromLocation +temp).c_str());
		}

	}
}

/**
copy a file from one folder (fileNameFrom) to another folder (fileNameTo)

input:
	string: location and name of the file to be copied from
	string: location and name of the file to be copied to

output: void
 **/


static void copyFile(const std::string& fileNameFrom, const std::string& fileNameTo)
{
	//Open file in and out
	std::ifstream in (fileNameFrom.c_str());
	std::ofstream out (fileNameTo.c_str());

	//Copy from file in to file out
	out << in.rdbuf();

	//Close file in and out
	out.close();
	in.close();
}



/*Convert a string from a number*/
string convertNumber2String(int Number) {
	ostringstream convert;
	convert << Number;
	return convert.str();
}

/**
		Get the name of all files in a folder (directory)

		input:
			string: folder path (directory)

		output: vector of string containing all files in the directory
 **/

vector <string> getNameFromDirectory (string directory) {
	string temp = "";
	DIR *dir;
	struct dirent *ent;
	vector <string> result;

	if ((dir = opendir (directory.c_str())) != NULL) {

		while ((ent = readdir (dir)) != NULL) {
			temp = ent->d_name;
			result.push_back(temp);
		}
		closedir (dir);
	}
	else {
		// could not open directory
		perror ("");
		printf("Could not open directory");
	}

	return result;
}


/**
A simple function to test increasingly bigger arrays from a min number to a max
umber with Incre increment and write time elapsed in a file

input:
  array of float A: molecule A to be compared
  array of float b: molecules B to be compared
  const int NA: number of atoms in molecule A
	const int NB: number of atoms in each molecule in B
  const int NMax: number of molecules in B
  const int Min: start point of loop
  const int Max: end point of loop
	const int Incr: increment step of loop
  float threshold: acceptable difference for coordinates to be the same
  string outputFileName: file name to write result to
  function pointer: to test for CPU implementation of compare matrix

output:
  void
 **/
void testFunctionCPU (float *A, int NA, int NB, int NMax, const int Min,const int Max,
		const int Incr, string outputFileName,	void (*calculateSimilarity)(float*, float*,
				const int, const int, const int, string)) {

	//	printf("before for loop Entering smilarity calculation\n");
	for (int q = Min; q < (Max/NMax); q +=Incr) {

		//Create a q times bigger array
		float* Aq = copy2NewArray(A, NA*NB*NMax, q);
		float* Cq = new float [NMax*q];
		//		printf("before Entering similarity calculation\n");
		calculateSimilarity(Cq, Aq,NA,NB, NMax*q,outputFileName);
		/**
		//print out result for correctness checking
		printf ("q is %d\t", q);
		for (int j =0; j<NMax; j++) {
			printf("C[0] is %.2f ", Cq[NMax*q-1]);
		}
		printf("\n");
		 **/
		delete[] Aq;
		delete[] Cq;


	}

}


/**
A simple function to test increasingly bigger arrays from a min number to a max
number with Incre increment and write time elapsed in a file

input:
  array of float A: molecule A to be compared
  array of float b: molecules B to be compared
  const int NA: number of atoms in molecule A
  const int NB: number of atoms in each molecule in B
  const int NMax: number of molecules in B
  const int Min: start point of loop
  const int Max: end point of loop
  const int Incr: increment step of loop
  float threshold: acceptable difference for coordinates to be the same
  string outputFileName: file name to write result to
  function pointer: to test for CPU implementation of compare matrix

output:
  void
 **/
void testFunctionCPU (float *A, float *B, int NA, int NB, int NMax, const int Min,
		const int Max, const int Incr,float threshold, string outputFileName,
		void (*compareMatrixParallelRow)(float*, const float*, const float*,
				const int, const int, const int, float, string)) {

	for (int q = Min; q < (Max/NMax); q +=Incr) {

		//Create a q times bigger array
		float* B20 = copy2NewArray(B, NB*NB*NMax, q);
		float *C20 = new float[NB*NA*NMax*q];
		compareMatrixParallelRow(C20, A,B20,NA,NB, NMax*q,0.5f, outputFileName);
		/**
		//print out result for correctness checking
		if (q <Max) {
			printf ("q is %d\t", q);
			for (int j =0; j<NB*NA; j++) {
				printf("%.2f ", C20[NB*NA*(q-1)+j]);
			}
			printf("\n");
		}
		 **/
	}
}



/**
A simple function to test increasingly bigger arrays from a min number to a max
number with Incre increment and write time elapsed in a file

input:
  array of float A: molecule A to be compared
  const int NA: number of atoms in molecule A
  const int NB: number of atoms in each molecule in B
  const int NMax: number of molecules in B
  const int Min: start point of loop
  const int Max: end point of loop
  const int Incr: increment step of loop
  string outputFileName: file name to write result to
  function pointer: to test for CPU implementation of compare matrix

output:
  void
 **/
void testFunctionCPU (float *A, string *value, int size, int top, const int Min,
		const int Max,const int Incr, string outputFileName, string outPutTime,
		string molecule2Compare,void (*sort)(float*, string*, int, int,
				string, string, string)) {

	for (int q = Min; q < Max; q +=Incr) {
		//Create a q times bigger array
		float* Aq = copy2NewArray(A,size, q);
		string* valueQ = copy2NewArray(value, size, q);

		sort( Aq, valueQ, size*q, top, outputFileName, outPutTime, molecule2Compare);

		//Free Memory
		delete[] Aq;
		delete[] valueQ;
	}
}



/**
Make a 'q' times bigger from an initial array of size 'size' of float array
 **/
float* copy2NewArray(float* A, int size, int times) {
	float *B = new float [size*times];
	for (int i =0; i<times; i++) {
		for (int j=0; j<size; j++) {
			B[i*size +j] = A[j];
		}
	}
	return B;
}




/**
Make a 'q' times bigger from an initial array of size 'size' of string array
 **/
string* copy2NewArray(string* A, int size, int times) {
	string *B = new string [size*times];
	for (int i =0; i<times; i++) {
		for (int j=0; j<size; j++) {
			B[i*size +j] = A[j];
		}
	}
	return B;
}


/**
Write information of NA, NB, NMax and elapsed time to a specified outputFileName
Could be use for AtomMatching and calculateSimilarity

 **/
void writeResult2File (int NA, int NB, int NMax, float milliseconds, string unit,
		string outputFileName) {
	ofstream myfile;
	myfile.open(outputFileName.c_str(), ios:: app);
	myfile << setw(5) << NA << setw(5)<<"," <<NB << setw(10) <<","
	<<NMax <<setw(15)<< ","<<milliseconds<< right <<setw(20)<<"," << unit <<"\n";
	myfile.close();
}


/**
Write information of size of array (N) and  elapsed time to a specified outputFileName
Could be use for Sort
 **/
void writeResult2File (int N, float milliseconds, string unit, string outputFileName) {
	ofstream myfile;
	myfile.open(outputFileName.c_str(), ios:: app);
	myfile << setw(5) << N <<"," <<setw(15) << milliseconds<<"," << right <<setw(20)
	<< unit <<"\n";
	myfile.close();
}


/**
  Write top 'top' similarity result into a specified file
 **/

void writeTopSimilarityScore2File (float* keys, int * position, string* values,
		int size, int top, string outputFileName, string molecule2Compare) {

	ofstream myfile;
	// Append new content into file
	myfile.open(outputFileName.c_str(), ios::app);

	//Write in titile of the comparision
	myfile<<"The top "<< top <<" results for " << molecule2Compare <<endl;
	myfile << setw(10) << left <<"Number" << setw(20) << "Simmilarity Score"
	<< setw(20) << "Fomula" << setw(20) << "Label"
	<< setw(20) << "Name" << endl ;
	myfile << "----------------------------------------------------------------------------------" <<endl;

	// Get the mininum of the 'size' of array or the 'top' Number
	// if there is less molecules than the required 'top', print out everything,
	// else, only print out top number of molecules
	int min = size>top? top: size;

	for (int i =0; i<min; i++) {
		myfile << setw(10) << i+1 << setw(20) << keys[i] << values[position[i]] <<endl;
	}
	//Close file
	myfile.close();

}
