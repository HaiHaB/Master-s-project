#include <string>
#include <vector>
using namespace std;

#ifndef COMMONFUNCTION_H
#define COMMONFUNCTION_H

//define a struct containing all information of a fileData
struct info
{
  int atomNo;
  int moleculeNo;
  string* name;
  float* bondMatrix;
};


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




#endif
