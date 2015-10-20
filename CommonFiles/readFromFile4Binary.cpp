#include <string>
#include <stdio.h>
#include<iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "commonFunction.h"
#include "readFromFile.h"
#include <time.h>
#include <iostream>
#include <cstdio>
#include <stdlib.h>
using namespace std;

/* Get atom number from a name. i.e. 18AtomsName.txt -> 18 atoms*/
int getAtomNumber (string Name);

/**Get the number of line in a text file*/
int getNumberOfLines (string fileName);

/*Get the list of names of molecules from a text file into a array of string*/
string * getNameList (string fileNameNLocation, int moleculeNo);

/*Get the float from a specified file (fileNameLocation) to a array of float*/
float * getNumber2Array(string fileNameLocation);

/*Read values from all text file with name containing "Atoms.txt" of a location*/
vector <info> readAllValue (string allLocation);

/*Get all information including AtomNumber, moleculeNo, nameList and float
 data from a file
 */
info getFileData(string fileName, string folderName);

/* Read information from a binary file*/
float* readFromBinaryFile (string fileName, long &totalSize);


/**
int main () {

string ex2 = "C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\Test2\\12Atoms.txt";
//C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\CPU-implementation\\TestFiles\\8Atoms.txt";
//int atomNumber = getAtomNumber(ex2) ;
//int numberOfLines = getNumberOfLines(ex2);

//string* result = getNameList(ex2);
//float * result2 = getNumber2Array(ex2);
 readAllValue("C:\\Users\\Hai Ha\\Desktop\\DataTesting\\");
}

 **/

/**
  Read values from all text file with name containing "Atoms.txt" of a location

  input:
    string: folder path (allLocation)

  output:
    vector of info: containing all infomation in txt file with "Atoms.txt" in in the directory
 **/
vector <info> readAllValue (string allLocation) {
	//Start timer
	clock_t begin, end;
	//Start the clock
	begin = clock();


	vector <info > result;
	info A1;
	string temp, location;
	//get the list of File Name in the directory
	vector<string> nameList = getNameFromDirectory (allLocation);

	//for all files in the directory
	for (int i =0; i <nameList.size(); i++) {
		temp = nameList[i];

		//If the File Name contain "Atomstxt"
		if (temp.find("AtomsName.txt")!=std::string::npos) {
				printf("Name is %s\n", temp);

			//get the path to open a file
			location = allLocation + temp;
			A1 = getFileData(location, allLocation);
			//store the infomation into the result vector
			result.push_back (A1);
		}
	}

	//end clock
	end = clock();
	//Get elapsed time
	double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;
	//print to a specified file
	writeResult2File (result.size(), elapsed_secs, "seconds", "readFromFile-Time.txt");
	return result;
}

/**
Get AtomNumber, NuberOFline, NameList and Array2Number from a location
this fileName should be the name of text file comtaining all Names
 **/

info getFileData(string fileName, string folderName) {

	info A1;
	int atomNo = getAtomNumber(fileName);
	//get Name of data file name.
	string dataFileName = folderName + convertNumber2String(atomNo)+"AtomsData.bin";

	//Read from binary size into bondMatrix and long totalSize
	long totalSize;
	float *bondMatrix = readFromBinaryFile (dataFileName, totalSize);
//		for (int i =0; i <totalSize/sizeof(float); i++) printf("%.4f ", bondMatrix[i]);

	//molecule number is the total size of bondMatrix divide by (atomNo*atomNO);
	int moleculeNo =totalSize/(atomNo*atomNo);
	//Get the name list from name list files
	string *nameList =  getNameList (fileName, moleculeNo);
	for (int i =0; i <moleculeNo; i++) printf("%s ", nameList[i]);
	printf("\n");

//	A1 = {atomNo, moleculeNo, nameList, bondMatrix};
	A1.atomNo = atomNo;
	A1.moleculeNo = moleculeNo;
	A1.name = nameList;
	A1.bondMatrix = bondMatrix;
//	printf("done with %s containing %d molecules of %d atoms \n", fileName.c_str(), moleculeNo, atomNo);


	return A1;

}

/**
read from a binary file to an array
input: fileName: Name of the interested file
        totalSize: number of Data in the the binary file
 **/
float* readFromBinaryFile (string fileName, long &totalSize) {

	FILE * pFile;
	float * buffer;
	size_t result;

	//Open fileName
	pFile = fopen ( fileName.c_str() , "rb" );
	if (pFile==NULL) {fputs ("File error",stderr); exit (1);}

	// obtain file size:
	fseek (pFile , 0 , SEEK_END);
	totalSize = ftell (pFile);
	totalSize /=(long)sizeof(float);
	rewind (pFile);

	// allocate memory to contain the whole file:
	buffer = (float*) malloc (sizeof(float)*totalSize);
	if (buffer == NULL) {fputs ("Memory error",stderr); exit (2);}

	// copy the file into the buffer:
	result = fread (buffer,sizeof(float),totalSize,pFile);
	//if (result != lSize) {fputs ("Reading error",stderr); exit (3);}

	  for (int i=0; i<totalSize; i++) printf("%.2f ", buffer[i]);


	// terminate and free memory
	fclose (pFile);
	//free (buffer);
return buffer;
}

/**
  Get the float from a specified file (fileNameLocation) to a array of float

  input:
    string: folder path and file name (fileNameNLocation)

  output:
    array of float: containing all the numbers
 **/

float * getNumber2Array(string fileNameNLocation) {
	//Get appropriate infomation about the specified file
	int atomNumber = getAtomNumber(fileNameNLocation) ;
	int numberOfLines = getNumberOfLines(fileNameNLocation);
	float * result = new float [atomNumber*atomNumber*numberOfLines/2];


	//Open the specified file
	ifstream myfile (fileNameNLocation.c_str());
	string temp;

	//for every molecule, there are two lines
	for (int i=0; i<numberOfLines/2; i++) {
		//first line contans name, label and fomular. Do nthing with it.
		getline(myfile, temp);

		//second line contans atomNumber^2 atoms. Get the atoms into the array
		getline(myfile, temp);
		istringstream Number(temp);

		// get all atomNumber^2 atoms into result array
		for (int t =0; t< atomNumber*atomNumber; t ++) {
			Number >> result[i*atomNumber*atomNumber+t];
		}

	}
	return result;
}

/**
  Get the list of names of molecules from a text file into a array of string

  input:
    string: folder path and file name (fileNameNLocation)

  output:
    array of string: containing all the chemical names (name, fomular, label)
 **/
string * getNameList (string fileNameNLocation, int moleculeNo) {
	string * result = new string[moleculeNo];

	ifstream myfile (fileNameNLocation.c_str());
	string temp;

	printf("In name List of %s including %d\n", fileNameNLocation.c_str(), moleculeNo );
	//for each molecule, there are two lines
	for (int i=0; i<moleculeNo; i++) {
		//first line contans name, label and fomular. Get that line into the result file
		getline(myfile, result[i]);
		printf("%s ", result[i]);
	}

	myfile.close();
	return result;

}



/**
  Get the number of atom for a text file

  input:
    string: file name (Name)

  output:
    int: number of atoms
 **/
int getAtomNumber (string Name) {
	int result;

	// example of Name is "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation\12Atoms.txt"
	if (Name.find("Atoms")) {
		// get the position of the last '\'
		result= Name.find_last_of('\\');

		//Get the substring from the last '\' get 12Atoms.txt
		Name = Name.substr(result+1,Name.size()-result);

		//Get the number from the substring, example result = 12 from '12Atoms.txt'
		sscanf(Name.c_str(), "%d[a-z|.]*", &result);
	}
	else {
		printf ("The correct file name should contain \"Atoms\" \n ");
	}
	return result;
}



/**
  Get the number of line in a text file

  input:
    string: file name (Name)

  output:
    int: number of line
 **/
int getNumberOfLines (string fileName) {
	ifstream myfile (fileName.c_str());
	string line;
	int result =0;


	//Keep counting until the end of file
	while (getline(myfile, line)) {
		++result;
	}
	return result;
}
