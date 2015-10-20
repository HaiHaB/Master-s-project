#include <string>
#include <stdio.h>
#include<iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <time.h>
#include <iostream>
#include <cstdio>
#include <stdlib.h>
using namespace std;
/**
  Write a txt file the name and a binary file the calculated result of bond matrix
 **/
string write2File2Binary (int numberOfAtoms,string nameAndID, vector <float> output,
		string fileLocation, bool file2Compare) {

	ofstream myfile;
	string FileName, dataName;
	//FILE* myffile;
	FILE* data;

	//If this file is a molecule to be compare, we need to name it differently
	if (file2Compare) {
		FileName = fileLocation +  convertNumber2String(numberOfAtoms) +"AtomsName2Compare.txt";
		dataName = fileLocation +  convertNumber2String(numberOfAtoms) +"AtomsData2Compare.bin";
		// Open text file 'myfile' and  binary file 'data' and delete everything before
		myfile.open (FileName.c_str());
		data = fopen (dataName.c_str(), "wb");
	}
	else {
		FileName = fileLocation + convertNumber2String(numberOfAtoms) + "AtomsName.txt";
		dataName = fileLocation +  convertNumber2String(numberOfAtoms) +"AtomsData.bin";
			// Open text file 'myfile' and  binary file 'data' and append it
			myfile.open (FileName.c_str(), ios_base:: app);
				data = fopen (dataName.c_str(), "ab");
	}


	// Write the tile and labels
	myfile  << nameAndID << endl;
	printf("output is:\n");
	// write whole vector<float> output of size numberOfAtoms*numberOfAtoms to binary files data
	for (int i = 0;  i<numberOfAtoms*numberOfAtoms; i++) {
			printf("%.3f ", output[i]);
		fwrite(&output, sizeof(float), 1, data);
	}

	//close both files
	myfile.close();
	fclose(data);

	return FileName;
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
