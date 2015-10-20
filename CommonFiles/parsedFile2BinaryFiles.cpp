#include<iostream>
#include <fstream>      // std::ofstream
#include <sstream>      // std::istringstream
#include <vector>       // std:: vector
#include<cmath>
#include <iterator>     // std::ostream_iterator
#include <dirent.h>
#include "fileCollection.h"
#include "commonFunction.h"
#include "parsedFile2BinaryFiles.h"
using namespace std;

/**
Parse one file and write it the a txt and binary file
bool file2Compare: signal whether this is a molecule to be compared
 **/
string parsedFileBinary(string fileLocation, string fiName, bool file2Compare);
/**  Parse all file in the directory**/
bool parsedAllFileBinary(string fileLocation);
/**  Write a txt file the name and a binary file the calculated result of bond matrix  **/
string write2File2Binary (int numberOfAtoms,string nameAndID, vector <float> output,
		string fileLocation, bool file2Compare);

/**
  int main () {

    string fileLocation ="C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\Test2\\";//"C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\CPU-implementation\\TestFiles\\";
    string fromLocation = "C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\Test\\";//"C:\\Users\\Hai Ha\\matlab_chemistry\\";
    //moveAllFileContaining(".mol", "(",fromLocation, fileLocation);
  //  string fiName = "CHEMBL15939.mol";

  //int x = add(3,4);
  bool  result = parsedAllFile("C:\\Users\\Hai Ha\\Desktop\\DataTesting\\");//"C:\\Users\\Hai Ha\\Desktop\\TobeProcess\\");
  //"C:\\Users\\Hai Ha\\Desktop\\DataTesting\\");
    return 0;
  }


 **/

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
      Parse all file in the directory
 **/
bool parsedAllFileBinary(string fileLocation) {

	//delete all txt from in the fileLocation
	deleteAllFileWithName (fileLocation, "Atoms");

	vector <string> allName =  getNameFromDirectory (fileLocation);

	string temp;
	int vectorSize = allName.size();
	for (int i =0; i <vectorSize; i++) {
		temp = allName[i];
		if (temp.find(".mol")!=std::string::npos ){
			//if the file is not empty, parse the file
			bool result = isEmptyFile((fileLocation+temp).c_str()) ;
			if(!result) {
				//printf("start temp is %s\n", temp.c_str());
				// copy the to the specified file direction

				parsedFileBinary(fileLocation,temp,false);

			}
			else {
				printf("%d is empty\n", temp.c_str());
			}
		}

	}
return true;
}


/**
    Parse one file and write it the a txt and binary file
    bool file2Compare: signal whether this is a molecule to be compared
 **/
string parsedFileBinary(string fileLocation, string fiName, bool file2Compare) {
	string nameAndID, temp, nAtomsAndBonds;
	int numberOfAtoms, numberOfBonds;
	ifstream infile;

	infile.open ((fileLocation+fiName).c_str());


	//Get nameAndID  and Formula
	getline(infile,nameAndID);
	//cout<<nameAndID << endl;

	//Get randNumber and copyRightNotice
	getline(infile,temp);
	getline(infile,temp);

	//Get number of bond and atoms
	getline(infile,nAtomsAndBonds);
	//cout<<nAtomsAndBonds << endl;

	istringstream iss(nAtomsAndBonds);
	iss >> numberOfAtoms;
	iss >> numberOfBonds;

	//Stop if either numberOfAtoms or numberOfBonds is 0
	if (numberOfAtoms ==0 | numberOfBonds ==0) {
		printf(fiName.c_str(), "\n");
		return "Error here";
	}

	vector <float> coords (numberOfAtoms*3, 0);
	for (int i = 0; i<numberOfAtoms; i++) {
		getline(infile,temp);
		istringstream iss1(temp);
		for (int j =0; j <3 ; j++) 			iss1 >> coords[i*3+j];
	}

	vector <float> bonds (numberOfBonds*3, 0);
	for (int i = 0; i<numberOfBonds; i++) {
		getline(infile,temp);
		istringstream iss2(temp);
		for (int j =0; j <3 ; j++) 			iss2 >> bonds[i*3+j];
	}

	float result;
	//For each bond
	for (int k = 0; k < numberOfBonds; k++) {
		result = 0;
		// euclidean distance = sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
		for (int m = 0; m <3; m++) {
			int x1 = bonds[k*3]-1; // First element
			int x2 = bonds[k*3+ 1]-1;	//second element
			result += pow( (coords[x1*3 + m] - coords[x2*3 +m]),2);
		}
		bonds[k*3 +2] = sqrt(result);
	}

	vector< float> output (numberOfAtoms*numberOfAtoms, 0);
	for (int n=0; n < numberOfBonds; n++) {
		int x = (int) bonds[n*3] -1;
		int y = (int) bonds[n*3+ 1] -1;
		float distance = bonds[n*3+ 2];
		output[x*numberOfAtoms + y] = distance;
		output[y*numberOfAtoms + x] = distance;
	}

	//write to a binaru file
	return  write2File2Binary (numberOfAtoms,nameAndID, output,fileLocation,file2Compare);


	//free memory for vector
	vector<float>().swap(coords);
	vector<float>().swap(bonds);
	vector<float>().swap(output);

}
