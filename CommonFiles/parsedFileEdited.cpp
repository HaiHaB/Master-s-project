#include<iostream>
#include <fstream>      // std::ofstream
#include <sstream>      // std::istringstream
#include <vector>       // std:: vector
#include<cmath>
#include <iterator>     // std::ostream_iterator
#include <dirent.h>
#include "fileCollection.h"
#include "commonFunction.h"
#include "parsedFile.h"
#include <deque>
using namespace std;

//  string write2File (int numberOfAtoms,string nameAndID, vector <float> output,
//  string fileLocation);
string write2File (int numberOfAtoms,string nameAndID, deque <float> output,
		string fileLocation);
string convertNumber2String(int Number);
string parsedFile(string fileLocation, string fiName, bool file2Compare);
bool parsedAllFile(string fileLocation);
bool isEmptyFile(string fileName);
void check4Success( bool success, string error123);


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



string convertNumber2String(int Number) {
	ostringstream convert;
	convert << Number;
	return convert.str();
}



string write2File (int numberOfAtoms,string nameAndID, deque <float> output,
		string fileLocation, bool file2Compare) {

	ofstream myfile;
	string FileName;

	if (file2Compare) {
		FileName = fileLocation +  convertNumber2String(numberOfAtoms) +"AtomsFile2Compare.txt";
		myfile.open(FileName.c_str());
	}
	else {
		FileName = fileLocation + convertNumber2String(numberOfAtoms) + "Atoms.txt";
		myfile.open(FileName.c_str(), ios_base:: app);
	}
	// Write the tile and labels
	myfile  << nameAndID << endl;

	copy(output.begin(), output.end(), ostream_iterator<float>(myfile , " "));
	myfile << endl;
	myfile.close();

	return FileName;
}




bool parsedAllFile(string fileLocation) {

	//delete all txt from in the fileLocation
	deleteAllFileWithName (fileLocation, "Atoms.txt");

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
				printf("doing temp is %s\n", temp.c_str());
				parsedFile(fileLocation,temp,false);

			}
			else {
				//printf("%d is empty\n", temp.c_str());
			}
		}

	}

}



string parsedFile(string fileLocation, string fiName, bool file2Compare) {
	string nameAndID, temp, nAtomsAndBonds;
	int numberOfAtoms, numberOfBonds;
	ifstream infile;
	bool success;

	infile.open ((fileLocation+fiName).c_str());

	//  printf("Opening files");
	//Get nameAndID  and Formula
	success = getline(infile,nameAndID);
	check4Success(success, "Error at getting nameAndID");
	//cout<<nameAndID << endl;

	//Get randNumber and copyRightNotice
	success =getline(infile,temp);
	check4Success(success, "Error at getting randNumber");
	success =getline(infile,temp);
	check4Success(success, "Error at getting copyRightNotice");

	//Get number of bond and atoms
	success =getline(infile,nAtomsAndBonds);
	check4Success(success, "Error at getting nAtomsAndBonds");
	//cout<<nAtomsAndBonds << endl;

	//  printf("getnAtomAndBonds\n");
	istringstream iss(nAtomsAndBonds);
	iss >> numberOfAtoms;
	//if numberOfAtoms is greater than 100, chances that numberOfAtoms and numberOfBonds will stick together

	iss >> numberOfBonds;
	//Stop if either numberOfAtoms or numberOfBonds is 0
	if (numberOfAtoms ==0 | numberOfBonds ==0) {
		printf(fiName.c_str(), "\n");
		return "Error here";
	}

	//  vector <float> coords (numberOfAtoms*3, 0);
	deque <float> coords (numberOfAtoms*3, 0);
	for (int i = 0; i<numberOfAtoms; i++) {
		success =getline(infile,temp);
		check4Success(success, "Error in coords");
		istringstream iss1(temp);
		for (int j =0; j <3 ; j++) 			iss1 >> coords[i*3+j];
	}

	//    printf("bonds matrix\n");
	deque <float> bonds (numberOfBonds*3, 0);
	//vector <float> bonds (numberOfBonds*3, 0);
	for (int i = 0; i<numberOfBonds; i++) {
		success =getline(infile,temp);
		check4Success(success, "Error in bonds");
		istringstream iss2(temp);
		for (int j =0; j <3 ; j++)
		{		iss2 >> bonds[i*3+j];
		//    printf("%2.f ",bonds[i*3+j]);
		}
		//    printf("\n");
	}

	//    printf("calculating\n");
	float result;
	//For each bond
	for (int k = 0; k < numberOfBonds; k++) {
		result = 0;
		// euclidean distance = sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
		for (int m = 0; m <3; m++) {
			int x1 = bonds[k*3]-1; // First element
			int x2 = bonds[k*3+ 1]-1;	//second element

			//      printf("before pow, k is %d, m is %d, x1 is %.2f x2 is %.2f\n", k,m,x1,x2);
			//      printf("%.2f, %0.2f\n", coords[x1*3 + m],coords[x2*3 +m]);
			result += pow( (coords[x1*3 + m] - coords[x2*3 +m]),2);
			//      printf("result is %.2f\n", result);
		}
		//      printf("before sqrt\n");
		bonds[k*3 +2] = sqrt(result);
	}

	//    printf("output matrix\n");
	deque< float> output (numberOfAtoms*numberOfAtoms, 0);
	for (int n=0; n < numberOfBonds; n++) {
		int x = (int) bonds[n*3] -1;
		int y = (int) bonds[n*3+ 1] -1;
		float distance = bonds[n*3+ 2];
		output[x*numberOfAtoms + y] = distance;
		output[y*numberOfAtoms + x] = distance;
	}

	return  write2File (numberOfAtoms,nameAndID, output,fileLocation,file2Compare );


	//  return "Hello";
	Error:
		//free memory for vector
		deque<float>().swap(coords);
		deque<float>().swap(bonds);
		deque<float>().swap(output);

}

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
if is the a success, break and print out error
 **/
void check4Success( bool success, string error123) {
	if (!success) {

		printf(error123.c_str());
		printf("\n");
		//  goto Error;
	}

}

/** get the number of decimal number that an int has. Examplee 52 has 2 number
 **/
int getDecimalNumber (int A) {
	int count;
	while (A>0) {
		A = A/10;
		count++;
	}
	return count;
}

/**
Get n decimal place to m decimal place, starting from the begining (0)of a number
start has to be larger or equal than 0
end is less than length of a number
 **/
int getPartialNumber (int A, int start, int end) {
	//Get the total length of number
	int length = getDecimalNumber(A);

	//check for that start>=0 and end<= length-1
	if (start <0 | end >length-1) return 0;


}
