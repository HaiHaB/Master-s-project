#include <string>
#include <vector>
using namespace std;

#ifndef PARSEDFILE2BINARYFILES_H
#define PARSEDFILE2BINARYFILES_H

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

#endif
