#include "commonFunction.h"


#ifndef READFROMFILE4BINARY_H
#define READFROMFILE4BINARY_H


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


#endif
