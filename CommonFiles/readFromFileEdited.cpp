#include <string>
#include <stdio.h>
#include<iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <dirent.h>
#include "commonFunction.h"
#include "readFromFile.h"
#include <time.h>
using namespace std;


int getAtomNumber (string Name);
int getNumberOfLines (string fileName);
string * getNameList (string fileNameNLocation);
float * getNumber2Array(string fileNameLocation);
vector <info> readAllValue (string allLocation);
info getFileData(string fileName) ;
/**

int main () {

string ex2 = "C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\Test2\\12Atoms.txt";
//C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\CPU-implementation\\TestFiles\\8Atoms.txt";
//int atomNumber = getAtomNumber(ex2) ;
//int numberOfLines = getNumberOfLines(ex2);

//string* result = getNameList(ex2);
float * result2 = getNumber2Array(ex2);
// readAllValue("C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\Test2\\");
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
    if (temp.find("Atoms.txt")!=std::string::npos) {
      //get the path to open a file
      location = allLocation+temp;
/**
      //Get the information from each text file
      int atomNo = getAtomNumber(location) ;
      int moleculeNo = getNumberOfLines(location)/2;
      string * nameList = getNameList(location);
      float * bondMaxtrix = getNumber2Array(location);

       A1 = {atomNo, moleculeNo, nameList, bondMaxtrix};
**/
      A1 = getFileData(location);
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
**/

info getFileData(string fileName) {

    info A1;
    int atomNo = getAtomNumber(fileName);
    int moleculeNo = getNumberOfLines(fileName)/2;
    string *nameList =  new string[moleculeNo];
    float *bondMatrix = new float [atomNo*atomNo*moleculeNo];

    //Open the specified file
    ifstream myfile (fileName.c_str());
    string temp;

    //    printf("working on %s\n", fileName.c_str());
    //for every molecule, there are two lines
    for (int i=0; i<moleculeNo; i++) {
      //first line contans name, label and fomular. Write to nameList[i]
      getline(myfile, nameList[i]);

      //printf("at molecule number %d \n", i);
      //second line contans atomNumber^2 atoms. Get the atoms into the array
      getline(myfile, temp);
      istringstream Number(temp);

      // get all atomNumber^2 atoms into result array
      for (int t =0; t< atomNo*atomNo; t ++) {
        //      printf("at t= %d \n", t);
        Number >> bondMatrix[i*atomNo*atomNo+t];
        }
   }
      A1 = {atomNo, moleculeNo, nameList, bondMatrix};
      printf("done with %s containing %d molecules of %d atoms \n", fileName.c_str(), moleculeNo, atomNo);


   return A1;

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
string * getNameList (string fileNameNLocation) {
  //Get number of line in the txt file
  int numberOfLines = getNumberOfLines(fileNameNLocation);
  string * result = new string[numberOfLines/2];

  ifstream myfile (fileNameNLocation.c_str());
  string temp;

  //for each molecule, there are two lines
  for (int i=0; i<numberOfLines/2; i++) {
    //first line contans name, label and fomular. Get that line into the result file
    getline(myfile, result[i]);
    //second line contans atomNumber^2 atoms. Do nothing with it
  //  getline(myfile, temp);
    cout << i << " line is: " << result[i] << endl;

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
