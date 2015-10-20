#include<iostream>
//#include <windows.h>
#include <string>
#include <sstream>
#include <cstdio>
#include <fstream>
//#include <Windows.h>    // for Sleep();
#include <vector>
#include "fileCollection.h"
#include "commonFunction.h"
using namespace std;


//Download a specified number of files from ChemBL website to a specified folder
bool downloadAllFile4ChemBL( string toLocation, string fromLocation,
  string link, int min,  int number);




    /**
    Download a number of files from a specified website (link) and move it from a
    temporary folder (default donwload folder) to another folder (toLocation)

    input:
      int Number as the total number of molecule to download
      string toLocation: path to the place to keep the file
      strin fromLocation: path of default download folder
      string link: the website you want to download from
      ex.  https://www.ebi.ac.uk/chembl/download_helper/getmol/
      int Number: the number of files to be downloaded

    output: void
    Note: Only applicable for ChemBL website
    **/
   bool downloadAllFile4ChemBL( string toLocation, string fromLocation,
     string link, int min,  int number) {

    string website;
    ostringstream s;

    //Download "number" of files: from 0 to number
    for (int i =min; i<number; i++) {

      //clear ostringstream s
      s.str("");
      s.clear();

      //concatenate two strings to get the file download link
      s << link.c_str()<< i;

      //Open links to download files
      ShellExecute(NULL, "open", s.str().c_str(),NULL, NULL, SW_SHOWNORMAL);
      if (i%100 ==0) {

      //delete all dulicate file in the default download position
      deleteAllFileWithName (toLocation, "(");

      //Move all data containing CHEM substring but not 'crdownload...' (download-in-progress)
      moveAllFileContaining("CHEM", "crd","C:\\Users\\Hai Ha\\Downloads\\", toLocation);//"C:\\Users\\Hai Ha\\Project Daata\\New file\\");
      //Let the system break
     // Sleep(500);

      }
      //Wait for sometimes for the download to be completed
    //  Sleep(400);
    }

    return true;
  }



/**
// from 0 to 150000
  int main()
  {
    string toLocation = "C:\\Users\\Hai Ha\\matlab_chemistry\\";
    string test = "C:\\Users\\Hai Ha\\Desktop\\HH_SVN_BHAM\\CPU-implementation\\";
  //  getNameFromDirectory (test);
  //  deleteAllFileWithName (test, "Copy");
    string fromLocation = "C:\\Users\\Hai Ha\\Downloads\\";
    string link = "https://www.ebi.ac.uk/chembl/download_helper/getmol/";
  //  downloadNMove (toLocation, fromLocation, link,149000, 150000 );
  //  downloadNMove (toLocation, fromLocation, link,182388, 1500000 );
    downloadAllFile4ChemBL(toLocation,fromLocation, link, 182399,  1500000);
  }
**/
