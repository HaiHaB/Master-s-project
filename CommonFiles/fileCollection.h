#include "commonFunction.h"

#ifndef FILECOLLECTION_H
#define FILECOLLECTION_H

//Download a specified number of files from ChemBL website to a specified folder
bool downloadAllFile4ChemBL( string toLocation, string fromLocation,
  string link, int min,  int number);

#endif
