#ifndef PARSEDFILE_H
#define PARSEDFILE_H

/**
Write to 2 txt file the name and calculated result of bond matrix
 **/
string write2File (int numberOfAtoms,string nameAndID, vector <float> output,
		string fileLocation);
/**Parse a single file**/
string parsedFile(string fileLocation, string fiName, bool file2Compare);
/**Parse all file in the given directory**/
bool parsedAllFile(string fileLocation);

#endif
