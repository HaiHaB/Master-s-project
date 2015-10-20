#include <cmath>
#include <vector>
#include <string>
#include <iomanip>  // for setw()
#include <fstream>
#include <algorithm>
#include <time.h>
#include "../CommonFiles/commonFunction.h"
using namespace std;


bool compareKey (const pair<float, int>& lhs, const pair<float, int>& rhs) {
    return lhs.first > rhs.first; 
} 

/**
@param:   array of float containing the simmilarity score
          array of string containing all the string
          string fileNameNLocation
          write the result of the sorted string to an file
@ result  void
**/
void sort(float* keys, string* values, int size, int top, string outputFileName,
    string outputTime, string molecule2Compare) {
    clock_t begin, end;
    //Start the clock
    begin = clock();

    //Create a vector pair of key and position to sort_by_key
   vector <pair <float, int> > keyNPosition;
   for (int i = 0; i< size; i++) {
     keyNPosition.push_back(make_pair(keys[i],i));
   }

   //Sort the vector by keys in descending order, while keeping track the iniitial position
   // Note lambda expression is used here, so remember to include "-std=c++11" when compiling.
   /**
   sort (keyNPosition.begin(), keyNPosition.begin()+size, [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
             return lhs.first > rhs.first; } );
**/

   sort (keyNPosition.begin(), keyNPosition.begin()+size, compareKey);
   int *position = new int [size];
   //Get sorted keys and its position of from the sorted keyNPostion vector
   for (int i = 0; i< size; i++) {
     keys[i] = keyNPosition[i].first;
     position[i] =  keyNPosition[i].second;
   }


  //end clock
  end = clock();
  //Get elapsed time
  double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;

  //print the elapsedTime to a specified file "outputTime"
  //writeResult2File (size, elapsed_secs, "seconds", outputTime);

  //print the result of sorting into a outputFileName
  writeTopSimilarityScore2File (keys, position, values, size, top,outputFileName,
     molecule2Compare);

}



/**
int main() {
    const int N = 6;
    float keys[N] = { 1.9, 4.4, 2.6, 8.1, 5.8, 7.3};
    int  position[N] = {0,1,2,3,4,5};
    string values[N] = {"Hello", "there", "have", "a", "wonderful", "day"};
    string A = "hello";
    sort(keys, values, N, 6, "testing123.txt", "testing123Time.txt", A);

  //  testFunctionCPU (keys,values, N, 10, 100000, 1000000,100000, "testing123.txt",
    //        "testing123Time.txt",A, &sort);

return 0;
}
**/
