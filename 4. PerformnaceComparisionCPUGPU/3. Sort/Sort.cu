#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <thrust/sort.h>
#include <vector>
#include <string>
#include <iomanip>  // for setw()
#include <fstream>
#include <algorithm>
#include <time.h>
#include "../CommonFiles/commonFunction.h"
#include "Sort.h"
using namespace std;


/**
@param:   array of float containing the simmilarity score
          array of int containing the position of string
          array of string containing all the string
          string fileNameNLocation
          write the result of the sorted string to an file
@ result  void
**/
void sort(float* keys, string* values, int size, int top,
  string outputFileName, string outPutTime, string molecule2Compare) {
    clock_t begin, end;
    //Start the clock
    begin = clock();

    //Initialised a position vector to use with sort_by_ky
    int *position = new int [size];
    //initialised position from 0 to size-1
    for (int i= 0; i<size; i++)    position[i] = i;

  thrust::sort_by_key(keys, keys + size, position, thrust::greater<float>());
  //end clock
  end = clock();
  //Get elapsed time
  double elapsed_secs = (double)(end - begin) / CLOCKS_PER_SEC;

  //print to a specified file
  writeResult2File (size, elapsed_secs, "seconds", outPutTime);


  writeTopSimilarityScore2File (keys, position, values, size, top,outputFileName,
     molecule2Compare);

     //free memory the end
     delete [] position;

}


/**
int main() {
    const int N = 6;
    float keys[N] = { 1.9, 4.4, 2.6, 8.1, 5.8, 7.3};
    int  position[N] = {0,1,2,3,4,5};
    string values[N] = {"Hello", "there", "have", "a", "wonderful", "day"};
    string A = "hello";
    sort(keys, values, N, 5, "testing123.txt", "testing123Time.txt", A);

    //testFunction(keys,values, N, 10, 100000, 1000000,100000, "testing123.txt",
      //      "testing123Time.txt",A, &sort);

return 0;
}
**/

