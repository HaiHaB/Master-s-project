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
using namespace std;


#ifndef SORT_H
#define SORT_H

void sort(float* keys, string* values, int size, int top,
  string outputFileName, string outPutTime, string molecule2Compare);

#endif
