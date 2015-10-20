#ifndef ATOMMATCHINGCPU_H
#define ATOMMATCHINGCPU_H

/**
Compare every element in each row of a to every element in each row of b.
The match distances is stored in c.
Time elapsed were recorded in a specified file **/

void atomMatchingCPU(float* c, const float *a,  const float *b,
	 const int NA, const int NB, const int NMax, float threshold, string fileName);

#endif
