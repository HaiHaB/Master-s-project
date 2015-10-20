cd /media/A015-9565/HH_SVN_BHAM/performnaceComparisionCPUGPU/Atom Matching

rm atomMatching.exe

cd /media/A015-9565/HH_SVN_BHAM/performnaceComparisionCPUGPU/Atom Matching
nvcc AtomMatchingCompare.cu AtomMatching.cu /media/A015-9565/HH_SVN_BHAM/CommonFiles/commonFunction.cpp /media/A015-9565/HH_SVN_BHAM/CommonFiles/TestFunction.cu -I/media/A015-9565/HH_SVN_BHAM/CommonFiles  /media/A015-9565/HH_SVN_BHAM/CPU-implementation/atomMatchingCPU.cpp -I/media/A015-9565/HH_SVN_BHAM/CPU-implementation -o atomMatching.exe

cd /media/A015-9565/HH_SVN_BHAM/performnaceComparisionCPUGPU/Atom Matching
atomMatching.exe
