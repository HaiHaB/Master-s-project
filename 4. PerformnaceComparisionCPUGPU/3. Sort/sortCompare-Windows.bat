set Pathname="C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\performnaceComparisionCPUGPU\Sort"
cd /d %Pathname%

nvcc SortCompare.cu Sort.cu SortCPU.cpp /media/A015-9565/HH_SVN_BHAM/CommonFiles/commonFunction.cpp -I/media/A015-9565/HH_SVN_BHAM/CommonFiles -o sort.exe

nvcc AtomMatchingCompare.cu AtomMatching.cu /media/A015-9565/HH_SVN_BHAM/CommonFiles/commonFunction.cpp /media/A015-9565/HH_SVN_BHAM/CommonFiles/TestFunction.cu -I/media/A015-9565/HH_SVN_BHAM/CommonFiles  /media/A015-9565/HH_SVN_BHAM/CPU-implementation/atomMatchingCPU.cpp -I/media/A015-9565/HH_SVN_BHAM/CPU-implementation -o atomMatching.exe
sort.exe
