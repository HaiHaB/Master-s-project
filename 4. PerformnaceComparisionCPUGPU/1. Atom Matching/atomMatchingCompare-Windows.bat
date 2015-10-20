set Pathname="C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\performnaceComparisionCPUGPU\Atom Matching"
cd /d %Pathname%

nvcc AtomMatchingCompare.cu AtomMatching.cu "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\commonFunction.cpp" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\TestFunction.cu" -I"C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles"  "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation\atomMatchingCPU.cpp" -I "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation-Parallel\atomMatchingCPUParallel.cpp" -I"C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation-Parallel" -o atomMatching.exe

atomMatching.exe
