set Pathname="C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\performnaceComparisionCPUGPU\Calculate Similarity\"
cd /d %Pathname%

nvcc calculateSimilarityCompare.cu calculateSimilarity.cu "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\commonFunction.cpp" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\TestFunction.cu" -I"C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation\calculateSimilarityCPU.cpp" -I "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation-Parallel\calculateSimilarityCPUParallel.cpp" -I"C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CPU-implementation-Parallel" -o calculateSimilarity.exe

calculateSimilarity.exe
