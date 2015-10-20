set Pathname="C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\performnaceComparisionCPUGPU\Calculate Similarity\"
cd /d %Pathname%

nvcc calculateSimilarityCompare.cu calculateSimilarity.cu /media/A015-9565/HH_SVN_BHAM/CommonFiles/commonFunction.cpp /media/A015-9565/HH_SVN_BHAM/CommonFiles/TestFunction.cu -I/media/A015-9565/HH_SVN_BHAM/CommonFiles /media/A015-9565/HH_SVN_BHAM/CPU-implementation/calculateSimilarityCPU.cpp -I/media/A015-9565/HH_SVN_BHAM/CPU-implementation -o calculateSimilarity.exe



calculateSimilarity.exe
