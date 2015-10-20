cd C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\GPU implementation

nvcc GPU-Implementation.cu calculateSimilarity.cu AtomMatching.cu Sort.cu "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\commonFunction.cpp" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\readFromFile.cpp" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\TestFunction.cu" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\parsedFile.cpp" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\fileCollection.cpp" -I"C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles" -o GPU-Implementation.exe
GPU-Implementation.exe
