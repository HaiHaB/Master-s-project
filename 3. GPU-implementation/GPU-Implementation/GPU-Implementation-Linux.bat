cd /media/A015-9565/HH_SVN_BHAM/GPU-implementation/

nvcc GPU-Implementation.cu calculateSimilarity.cu AtomMatching.cu Sort.cu /media/A015-9565/HH_SVN_BHAM/CommonFiles/commonFunction.cpp /media/A015-9565/HH_SVN_BHAM/CommonFiles/readFromFile.cpp /media/A015-9565/HH_SVN_BHAM/CommonFiles/parsedFile.cpp /media/A015-9565/HH_SVN_BHAM/CommonFiles/fileCollection.cpp -I/media/A015-9565/HH_SVN_BHAM/CommonFiles -o  GPU-Implementation.out

 GPU-Implementation.out
