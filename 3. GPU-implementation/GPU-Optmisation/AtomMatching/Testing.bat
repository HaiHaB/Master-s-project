cd C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\Optmisation\AtomMatching

nvcc atomMatchingTest.cu atomMatchingOpt1.cu atomMatchingOpt3.cu atomMatchingOpt4.cu atomMatchingOpt5.cu atomMatchingOpt6.cu atomMatchingOpt7.cu "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\commonFunction.cpp" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\TestFunction.cu" -I"C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles" -o atomMatching.exe
atomMatching.exe

nvcc  atomMatchingOpt2.cu "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\commonFunction.cpp" "C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles\TestFunction.cu" -I"C:\Users\Hai Ha\Desktop\HH_SVN_BHAM\CommonFiles" -o atomMatching.exe
atomMatching.exe
