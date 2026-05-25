@echo off
cd /d "C:\McMaker Projects\Projects\Cerberus - Copy\code"
set "C=C:\gcc-15.2.0\mingw64\bin\g++.exe"
%C% -std=c++26 -Wall -Wextra -Wpedantic -Werror -c -o test_hdr.o test_hdr.cpp ^
  "-IC:\McMaker Projects\Projects\Cerberus - Copy\code\include" ^
  "-IC:\McMaker Projects\Projects\Cerberus - Copy\code\openvino\include" ^
  "-isystemC:\McMaker Projects\Projects\Cerberus - Copy\code\ort\include" ^
  "-isystemC:\PROGRA~1\NVIDIA~2\CUDA\v13.2\include" ^
  "-isystemC:\McMaker Projects\Projects\PsiForceDB\Src\Inc\Intake\include" ^
  "-isystemC:\McMaker Projects\Projects\PsiForceDB\Src\Database\Intake\include" ^
  "-isystemC:\McMaker Projects\Projects\PsiForceDB\Src\Unclassified\Intake\include" ^
  "-isystemC:\McMaker Projects\Projects\PsiForceDB\Src\Inc\Concurrency\include" ^
  "-isystemC:\McMaker~1\Projects\LFSSL-~1\include"
