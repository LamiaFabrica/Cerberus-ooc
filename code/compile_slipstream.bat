@echo off
set "C=C:\gcc-15.2.0\mingw64\bin\g++.exe"
%C% -std=c++26 -O3 -DNDEBUG "-IC:\McMaker Projects\Projects\Cerberus - Copy\code\include" -c -o slipstream_test.o test_slipstream.cpp 2^>^&1
