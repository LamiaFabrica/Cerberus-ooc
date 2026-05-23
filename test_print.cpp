#include <cstdio>
#include <iostream>

int main(int argc, char**) {
    std::printf("C printf: hello\n");
    std::cout << "C++ cout: hello" << std::endl;
    std::cerr << "C++ cerr: hello" << std::endl;
    std::fflush(stdout);
    return argc;
}
