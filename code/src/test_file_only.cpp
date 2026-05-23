#include <fstream>
int main() {
    std::ofstream log("C:\\Users\\david\\test_log.txt", std::ios::trunc);
    log << "alive" << std::endl;
    return 0;
}
