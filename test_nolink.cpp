#include <cstdio>
int main() {
    std::fprintf(stderr, "BEFORE_INIT\n");
    fflush(stderr);
    std::printf("BEFORE_INIT_STDOUT\n");
    fflush(stdout);

    bool ok = true;
    std::fprintf(stderr, "RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
