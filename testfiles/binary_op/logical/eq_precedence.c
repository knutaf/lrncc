// Mode: success
// ExitCode: 1
#ifdef SUPPRESS_WARNINGS
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wparentheses"
#endif
#endif

int main(void) {
    return 0 == 0 != 0;
}
