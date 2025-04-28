// Mode: success
// ExitCode: 1
#ifdef SUPPRESS_WARNINGS
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

int main(void) {
    return 20 >> 4 <= 3 << 1;
}
