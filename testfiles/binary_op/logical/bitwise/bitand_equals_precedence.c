// Mode: success
// ExitCode: 0
#ifdef SUPPRESS_WARNINGS
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

int main(void) {
    // & has lower precedence than ==
    return 4 & 7 == 4;
}
