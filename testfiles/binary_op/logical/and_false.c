// Mode: success
// ExitCode: 0
#ifdef SUPPRESS_WARNINGS
#ifdef __clang__
#pragma clang diagnostic ignored "-Wconstant-logical-operand"
#endif
#endif

int main(void) {
    return (10 && 0) + (0 && 4) + (0 && 0);
}
