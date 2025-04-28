// Mode: success
// ExitCode: 1
#ifdef SUPPRESS_WARNINGS
#pragma GCC diagnostic ignored "-Wunused-label"
#endif
int main(void) {
    goto label2;
    return 0;
    // okay to have space or newline between label and colon
    label1 :
    return 1;
    label2
    :
    goto label1;
}
