// Mode: success
// ExitCode: 252
#ifdef SUPPRESS_WARNINGS
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wempty-body"
#endif
#endif

int main(void) {
    int i = 502;
    do ; while ((i = i - 5) >= 256);

    return i;
}
