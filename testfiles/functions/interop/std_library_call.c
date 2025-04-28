// Mode: library
int putchar(int c);

int incr_and_print(int b) {
    return putchar(b + 2);
}

// Mode: success_and_output
// ExitCode: 0
// Output: H
int incr_and_print(int c);

int main(void) {
    incr_and_print(70);
    return 0;
}
