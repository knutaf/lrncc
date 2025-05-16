// Mode: library
/* Tentatively define a variable, and make sure it's initialized to 0. */
int x;

int read_x(void) {
    return x;
}

// Mode: success
// ExitCode: 0
/* Read and write a variable with external linkage
 * that is declared in this file and tentatively defined
 * in another file.
 */
int read_x(void);

int main(void) {
    // This brings x into scope
    extern int x;
    if (x != 0)
        // x is initialized to zero in external_tentative_var.c
        return 1;

    // update x
    x = 3;
    // make sure update is visible to code in external_tentative_var.c
    if (read_x() != 3)
        return 1;

    return 0;
}
