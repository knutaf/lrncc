// Mode: library
/* this function defines a variable with external linkage */

/* a tentative definition of x;
 * this will be treated like a declaration,
 * since x is defined later */
int x;

/* a variable with external linkage can be tentatively defined
 * and declared multiple times in a file
 */
extern int x;
int x;

int update_x(int new_val) {
    x = new_val;
    return 0;
}

int read_x(void) {
    return x;
}

/* the definition of x */
int x = 3;

// Mode: success
// ExitCode: 0
/* This test program contains a variable with external linkage named 'x'.
 * It verifies that the variable is initialized correctly,
 * code in both files can read and update this variable,
 * and updates made in one file are visible in the other.
 */

int update_x(int new_val);
int read_x(void);


// an external variable can be declared multiple times.
extern int x;

int main(void) {
    /* x is initialized to 3 in external_variable.c,
     * where it's defined.
     */
    if (x != 3)
        return 1;
    if (read_x() != 3)
        return 1;


    /* when we make updates, they're visible here and in external_variable.c */
    x = 4;
    if (x != 4)
        return 1;
    if (read_x() != 4)
        return 1;

    /* when code in external_variable.c makes updates,
     * they're visible here and in external_variable.c
     */
    update_x(5);
    if (x != 5)
        return 1;
    if (read_x() != 5)
        return 1;

    return 0;
}
