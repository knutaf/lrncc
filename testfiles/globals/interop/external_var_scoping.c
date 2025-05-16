// Mode: library
int read_x(void) {
    //  define x without linkage
    int x = 4;
    if (x == 4) {
        /* declare x with linkage, shadowing previous definition;
         * this refers to the variable defined in the other file.
         */
        extern int x;
        return x;
    } else {
        return -1;
    }
}

// Mode: success
// ExitCode: 0
/* Validate that the 'extern' keyword can bring a variable
 * with external linkage into scope, whether it's defined in the current
 * translation unit or a different one
 */

int x = 10;

int read_x(void);

int main(void) {
    // shadow x
    int x = 0;

    if (x == 0) {
        /* the value of x is still visible in external_var_scoping.c
         * even if it's shadowed here
         */
        if (read_x() != 10)
            return 1;

        // bring x back into scope
        extern int x;
        if (x != 10)
            return 1;

        return 0;
    }
    return 1;
}
