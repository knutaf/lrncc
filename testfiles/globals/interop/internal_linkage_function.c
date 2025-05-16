// Mode: library
/* A function with internal linkage can be declared multiple times */
static int my_fun(void);

int call_static_my_fun(void) {
    return my_fun();
}

int call_static_my_fun_2(void) {
    /* when you declare a function at block scope,
     * it takes on the linkage of already-visible declaration
     */
    int my_fun(void);
    return my_fun();
}

extern int my_fun(void);

static int my_fun(void);

int my_fun(void) {
    static int i = 0;
    i = i + 1;
    return i;
}

// Mode: success
// ExitCode: 0
/* This program contains two functions called 'my_fun'.
 * The one in this file has external linkage, and the one in
 * internal_linkage_function.c has internal linkage.
 * Verify that these are distinct functions, each of which is only
 * callable in the file where it's defined.
 */

/* forward declration of function with external linkage,
 * defined in this file
 */
extern int my_fun(void);

// functions with external linkage from other file
int call_static_my_fun(void);
int call_static_my_fun_2(void);

int main(void) {

    if (call_static_my_fun() != 1)
        return 1;
    if (my_fun() != 100)
        return 1;
    if (call_static_my_fun_2() != 2)
        return 1;
    return 0;
}

int my_fun(void) {
    return 100;
}
