// Mode: fail
int main(void) {
    {
        /* This declares a function 'a'
         * with external linkage
         */
        extern int a();
    }
    /* a is no longer in scope after the end of the block */
    return a();
}

int a() {
    return 0;
}
