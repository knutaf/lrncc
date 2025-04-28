// Mode: fail
// make sure we apply usual label validation in switch statement
int main(void) {
    int a = 3;
    switch (a) {
        default: goto foo;
        case 1: return 0;
    }
    return 0;
}
