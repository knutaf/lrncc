// Mode: fail
int main(void) {
    int x = 0;
    if (x) {
        x = 5;
        goto l;
        return 0;
        l:
            return x;
    } else {
        goto l;
        return 0;
        l:
            return x;
    }
}
