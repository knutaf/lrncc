// Mode: fail
int main(void) {
    switch(4) {
        case 5: return 0;
        case 4: return 1;
        case ((4 * 100 - 395) << 1) >> 1: return 0; // duplicate of previous case 5
        default: return 2;
    }
}
