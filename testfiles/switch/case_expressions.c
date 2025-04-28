// Mode: success
// ExitCode: 7
int main(void) {
    int acc = 0;
    for (int i = 0; i <= 6; i++) {
        switch(i) {
            case 0 || 0: acc++; break;
            case 1 && 1: acc++; break;
            case (3 < 5) * 2: acc++; break;
            case (3 << 1) >> 1: acc++; break;
            case (4 > 1) * 400 / 100: acc++; break;
            case ~~(5 ^ 5 ^ 5): acc++; break;
            case -(-6): acc++; break;
        }
    }
    return acc;
}
