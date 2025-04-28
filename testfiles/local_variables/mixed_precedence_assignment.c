// Mode: success
// ExitCode: 20
int main(void) {
    int x = 5;
    int y = 4;
    x = 3 * (y = x);
    return x + y;
}
