// Mode: library
int add(int x, int y) {
    return x + y;
}

// Mode: success
// ExitCode: 3
int add(int x, int y);

int main(void) {
    return add(1, 2);
}
