// Mode: success
// ExitCode: 56
int func0()
{
    return 1;
}

int func1(int a)
{
    return a;
}

int func2(int a, int b)
{
    return a + b;
}

int func3(int a, int b, int c)
{
    return a + b + c;
}

int func4(int a, int b, int c, int d)
{
    return a + b + c + d;
}

int func5(int a, int b, int c, int d, int e)
{
    return a + b + c + d + e;
}

int func6(int a, int b, int c, int d, int e, int f)
{
    return a + b + c + d + e + f;
}

int main() {
    return func6(
        func1(func0()),
        func2(1, 2),
        func3(1, 2, 3),
        func4(1, 2, 3, 4),
        func5(1, 2, 3, 4, 5),
        func6(1, 2, 3, 4, 5, 6));
}

