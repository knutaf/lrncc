// Mode: success
// ExitCode: 21
int func()
{
    int a = 1;
    {
        int b = 2;
        {
            int c = 3;
            {
                int d = 4;
                a = a + b + c + d;
            }
        }
    }
    int e = 5;
    int f = 6;
    return a + e + f;
}

int main() {
    return func();
}

