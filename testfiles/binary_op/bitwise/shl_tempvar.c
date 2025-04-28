// Mode: success
// ExitCode: 20
/* test support for left and right shift operations where the right operand
 * (i.e. the shift count) is the result of another expression, not a constant.
 */

int main(void) {
    return 5 << (2 * 1);
}
