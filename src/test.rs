use {
    crate::*,
    tracing_subscriber,
    tracing_subscriber::{util::SubscriberInitExt, EnvFilter, Registry},
};

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! test {
        ($name:ident $body:block) => {
            #[test]
            fn $name() {
                let _ = Registry::default()
                    .with(
                        tracing_forest::ForestLayer::new(
                            tracing_forest::printer::TestCapturePrinter::new(),
                            tracing_forest::tag::NoTag,
                        )
                        .with_filter(EnvFilter::from_default_env()),
                    )
                    .try_init();

                $body
            }
        };
    }

    mod lex {
        use super::*;

        test!(simple {
            let input = r"int main() {
        return 2;
    }";
            assert_eq!(
                lex_all_tokens(&input),
                Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"])
            );
        });

        test!(no_whitespace {
            let input = r"int main(){return 2;}";
            assert_eq!(
                lex_all_tokens(&input),
                Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"])
            );
        });

        test!(negative {
            assert_eq!(
                lex_all_tokens("int main() { return -1; }"),
                Ok(vec![
                    "int", "main", "(", ")", "{", "return", "-", "1", ";", "}"
                ])
            );
        });

        test!(bitwise_not {
            assert_eq!(
                lex_all_tokens("int main() { return ~1; }"),
                Ok(vec![
                    "int", "main", "(", ")", "{", "return", "~", "1", ";", "}"
                ])
            );
        });

        test!(logical_not {
            assert_eq!(
                lex_all_tokens("int main() { return !1; }"),
                Ok(vec![
                    "int", "main", "(", ")", "{", "return", "!", "1", ";", "}"
                ])
            );
        });

        test!(no_at {
            assert!(lex_all_tokens("int main() { return 0@1; }").is_err());
        });

        test!(no_backslash {
            assert!(lex_all_tokens("\\").is_err());
        });

        test!(no_backtick {
            assert!(lex_all_tokens("`").is_err());
        });

        test!(bad_identifier {
            assert!(lex_all_tokens("int main() { return 1foo; }").is_err());
        });

        test!(no_at_identifier {
            assert!(lex_all_tokens("int main() { return @b; }").is_err());
        });
    }

    fn codegen_run_and_check_exit_code_or_compile_failure(
        input: &str,
        expected_result: Option<i32>,
    ) {
        let args = LcArgs {
            input_path: String::new(),
            output_path: Some(format!("test_{}.exe", generate_random_string(8))),
            mode: Mode::All,
            verbose: true,
        };

        let compile_result = compile_and_link(&args, input, true, expected_result.is_some());
        if compile_result.is_ok() {
            let exe_path_abs = Path::new(args.output_path.as_ref().unwrap())
                .canonicalize()
                .unwrap();
            let exe_path_str = exe_path_abs.to_str().unwrap();
            let mut pdb_path = exe_path_abs.clone();
            pdb_path.set_extension("pdb");

            if let Some(expected_exit_code) = expected_result {
                let actual_exit_code = Command::new(exe_path_str)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .expect(&format!("failed to run {}", exe_path_str))
                    .code()
                    .expect("all processes must have exit code");

                assert_eq!(expected_exit_code, actual_exit_code);
                std::fs::remove_file(&exe_path_abs);
                std::fs::remove_file(&pdb_path);
            } else {
                assert!(false, "compile succeeded but expected failure");
            }
        } else {
            println!("compile failed! {:?}", compile_result);
            assert!(expected_result.is_none());
        }
    }

    fn validate_error_count(input: &str, expected_error_count: usize) {
        match parse_and_validate(Mode::All, input) {
            Ok(_ast) => {
                // If parsing succeeded, then the caller should have expected 0 errors.
                assert_eq!(expected_error_count, 0);
            }
            Err(errors) => {
                assert_eq!(expected_error_count, errors.len());
            }
        }
    }

    fn codegen_run_and_check_exit_code(input: &str, expected_exit_code: i32) {
        codegen_run_and_check_exit_code_or_compile_failure(input, Some(expected_exit_code))
    }

    fn codegen_run_and_expect_compile_failure(input: &str) {
        codegen_run_and_check_exit_code_or_compile_failure(input, None)
    }

    fn test_codegen_expression(expression: &str, expected_exit_code: i32) {
        codegen_run_and_check_exit_code(
            &format!("int main() {{ return {}; }}", expression),
            expected_exit_code,
        );
    }

    fn test_codegen_mainfunc(body: &str, expected_exit_code: i32) {
        codegen_run_and_check_exit_code(
            &format!("int main() {{\n{}\n}}", body),
            expected_exit_code,
        );
    }

    fn test_codegen_mainfunc_failure(body: &str) {
        codegen_run_and_expect_compile_failure(&format!("int main() {{\n{}\n}}", body))
    }

    mod fail {
        use super::*;

        test!(parse_extra_paren {
            test_codegen_mainfunc_failure("return (3));");
        });

        test!(parse_unclosed_paren {
            test_codegen_mainfunc_failure("return (3;");
        });

        test!(parse_missing_immediate {
            test_codegen_mainfunc_failure("return ~;");
        });

        test!(parse_missing_immediate_2 {
            test_codegen_mainfunc_failure("return -~;");
        });

        test!(parse_missing_semicolon {
            test_codegen_mainfunc_failure("return 5");
        });

        test!(parse_missing_semicolon_binary_op {
            test_codegen_mainfunc_failure("return 5 + 6");
        });

        test!(parse_parens_around_operator {
            test_codegen_mainfunc_failure("return (-)5;");
        });

        test!(parse_operator_wrong_order {
            test_codegen_mainfunc_failure("return 5-;");
        });

        test!(parse_double_operator {
            test_codegen_mainfunc_failure("return 1 * / 2;");
        });

        test!(parse_unbalanced_paren {
            test_codegen_mainfunc_failure("return 1 + (2;");
        });

        test!(parse_missing_opening_paren {
            test_codegen_mainfunc_failure("return 1 + 2);");
        });

        test!(parse_unexpected_paren {
            test_codegen_mainfunc_failure("return 1 (- 2);");
        });

        test!(parse_misplaced_semicolon_paren {
            test_codegen_mainfunc_failure("return 1 + (2;)");
        });

        test!(parse_missing_first_binary_operand {
            test_codegen_mainfunc_failure("return / 2;");
        });

        test!(parse_missing_second_binary_operand {
            test_codegen_mainfunc_failure("return 2 / ;");
        });

        test!(parse_relational_missing_first_operand {
            test_codegen_mainfunc_failure("return <= 2;");
        });

        test!(parse_relational_missing_second_operand {
            test_codegen_mainfunc_failure("return 1 < > 3;");
        });

        test!(parse_and_missing_second_operand {
            test_codegen_mainfunc_failure("return 2 && ~;");
        });

        test!(parse_or_missing_semicolon {
            test_codegen_mainfunc_failure("return 2 || 4");
        });

        test!(parse_unary_not_missing_semicolon {
            test_codegen_mainfunc_failure("return !10");
        });

        test!(parse_double_bitwise_or {
            test_codegen_mainfunc_failure("return 1 | | 2;");
        });

        test!(parse_unary_not_missing_operand {
            test_codegen_mainfunc_failure("return 10 <= !;");
        });

        test!(duplicate_variable {
            test_codegen_mainfunc_failure("int x = 5; int x = 4; return x;");
        });

        test!(duplicate_variable_after_use {
            test_codegen_mainfunc_failure("int x = 5; return x; int x = 4; return x;");
        });

        test!(unknown_variable {
            test_codegen_mainfunc_failure("return x;");
        });

        test!(unknown_variable_after_shortcircuit {
            test_codegen_mainfunc_failure("return 0 && x;");
        });

        test!(unknown_variable_in_binary_op {
            test_codegen_mainfunc_failure("return x < 5;");
        });

        test!(unknown_variable_in_bitwise_op {
            test_codegen_mainfunc_failure("return a >> 2;");
        });

        test!(unknown_variable_in_unary_op {
            test_codegen_mainfunc_failure("return -x;");
        });

        test!(unknown_variable_lhs_compound_assignment {
            test_codegen_mainfunc_failure("a += 1; return 0;");
        });

        test!(unknown_variable_rhs_compound_assignment {
            test_codegen_mainfunc_failure("int b = 10; b *= a; return 0;");
        });

        test!(malformed_plusequals {
            test_codegen_mainfunc_failure("int a = 0; a + = 1; return a;");
        });

        test!(malformed_decrement {
            test_codegen_mainfunc_failure("int a = 5; a - -; return a;");
        });

        test!(malformed_increment {
            test_codegen_mainfunc_failure("int a = 5; a + +; return a;");
        });

        test!(malformed_less_equals {
            test_codegen_mainfunc_failure("return 1 < = 2;");
        });

        test!(malformed_not_equals {
            test_codegen_mainfunc_failure("return 1 ! = 2;");
        });

        test!(malformed_divide_equals {
            test_codegen_mainfunc_failure("int a = 10; a =/ 5; return a;");
        });

        test!(missing_semicolon {
            test_codegen_mainfunc_failure("int a = 5 a = a + 5; return a;");
        });

        test!(return_in_assignment {
            test_codegen_mainfunc_failure("int a = return 5;");
        });

        test!(declare_keyword_as_var {
            test_codegen_mainfunc_failure("int return = 6; return return + 1;");
        });

        test!(declare_after_use {
            test_codegen_mainfunc_failure("a = 5; int a; return a;");
        });

        test!(invalid_lvalue_binary_op {
            test_codegen_mainfunc_failure("int a = 5; a + 3 = 4; return a;");
        });

        test!(invalid_lvalue_unary_op {
            test_codegen_mainfunc_failure("int a = 5; !a = 4; return a;");
        });

        test!(declare_invalid_var_name_with_space {
            test_codegen_mainfunc_failure("int x y = 3; return y;");
        });

        test!(declare_invalid_var_name_starting_number {
            test_codegen_mainfunc_failure("int 10 = 3; return 10;");
            test_codegen_mainfunc_failure("int 10a = 3; return 10a;");
        });

        test!(declare_invalid_type_name {
            test_codegen_mainfunc_failure("ints x = 3; return x;");
        });

        test!(invalid_mixed_precedence_assignment {
            test_codegen_mainfunc_failure("int a = 1; int b = 2; a = 3 * b = a; return a;");
        });

        test!(compound_initializer {
            test_codegen_mainfunc_failure("int a += 0; return a;");
        });

        test!(invalid_unary_lvalue {
            test_codegen_mainfunc_failure("int a = 0; -a += 1; return a;");
        });

        test!(invalid_compound_lvalue {
            test_codegen_mainfunc_failure("int a = 10; (a += 1) -= 2;");
        });

        test!(decrement_binary_op {
            test_codegen_mainfunc_failure("int a = 0; return a -- 1;");
        });

        test!(increment_binary_op {
            test_codegen_mainfunc_failure("int a = 0; return a ++ 1;");
        });

        test!(increment_declaration {
            test_codegen_mainfunc_failure("int a++; return 0;");
        });

        test!(double_postfix {
            test_codegen_mainfunc_failure("int a = 10; return a++--;");
        });

        test!(postfix_incr_non_lvalue {
            test_codegen_mainfunc_failure("int a = 0; (a = 4)++;");
        });

        test!(prefix_incr_non_lvalue {
            test_codegen_mainfunc_failure("int a = 1; ++(a+1); return 0;");
        });

        test!(prefix_decr_constant {
            test_codegen_mainfunc_failure("return --3;");
        });

        test!(postfix_undeclared_var {
            test_codegen_mainfunc_failure("a--; return 0;");
        });

        test!(prefix_undeclared_var {
            test_codegen_mainfunc_failure("++a; return 0;");
        });

        test!(declaration_as_statement {
            test_codegen_mainfunc_failure("if (5) int i = 0;");
        });

        test!(empty_if_body {
            test_codegen_mainfunc_failure("if (0) else return 0;");
        });

        test!(if_as_assignment {
            test_codegen_mainfunc_failure("int flag = 0; int a = if (flag) 2; else 3; return a;");
        });

        test!(if_no_parentheses {
            test_codegen_mainfunc_failure("if 0 return 1;");
        });

        test!(extra_else {
            test_codegen_mainfunc_failure("if (1) return 1; else return 2; else return 3;");
        });

        test!(undeclared_var_in_if {
            test_codegen_mainfunc_failure("if (1) return c; int c = 0;");
        });

        test!(incomplete_ternary {
            test_codegen_mainfunc_failure("return 1 ? 2;");
        });

        test!(ternary_extra_left {
            test_codegen_mainfunc_failure("return 1 ? 2 ? 3 : 4;");
        });

        test!(ternary_extra_right {
            test_codegen_mainfunc_failure("return 1 ? 2 : 3 : 4;");
        });

        test!(ternary_wrong_delimiter {
            test_codegen_mainfunc_failure("int x = 10; return x ? 1 = 2;");
        });

        test!(ternary_undeclared_var {
            test_codegen_mainfunc_failure("return a > 0 ? 1 : 2; int a = 5;");
        });

        test!(ternary_invalid_assign {
            test_codegen_mainfunc_failure(
                r"
            int a = 2;
            int b = 1;
            a > b ? a = 1 : a = 0;
            return a;
            ",
            );
        });

        test!(keywords_as_var_identifier {
            test_codegen_mainfunc_failure("int if = 0; return if;");
            test_codegen_mainfunc_failure("int int = 0; return int;");
            test_codegen_mainfunc_failure("int void = 0; return void;");
            test_codegen_mainfunc_failure("int return = 0; return return;");
        });

        test!(extra_closing_brace {
            test_codegen_mainfunc_failure("if(0){ return 1; }} return 2;");
        });

        test!(missing_closing_brace {
            test_codegen_mainfunc_failure("if(0){ return 1; return 2;");
        });

        test!(missing_semicolon_in_block {
            test_codegen_mainfunc_failure("int a = 4; { a = 5; return a }");
        });

        test!(block_in_ternary {
            test_codegen_mainfunc_failure("int a; return 1 ? { a = 2 } : a = 4;");
        });

        test!(duplicate_var_declaration {
            test_codegen_mainfunc_failure("{ int a; int a; }");
        });

        test!(use_var_after_scope {
            test_codegen_mainfunc_failure("{ int a = 2; } return a;");
        });

        test!(use_var_before_declare {
            test_codegen_mainfunc_failure("int a; { b = 10; } int b; return b;");
        });

        test!(duplicate_var_declaration_after_block {
            test_codegen_mainfunc_failure(
                r"
    int a = 3;
    {
        a = 5;
    }
    int a = 2;
    return a;
            ",
            );
        });

        test!(break_outside_loop {
            test_codegen_mainfunc_failure(r"if (1) break;");
        });

        test!(continue_outside_loop {
            test_codegen_mainfunc_failure(
                r"
    {
        int a;
        continue;
    }
    return 0;
            ",
            );
        });
    }

    test!(unary_neg {
        test_codegen_expression("-5", -5);
    });

    test!(unary_bitnot {
        test_codegen_expression("~12", -13);
    });

    test!(unary_not {
        test_codegen_expression("!5", 0);
        test_codegen_expression("!0", 1);
    });

    test!(unary_neg_zero {
        test_codegen_expression("-0", 0);
    });

    test!(unary_bitnot_zero {
        test_codegen_expression("~0", -1);
    });

    test!(unary_neg_min_val {
        test_codegen_expression("-2147483647", -2147483647);
    });

    test!(unary_bitnot_and_neg {
        test_codegen_expression("~-3", 2);
    });

    test!(unary_bitnot_and_neg_zero {
        test_codegen_expression("-~0", 1);
    });

    test!(unary_bitnot_and_neg_min_val {
        test_codegen_expression("~-2147483647", 2147483646);
    });

    test!(unary_grouping_outside {
        test_codegen_expression("(-2)", -2);
    });

    test!(unary_grouping_inside {
        test_codegen_expression("~(2)", -3);
    });

    test!(unary_grouping_inside_and_outside {
        test_codegen_expression("-(-4)", 4);
    });

    test!(unary_grouping_several {
        test_codegen_expression("-((((((10))))))", -10);
    });

    test!(unary_not_and_neg {
        test_codegen_expression("!-3", 0);
    });

    test!(unary_not_and_arithmetic {
        test_codegen_expression("!(4-4)", 1);
        test_codegen_expression("!(4 - 5)", 0);
    });

    test!(expression_binary_operation {
        test_codegen_expression("5 + 6", 11);
    });

    test!(expression_negative_divide {
        test_codegen_expression("-110 / 10", -11);
    });

    test!(expression_negative_multiply {
        test_codegen_expression("10 * -11", -110);
    });

    test!(expression_factors_and_terms {
        test_codegen_expression("(1 + 2 + 3 + 4) * (10 - 21)", -110);
    });

    test!(and_false {
        test_codegen_expression("(10 && 0) + (0 && 4) + (0 && 0)", 0);
    });

    test!(and_true {
        test_codegen_expression("1 && -1", 1);
    });

    test!(and_shortcircuit {
        test_codegen_expression("0 && (1 / 0)", 0);
    });

    test!(or_shortcircuit {
        test_codegen_expression("1 || (1 / 0)", 1);
    });

    test!(multi_shortcircuit {
        test_codegen_expression("0 || 0 && (1 / 0)", 0);
    });

    test!(and_or_precedence {
        test_codegen_expression("1 || 0 && 2", 1);
    });

    test!(and_or_precedence_2 {
        test_codegen_expression("(1 || 0) && 0", 0);
    });

    test!(relational_lt {
        test_codegen_expression("1234 < 1234", 0);
        test_codegen_expression("1234 < 1235", 1);
    });

    test!(relational_gt {
        test_codegen_expression("1234 > 1234", 0);
        test_codegen_expression("1234 > 1233", 1);
        test_codegen_expression("(1 > 2) + (1 > 1)", 0);
    });

    test!(relational_le {
        test_codegen_expression("1234 <= 1234", 1);
        test_codegen_expression("1234 <= 1233", 0);
        test_codegen_expression("1 <= -1", 0);
        test_codegen_expression("(0 <= 2) + (0 <= 0)", 2);
    });

    test!(relational_ge {
        test_codegen_expression("1234 >= 1234", 1);
        test_codegen_expression("1234 >= 1235", 0);
        test_codegen_expression("(1 >= 1) + (1 >= -4)", 2);
    });

    test!(equality_eq {
        test_codegen_expression("1234 == 1234", 1);
        test_codegen_expression("1234 == 1235", 0);
    });

    test!(equality_ne {
        test_codegen_expression("1234 != 1234", 0);
        test_codegen_expression("1234 != 1235", 1);
    });

    test!(logical_and {
        test_codegen_expression("0 && 1 && 2", 0);
        test_codegen_expression("5 && 6 && 7", 1);
        test_codegen_expression("5 && 6 && 0", 0);
    });

    test!(logical_or {
        test_codegen_expression("0 || 0 || 1", 1);
        test_codegen_expression("1 || 0 || 0", 1);
        test_codegen_expression("0 || 0 || 0", 0);
    });

    test!(equals_precedence {
        test_codegen_expression("0 == 0 != 0", 1);
    });

    test!(equals_relational_precedence {
        test_codegen_expression("2 == 2 >= 0", 0);
    });

    test!(equals_or_precedence {
        test_codegen_expression("2 == 2 || 0", 1);
    });

    test!(relational_associativity {
        test_codegen_expression("5 >= 0 > 1 <= 0", 1);
    });

    test!(compare_arithmetic_results {
        test_codegen_expression("~2 * -2 == 1 + 5", 1);
    });

    test!(all_operator_precedence {
        test_codegen_expression("-1 * -2 + 3 >= 5 == 1 && (6 - 6) || 7", 1);
    });

    test!(all_operator_precedence_2 {
        test_codegen_expression("(0 == 0 && 3 == 2 + 1 > 1) + 1", 1);
    });

    test!(arithmetic_operator_precedence {
        test_codegen_expression("1 * 2 + 3 * -4", -10);
    });

    test!(arithmetic_operator_associativity_minus {
        test_codegen_expression("5 - 2 - 1", 2);
    });

    test!(arithmetic_operator_associativity_div {
        test_codegen_expression("12 / 3 / 2", 2);
    });

    test!(arithmetic_operator_associativity_grouping {
        test_codegen_expression("(3 / 2 * 4) + (5 - 4 + 3)", 8);
    });

    test!(arithmetic_operator_associativity_grouping_2 {
        test_codegen_expression("5 * 4 / 2 - 3 % (2 + 1)", 10);
    });

    test!(sub_neg {
        test_codegen_expression("2- -1", 3);
    });

    test!(unop_add {
        test_codegen_expression("~2 + 3", 0);
    });

    test!(unop_parens {
        test_codegen_expression("~(1 + 2)", -4);
    });

    test!(modulus {
        test_codegen_expression("10 % 3", 1);
    });

    test!(bitand_associativity {
        test_codegen_expression("7 * 1 & 3 * 1", 3);
    });

    test!(or_xor_associativity {
        test_codegen_expression("7 ^ 3 | 3 ^ 1", 6);
    });

    test!(and_xor_associativity {
        test_codegen_expression("7 ^ 3 & 6 ^ 2", 7);
    });

    test!(shl_immediate {
        test_codegen_expression("5 << 2", 20);
    });

    test!(shl_tempvar {
        test_codegen_expression("5 << (2 * 1)", 20);
    });

    test!(sar_immediate {
        test_codegen_expression("20 >> 2", 5);
    });

    test!(sar_tempvar {
        test_codegen_expression("20 >> (2 * 1)", 5);
    });

    test!(shift_associativity {
        test_codegen_expression("33 << 4 >> 2", 132);
    });

    test!(shift_associativity_2 {
        test_codegen_expression("33 >> 2 << 1", 16);
    });

    test!(shift_precedence {
        test_codegen_expression("40 << 4 + 12 >> 1", 0x00140000);
    });

    test!(sar_negative {
        test_codegen_expression("-5 >> 1", -3);
    });

    test!(bitwise_precedence {
        test_codegen_expression("80 >> 2 | 1 ^ 5 & 7 << 1", 21);
    });

    test!(arithmetic_and_booleans {
        test_codegen_expression("~(0 && 1) - -(4 || 3)", 0);
    });

    test!(bitand_equals_precedence {
        test_codegen_expression("4 & 7 == 4", 0);
    });

    test!(bitor_notequals_precedence {
        test_codegen_expression("4 | 7 != 4", 5);
    });

    test!(shift_relational_precedence {
        test_codegen_expression("20 >> 4 <= 3 << 1", 1);
    });

    test!(xor_relational_precedence {
        test_codegen_expression("5 ^ 7 < 5", 5);
    });

    test!(var_use {
        test_codegen_mainfunc(
            "int _x = 5; int y = 6; int z; _x = 1; z = 3; return _x + y + z;",
            10,
        );
    });

    test!(assign_expr {
        test_codegen_mainfunc("int x = 5; int y = x = 3 + 1; return x + y;", 8);
    });

    test!(declaration_after_expression {
        test_codegen_mainfunc("int x; x = 5; int y = -x; return y;", -5);
    });

    test!(mixed_precedence_assignment {
        test_codegen_mainfunc("int x = 5; int y = 4; x = 3 * (y = x); return x + y;", 20);
    });

    test!(assign_after_not_short_circuit_or {
        test_codegen_mainfunc("int x = 0; 0 || (x = 1); return x;", 1);
    });

    test!(assign_after_short_circuit_and {
        test_codegen_mainfunc("int x = 0; 0 && (x = 1); return x;", 0);
    });

    test!(assign_after_short_circuit_or {
        test_codegen_mainfunc("int x = 0; 1 || (x = 1); return x;", 0);
    });

    test!(assign_low_precedence {
        test_codegen_mainfunc("int x; x = 0 || 5; return x;", 1);
    });

    test!(assign_var_in_initializer {
        test_codegen_mainfunc("int x = x + 5; return x;", 5);
    });

    test!(empty_main_body {
        test_codegen_mainfunc("", 0);
    });

    test!(null_statement {
        test_codegen_mainfunc(";", 0);
    });

    test!(null_then_return {
        test_codegen_mainfunc("; return 1;", 1);
    });

    test!(empty_expression {
        test_codegen_mainfunc("return 0;;;", 0);
    });

    test!(unused_expression {
        test_codegen_mainfunc("2 + 2; return 0;", 0);
    });

    test!(bitwise_in_initializer {
        test_codegen_mainfunc(
            r"
    int a = 15;
    int b = a ^ 5;  // 10
    return 1 | b;   // 11",
            11,
        );
    });

    test!(bitwise_ops_vars {
        test_codegen_mainfunc("int a = 3; int b = 5; int c = 8; return a & b | c;", 9);
    });

    test!(bitwise_shl_var {
        test_codegen_mainfunc("int x = 3; return x << 3;", 24);
    });

    test!(bitwise_sar_assign {
        test_codegen_mainfunc(
            "int var_to_shift = 1234; int x = 0; x = var_to_shift >> 4; return x;",
            77,
        );
    });

    test!(compound_bitwise_and {
        test_codegen_mainfunc("int to_and = 3; to_and &= 6; return to_and;", 2);
    });

    test!(compound_bitwise_or {
        test_codegen_mainfunc("int to_or = 1; to_or |= 30; return to_or;", 31);
    });

    test!(compound_bitwise_shl {
        test_codegen_mainfunc("int to_shiftl = 3; to_shiftl <<= 4; return to_shiftl;", 48);
    });

    test!(compound_bitwise_sar {
        test_codegen_mainfunc(
            "int to_shiftr = 382574; to_shiftr >>= 4; return to_shiftr;",
            23910,
        );
    });

    test!(compound_bitwise_xor {
        test_codegen_mainfunc("int to_xor = 7; to_xor ^= 5; return to_xor;", 2);
    });

    test!(compound_div {
        test_codegen_mainfunc("int to_divide = 8; to_divide /= 4; return to_divide;", 2);
    });

    test!(compound_subtract {
        test_codegen_mainfunc(
            "int to_subtract = 10; to_subtract -= 8; return to_subtract;",
            2,
        );
    });

    test!(compound_mod {
        test_codegen_mainfunc("int to_mod = 5; to_mod %= 3; return to_mod;", 2);
    });

    test!(compound_mult {
        test_codegen_mainfunc(
            "int to_multiply = 4; to_multiply *= 3; return to_multiply;",
            12,
        );
    });

    test!(compound_add {
        test_codegen_mainfunc("int to_add = 0; to_add += 4; return to_add;", 4);
    });

    test!(compound_assignment_chained {
        test_codegen_mainfunc(
            r"
    int a = 250;
    int b = 200;
    int c = 100;
    int d = 75;
    int e = -25;
    int f = 0;
    int x = 0;
    x = a += b -= c *= d /= e %= f = -7;
    return a == 2250 && b == 2000 && c == -1800 && d == -18 && e == -4 &&
           f == -7 && x == 2250;
           ",
            1,
        );
    });

    test!(compound_bitwise_assignment_chained {
        test_codegen_mainfunc(
            r"
    int a = 250;
    int b = 200;
    int c = 100;
    int d = 75;
    int e = 50;
    int f = 25;
    int g = 10;
    int h = 1;
    int j = 0;
    int x = 0;
    x = a &= b *= c |= d = e ^= f += g >>= h <<= j = 1;
    return (a == 40 && b == 21800 && c == 109 && d == 41 && e == 41 &&
            f == 27 && g == 2 && h == 2 && j == 1 && x == 40);
           ",
            1,
        );
    });

    test!(compound_assignment_lowest_precedence {
        test_codegen_mainfunc(
            r"
    int a = 10;
    int b = 12;
    a += 0 || b;  // a = 11
    b *= a && 0;  // b = 0

    int c = 14;
    c -= a || b;  // c = 13

    int d = 16;
    d /= c || d; // d = 16
    return (a == 11 && b == 0 && c == 13 && d == 16);
    ",
            1,
        );
    });

    test!(compound_bitwise_assignment_lowest_precedence {
        test_codegen_mainfunc(
            r"
    int a = 11;
    int b = 12;
    a &= 0 || b;  // a = 1
    b ^= a || 1;  // b = 13

    int c = 14;
    c |= a || b;  // c = 15

    int d = 16;
    d >>= c || d;  // d = 8

    int e = 18;
    e <<= c || d; // e = 36
    return (a == 1 && b == 13 && c == 15 && d == 8 && e == 36);
    ",
            1,
        );
    });

    test!(increment_decrement_expressions {
        test_codegen_mainfunc(
            "int a = 0; int b = 0; a++; ++a; b--; --b; return (a == 2 && b == -2);",
            1,
        );
    });

    test!(incr_decr_in_binary_expressions {
        test_codegen_mainfunc(
            r"
    int a = 2;
    int b = 3 + a++;
    int c = 4 + ++b;
    return (a == 3 && b == 6 && c == 10);
    ",
            1,
        );
    });

    test!(incr_decr_parentheses {
        test_codegen_mainfunc(
            r"
    int a = 1;
    int b = 2;
    int c = -++(a);
    int d = !(b)--;
    return (a == 2 && b == 1 && c == -2 && d == 0);
    ",
            1,
        );
    });

    test!(if_assign {
        test_codegen_mainfunc("int x = 5; if (x == 5) x = 4; return x;", 4);
    });

    test!(if_else_assign {
        test_codegen_mainfunc(
            "int x = 5; if (x == 5) x = 4; else x == 6; if (x == 6) x = 7; else x = 8; return x;",
            8,
        );
    });

    test!(if_binary_op_in_condition_true {
        test_codegen_mainfunc("if (1 + 2 == 3) return 5;", 5);
    });

    test!(if_binary_op_in_condition_false {
        test_codegen_mainfunc("if (1 + 2 == 4) return 5;", 0);
    });

    test!(if_else_if {
        test_codegen_mainfunc(
            r"
    int a = 1;
    int b = 0;
    if (a)
        b = 1;
    else if (b)
        b = 2;
    return b;
    ",
            1,
        );
    });

    test!(if_else_if_nested_execute_else {
        test_codegen_mainfunc(
            r"
    int a = 0;
    int b = 1;
    if (a)
        b = 1;
    else if (~b)
        b = 2;
    return b;
    ",
            2,
        );
    });

    test!(if_nested_twice {
        test_codegen_mainfunc(
            r"
    int a = 0;
    if ( (a = 1) )
        if (a == 1)
            a = 3;
        else
            a = 4;
    return a;
    ",
            3,
        );
    });

    test!(if_nested_twice_execute_else {
        test_codegen_mainfunc(
            r"
    int a = 0;
    if (!a)
        if (3 / 4)
            a = 3;
        else
            a = 8 / 2;
    return a;
    ",
            4,
        );
    });

    test!(nested_else_execute_outer_else {
        test_codegen_mainfunc(
            r"
    int a = 0;
    if (0)
        if (0)
            a = 3;
        else
            a = 4;
    else
        a = 1;
    return a;
    ",
            1,
        );
    });

    test!(if_null_body {
        test_codegen_mainfunc(
            r"
    int x = 0;
    if (0)
        ;
    else
        x = 1;
    return x;
    ",
            1,
        );
    });

    test!(multiple_if_else {
        test_codegen_mainfunc(
            r"
    int a = 0;
    int b = 0;

    if (a)
        a = 2;
    else
        a = 3;

    if (b)
        b = 4;
    else
        b = 5;

    return a + b;
    ",
            8,
        );
    });

    test!(if_compound_assignment_in_condition {
        test_codegen_mainfunc("int a = 0; if (a += 1) return a; return 10;", 1);
    });

    test!(if_postfix_in_condition {
        test_codegen_mainfunc(
            r"
    int a = 0;
    if (a--)
        return 0;
    else if (a--)
        return 1;
    return 0;
    ",
            1,
        );
    });

    test!(if_prefix_in_condition {
        test_codegen_mainfunc(
            r"
    int a = -1;
    if (++a)
        return 0;
    else if (++a)
        return 1;
    return 0;
    ",
            1,
        );
    });

    test!(assign_ternary {
        test_codegen_mainfunc("int a = 0; a = 1 ? 2 : 3; return a;", 2);
    });

    test!(ternary_binary_op_in_middle {
        test_codegen_mainfunc("int a = 1 ? 3 % 2 : 4; return a;", 1);
    });

    test!(ternary_logical_or_precedence {
        test_codegen_mainfunc("int a = 10; return a || 0 ? 20 : 0;", 20);
    });

    test!(ternary_logical_or_precedence_right {
        test_codegen_mainfunc("return 0 ? 1 : 0 || 2;", 1);
    });

    test!(ternary_in_assignment {
        test_codegen_mainfunc(
            r"
    int x = 0;
    int y = 0;
    y = (x = 5) ? x : 2;
    return (x == 5 && y == 5);
    ",
            1,
        );
    });

    test!(nested_ternary {
        test_codegen_mainfunc(
            r"
    int a = 1;
    int b = 2;
    int flag = 0;
    return a > b ? 5 : flag ? 6 : 7;
    ",
            7,
        );
    });

    test!(nested_ternary_literals {
        test_codegen_mainfunc(
            r"
    int a = 1 ? 2 ? 3 : 4 : 5;
    int b = 0 ? 2 ? 3 : 4 : 5;
    return a * b;
    ",
            15,
        );
    });

    test!(ternary_assignment_rhs {
        test_codegen_mainfunc(
            r"
    int flag = 1;
    int a = 0;
    flag ? a = 1 : (a = 0);
    return a;
    ",
            1,
        );
    });

    mod goto {
        use super::*;

        test!(skip_declaration {
            test_codegen_mainfunc(
                r"
    int x = 1;
    goto post_declaration;
    // we skip over initializer, so it's not executed
    int i = (x = 0);

    post_declaration:
    // even though we didn't initialize i, it's in scope, so we can use it
    i = 5;
    return (x == 1 && i == 5);
            ",
                1,
            );
        });

        test!(same_as_var_name {
            test_codegen_mainfunc(
                r"
    // it's valid to use the same identifier as a variable and label
    int ident = 5;
    goto ident;
    return 0;
ident:
    return ident;
            ",
                5,
            );
        });

        test!(same_as_func_name {
            test_codegen_mainfunc(
                r"
    // it's legal to use main as both a function name and label
    goto main;
    return 5;
main:
    return 0;
            ",
                0,
            );
        });

        test!(nested_label {
            test_codegen_mainfunc(
                r"
    goto labelB;

    labelA:
        labelB:
            return 5;
    return 0;
            ",
                5,
            );
        });

        test!(label_all_statements {
            test_codegen_mainfunc(
                r"
    int a = 1;
label_if:
    if (a)
        goto label_expression;
    else
        goto label_empty;

label_goto:
    goto label_return;

    if (0)
    label_expression:
        a = 0;

    goto label_if;

label_return:
    return a;

label_empty:;
    a = 100;
    goto label_goto;
            ",
                100,
            );
        });

        test!(label_name {
            test_codegen_mainfunc(
                r"
    goto _foo_1_;  // a label may include numbers and underscores
    return 0;
_foo_1_:
    return 1;
            ",
                1,
            );
        });

        test!(unused_label {
            test_codegen_mainfunc(
                r"
unused:
    return 0;
            ",
                0,
            );
        });

        test!(whitespace_after_label {
            test_codegen_mainfunc(
                r"
    goto label2;
    return 0;
    // okay to have space or newline between label and colon
    label1 :
    return 1;
    label2
    :
    goto label1;
            ",
                1,
            );
        });

        test!(goto_after_declaration {
            test_codegen_mainfunc(
                r"
    int a = 0;
    {
        if (a != 0)
            return_a:
                return a;
        int a = 4;
        goto return_a;
    }
            ",
                0,
            );
        });

        test!(goto_inner_scope {
            test_codegen_mainfunc(
                r"
    int x = 5;
    goto inner;
    {
        int x = 0;
        inner:
        x = 1;
        return x;
    }
            ",
                1,
            );
        });

        test!(goto_outer_scope {
            test_codegen_mainfunc(
                r"
    int a = 10;
    int b = 0;
    if (a) {
        int a = 1;
        b = a;
        goto end;
    }
    a = 9;
end:
    return (a == 10 && b == 1);
            ",
                1,
            );
        });

        test!(jump_between_sibling_scopes {
            test_codegen_mainfunc(
                r"
    int sum = 0;
    if (1) {
        int a = 5;
        goto other_if;
        sum = 0;  // not executed
    first_if:
        // when we jump back into block at this label, a is uninitialized, so we need to initialize it again
        a = 5;
        sum = sum + a;  // sum = 11
    }
    if (0) {
    other_if:;
        int a = 6;
        sum = sum + a;  // sum = 6
        goto first_if;
        sum = 0;
    }
    return sum;
            ",
                11,
            );
        });

        mod fail {
            use super::*;

            test!(label_name {
                test_codegen_mainfunc_failure(r"0invalid_label: return 0;");
            });

            test!(label_keyword_name {
                test_codegen_mainfunc_failure(r"return: return 0;");
            });

            test!(whitespace_after_label {
                test_codegen_mainfunc_failure(
                    r"
    goto;
lbl:
    return 0;
                ",
                );
            });

            test!(label_declaration {
                test_codegen_mainfunc_failure(
                    r"
// NOTE: this is a syntax error in C17 but valid in C23
label:
    int a = 0;
                ",
                );
            });

            test!(label_without_statement {
                test_codegen_mainfunc_failure(
                    r"
    // NOTE: this is invalid in C17, but valid in C23
    foo:
                ",
                );
            });

            test!(parenthesized_label {
                test_codegen_mainfunc_failure(
                    r"
    goto(a);
a:
    return 0;
                ",
                );
            });

            test!(duplicate_label {
                test_codegen_mainfunc_failure(
                    r"
    int x = 0;
label:
    x = 1;
label:
    return 2;
                ",
                );
            });

            test!(unknown_label {
                test_codegen_mainfunc_failure(
                    r"
    goto label;
    return 0;
                ",
                );
            });

            test!(variable_as_label {
                test_codegen_mainfunc_failure(
                    r"
    int a;
    goto a;
    return 0;
                ",
                );
            });

            test!(undeclared_var_in_labeled_statement {
                test_codegen_mainfunc_failure(
                    r"
lbl:
    return a;
    return 0;
                ",
                );
            });

            test!(label_as_variable {
                test_codegen_mainfunc_failure(
                    r"
    int x = 0;
    a:
    x = a;
    return 0;
                ",
                );
            });

            test!(label_in_expression {
                test_codegen_mainfunc_failure(r"1 && label: 2;");
            });

            test!(label_outside_function {
                codegen_run_and_expect_compile_failure(
                    r"
label:
int main(void) {
    return 0;
}
                ",
                );
            });

            test!(different_label_same_scope {
                test_codegen_mainfunc_failure(
                    r"
    // different labels do not define different scopes
label1:;
    int a = 10;
label2:;
    int a = 11;
    return 1;
                ",
                );
            });

            test!(duplicate_labels_different_scopes {
                test_codegen_mainfunc_failure(
                    r"
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
                ",
                );
            });

            test!(goto_use_before_declare {
                test_codegen_mainfunc_failure(
                    r"
    int x = 0;
    if (x != 0) {
        return_y:
        return y; // not declared
    }
    int y = 4;
    goto return_y;
                ",
                );
            });

            test!(labeled_break_outside_loop {
                test_codegen_mainfunc_failure(
                    r"
    // make sure our usual analysis of break/continue labels also traverses labeled statements
    label: break;
    return 0;
                ",
                );
            });
        }
    }

    test!(var_assign_to_self {
        test_codegen_mainfunc(
            r"
    int a = 3;
    {
        int a = a = 4;
        return a;
    }
        ",
            4,
        );
    });

    test!(var_assign_to_self_inner_block {
        test_codegen_mainfunc(
            r"
    int a = 3;
    {
        int a = a = 4;
    }
    return a;
        ",
            3,
        );
    });

    test!(var_assign_to_self_from_other_var_declaration_in_inner_block {
        test_codegen_mainfunc(
            r"
    int a;
    {
        int b = a = 1;
    }
    return a;
        ",
            1,
        );
    });

    test!(empty_blocks {
        test_codegen_mainfunc(
            r"
    int ten = 10;
    {}
    int twenty = 10 * 2;
    {{}}
    return ten + twenty;
        ",
            30,
        );
    });

    test!(var_hidden_then_visible {
        test_codegen_mainfunc(
            r"
    int a = 2;
    int b;
    {
        a = -4;
        int a = 7;
        b = a + 1;
    }
    return b == 8 && a == -4;
        ",
            1,
        );
    });

    test!(var_shadowed {
        test_codegen_mainfunc(
            r"
    int a = 2;
    {
        int a = 1;
        return a;
    }
        ",
            1,
        );
    });

    test!(var_inner_uninitialized {
        test_codegen_mainfunc(
            r"
    int x = 4;
    {
        int x;
    }
    return x;
        ",
            4,
        );
    });

    test!(var_same_name_different_blocks {
        test_codegen_mainfunc(
            r"
    int a = 0;
    {
        int b = 4;
        a = b;
    }
    {
        int b = 2;
        a = a - b;
    }
    return a;
        ",
            2,
        );
    });

    test!(nested_if_compound_statements {
        test_codegen_mainfunc(
            r"
    int a = 0;
    if (a) {
        int b = 2;
        return b;
    } else {
        int c = 3;
        if (a < c) {
            return !a;
        } else {
            return 5;
        }
    }
    return a;
        ",
            1,
        );
    });

    test!(nested_var_declarations {
        test_codegen_mainfunc(
            r"
    int a; // a0
    int result;
    int a1 = 1; // a10
    {
        int a = 2; //a1
        int a1 = 2; // a11
        {
            int a; // a2
            {
                int a; // a3
                {
                    int a; // a4
                    {
                        int a; // a5
                        {
                            int a; // a6
                            {
                                int a; // a7
                                {
                                    int a; // a8
                                    {
                                        int a; // a9
                                        {
                                            int a = 20; // a10
                                            result = a;
                                            {
                                                int a; // a11
                                                a = 5;
                                                result = result + a;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result = result + a1;
    }
    return result + a1;
        ",
            28,
        );
    });

    mod loops {
        use super::*;

        mod for_loop {
            use super::*;

            test!(empty_header {
                test_codegen_mainfunc(
                    r"
        int a = 0;
        for (; ; ) {
            a = a + 1;
            if (a > 3)
                break;
        }

        return a;
                ",
                    4,
                );
            });

            test!(break_statement {
                test_codegen_mainfunc(
                    r"
        int a = 10;
        int b = 20;
        for (b = -20; b < 0; b = b + 1) {
            a = a - 1;
            if (a <= 0)
                break;
        }

        return a == 0 && b == -11;
                ",
                    1,
                );
            });

            test!(continue_statement {
                test_codegen_mainfunc(
                    r"
        int sum = 0;
        int counter;
        for (int i = 0; i <= 10; i = i + 1) {
            counter = i;
            if (i % 2 == 0)
                continue;
            sum = sum + 1;
        }

        return sum == 5 && counter == 10;
                ",
                    1,
                );
            });

            test!(declaration {
                test_codegen_mainfunc(
                    r"
        int a = 0;

        for (int i = -100; i <= 0; i = i + 1)
            a = a + 1;
        return a;
                ",
                    101,
                );
            });

            test!(shadow_declaration {
                test_codegen_mainfunc(
                    r"
        int shadow = 1;
        int acc = 0;
        for (int shadow = 0; shadow < 10; shadow = shadow + 1) {
            acc = acc + shadow;
        }
        return acc == 45 && shadow == 1;
                ",
                    1,
                );
            });

            test!(nested_shadowed_declarations {
                test_codegen_mainfunc(
                    r"
        int i = 0;
        int j = 0;
        int k = 1;
        for (int i = 100; i > 0; i = i - 1) {
            int i = 1;
            int j = i + k;
            k = j;
        }

        return k == 101 && i == 0 && j == 0;
                ",
                    1,
                );
            });

            test!(no_post_expression {
                test_codegen_mainfunc(
                    r"
        int a = -2147483647;
        for (; a % 5 != 0;) {
            a = a + 1;
        }
        return a % 5 == 0;
                ",
                    1,
                );
            });

            test!(continue_with_no_post_expression {
                test_codegen_mainfunc(
                    r"
        int sum = 0;
        for (int i = 0; i < 10;) {
            i = i + 1;
            if (i % 2)
                continue;
            sum = sum + i;
        }
        return sum;
                ",
                    30,
                );
            });

            test!(no_condition {
                test_codegen_mainfunc(
                    r"
        for (int i = 400; ; i = i - 100)
            if (i == 100)
                return 0;
                ",
                    0,
                );
            });

            test!(nested_break {
                test_codegen_mainfunc(
                    r"
        int ans = 0;
        for (int i = 0; i < 10; i = i + 1)
            for (int j = 0; j < 10; j = j + 1)
                if ((i / 2)*2 == i)
                    break;
                else
                    ans = ans + i;
        return ans;
                ",
                    250,
                );
            });

            test!(compound_assignent_in_post_expression {
                test_codegen_mainfunc(
                    r"
        int i = 1;
        for (i *= -1; i >= -100; i -=3)
            ;
        return (i == -103);
                ",
                    1,
                );
            });

            test!(jump_past_initializer {
                test_codegen_mainfunc(
                    r"
        int i = 0;
        goto target;
        for (i = 5; i < 10; i = i + 1)
        target:
            if (i == 0)
                return 1;
        return 0;
                ",
                    1,
                );
            });

            test!(jump_within_body {
                test_codegen_mainfunc(
                    r"
        int sum = 0;
        for (int i = 0;; i = 0) {
        lbl:
            sum = sum + 1;
            i = i + 1;
            if (i > 10)
                break;
            goto lbl;
        }
        return sum;
                ",
                    11,
                );
            });

            mod fail {
                use super::*;

                test!(extra_header_clause {
                    test_codegen_mainfunc_failure(
                        r"
        for (int i = 0; i < 10; i = i + 1; )
            ;
        return 0;
                    ",
                    );
                });

                test!(missing_header_clause {
                    test_codegen_mainfunc_failure(
                        r"
        for (int i = 0;)
            ;
        return 0;
                    ",
                    );
                });

                test!(extra_parens {
                    test_codegen_mainfunc_failure(
                        r"
        for (int i = 2; ))
            int a = 0;
                    ",
                    );
                });

                test!(invalid_declaration_compound_assignment {
                    test_codegen_mainfunc_failure(
                        r"
        for (int i += 1; i < 10; i += 1) {
            return 0;
        }
                    ",
                    );
                });

                test!(declaration_in_condition {
                    test_codegen_mainfunc_failure(
                        r"
        for (; int i = 0; i = i + 1)
            ;
        return 0;
                    ",
                    );
                });

                test!(label_in_header {
                    test_codegen_mainfunc_failure(
                        r"
        for (int i = 0; label: i < 10; i = i + 1) {
            ;
        }
        return 0;
                    ",
                    );
                });

                test!(undeclared_variable {
                    test_codegen_mainfunc_failure(
                        r"
        for (i = 0; i < 1; i = i + 1)
        {
            return 0;
        }
                    ",
                    );
                });

                test!(reference_body_variable_in_condition {
                    test_codegen_mainfunc_failure(
                        r"
        for (;; i++) {
            int i = 0;
        }
                    ",
                    );
                });
            }
        }

        mod while_loop {
            use super::*;

            test!(break_immediate {
                test_codegen_mainfunc(
                    r"
        int a = 10;
        while ((a = 1))
            break;
        return a;
                ",
                    1,
                );
            });

            test!(multi_break {
                test_codegen_mainfunc(
                    r"
        int i = 0;
        while (1) {
            i = i + 1;
            if (i > 10)
                break;
        }
        int j = 10;
        while (1) {
            j = j - 1;
            if (j < 0)
                break;
        }
        int result = j == -1 && i == 11;
        return result;
                ",
                    1,
                );
            });

            test!(nested_continue {
                test_codegen_mainfunc(
                    r"
        int x = 5;
        int acc = 0;
        while (x >= 0) {
            int i = x;
            while (i <= 10) {
                i = i + 1;
                if (i % 2)
                    continue;
                acc = acc + 1;
            }
            x = x - 1;
        }
        return acc;
                ",
                    24,
                );
            });

            test!(nested_loops {
                test_codegen_mainfunc(
                    r"
        int acc = 0;
        int x = 100;
        while (x) {
            int y = 10;
            x = x - y;
            while (y) {
                acc = acc + 1;
                y = y - 1;
            }
        }
        return acc == 100 && x == 0;
                ",
                    1,
                );
            });

            test!(labeled_body {
                test_codegen_mainfunc(
                    r"
    int result = 0;
    goto label;
    while (0)
    label: { result = 1; }

    return result;
                ",
                    1,
                );
            });

            test!(condition_postfix {
                test_codegen_mainfunc(
                    r"
    int i = 100;
    int count = 0;
    while (i--) count++;
    if (count != 100)
        return 0;
    i = 100;
    count = 0;
    while (--i) count++;
    if (count != 99)
        return 0;
    return 1;
                ",
                    1,
                );
            });

            mod fail {
                use super::*;

                test!(declaration_in_body {
                    test_codegen_mainfunc_failure(
                        r"
    while (1)
        int i = 0;
    return 0;
                    ",
                    );
                });

                test!(declaration_in_condition {
                    test_codegen_mainfunc_failure(
                        r"
    while(int a) {
        2;
    }
                    ",
                    );
                });

                test!(missing_parentheses {
                    test_codegen_mainfunc_failure(
                        r"
    while 1 {
        return 0;
    }
                    ",
                    );
                });
            }
        }

        mod do_while_loop {
            use super::*;

            test!(simple {
                test_codegen_mainfunc(
                    r"
    int a = 1;
    do {
        a = a * 2;
    } while(a < 11);

    return a;
                ",
                    16,
                );
            });

            test!(break_immediate {
                test_codegen_mainfunc(
                    r"
    int a = 10;
    do
        break;
    while ((a = 1));
    return a;
                ",
                    10,
                );
            });

            test!(no_body {
                test_codegen_mainfunc(
                    r"
    int i = 502;
    do ; while ((i = i - 5) >= 256);

    return i;
                ",
                    252,
                );
            });

            test!(multi_continue_same_loop {
                test_codegen_mainfunc(
                    r"
    int x = 10;
    int y = 0;
    int z = 0;
    do {
        z = z + 1;
        if (x <= 0)
            continue;
        x = x - 1;
        if (y >= 10)
            continue;
        y = y + 1;
    } while (z != 50);
    return z == 50 && x == 0 && y == 10;
                ",
                    1,
                );
            });

            mod fail {
                use super::*;

                test!(semicolon_after_body {
                    test_codegen_mainfunc_failure(
                        r"
    do {
        int a;
    }; while(1);
    return 0;
                    ",
                    );
                });

                test!(missing_final_semicolon {
                    test_codegen_mainfunc_failure(
                        r"
    do {
        4;
    } while(1)
    return 0;
                    ",
                    );
                });

                test!(empty_condition {
                    test_codegen_mainfunc_failure(
                        r"
    do
        1;
    while ();
    return 0;
                    ",
                    );
                });

                test!(variable_in_body_not_in_scope_for_condition {
                    test_codegen_mainfunc_failure(
                        r"
    do {
        int a = a + 1;
    } while (a < 100);
                    ",
                    );
                });
            }
        }

        test!(labeled_loops {
            test_codegen_mainfunc(
                r"
    int sum = 0;
    goto do_label;
    return 0;

do_label:
    do {
        sum = 1;
        goto while_label;
    } while (1);

while_label:
    while (1) {
        sum = sum + 1;
        goto break_label;
        return 0;
    break_label:
        break;
    };
    goto for_label;
    return 0;

for_label:
    for (int i = 0; i < 10; i = i + 1) {
        sum = sum + 1;
        goto continue_label;
        return 0;
    continue_label:
        continue;
        return 0;
    }
    return sum;
            ",
                12,
            );
        });

        mod fail {
            use super::*;

            test!(label_is_not_block {
                test_codegen_mainfunc_failure(
                    r"
    int a = 0;
    int b = 0;
    // a label does not start a new block, so you can't use it
    // to delineate a multi-statement loop body
    do
    do_body:
        a = a + 1;
        b = b - 1;
    while (a < 10)
        ;
    return 0;
                ",
                );
            });

            test!(duplicate_label_in_body {
                test_codegen_mainfunc_failure(
                    r"
    do {
        // make sure our label-validation analysis also traverses loop bodies
    lbl:
        return 1;
    lbl:
        return 2;
    } while (1);
    return 0;
                ",
                );
            });
        }
    }

    mod switch {
        use super::*;

        test!(simple {
            test_codegen_mainfunc(
                r"
    switch(3) {
        case 0: return 0;
        case 1: return 1;
        case 3: return 3;
        case 5: return 5;
    }
            ",
                3,
            );
        });

        test!(simple_with_breaks {
            test_codegen_mainfunc(
                r"
    int a = 5;
    switch (a) {
        case 5:
            a = 10;
            break;
        case 6:
            a = 0;
            break;
    }
    return a;
            ",
                10,
            );
        });

        test!(simple_with_default {
            test_codegen_mainfunc(
                r"
    int a = 0;
    switch(a) {
        case 1:
            return 1;
        case 2:
            return 9;
        case 4:
            a = 11;
            break;
        default:
            a = 22;
    }
    return a;
            ",
                22,
            );
        });

        test!(case_expressions {
            test_codegen_mainfunc(
                r"
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
            ",
                7,
            );
        });

        test!(default_fallthrough_out_of_order {
            test_codegen_mainfunc(
                r"
// test that we can fall through from default to other cases
// if default isn't last
    int a = 5;
    switch(0) {
        default:
            a = 0;
        case 1:
            return a;
    }
    return a + 1;
            ",
                0,
            );
        });

        test!(only_default {
            test_codegen_mainfunc(
                r"
    int a = 1;
    switch(a) default: return 1;
    return 0;
            ",
                1,
            );
        });

        test!(single_case {
            test_codegen_mainfunc(
                r"
    int a = 1;
    // a switch statement body may be a single case
    switch(a) case 1: return 1;
    return 0;
            ",
                1,
            );
        });

        test!(execute_condition_empty_body {
            test_codegen_mainfunc(
                r"
    int x = 10;
    // two versions of empty switch statements;
    // in both , we execute the controlling expression even though there's
    // nothing to execute in the body
    switch(x = x + 1) {

    }
    switch(x = x + 1)
    ;
    return x;
            ",
                12,
            );
        });

        test!(skip_body_no_cases {
            test_codegen_mainfunc(
                r"
    // if a switch statement body contains no case statements,
    // nothing in it will be executed
    int a = 4;
    switch(a)
        return 0;
    return a;
            ",
                4,
            );
        });

        test!(fallthrough {
            test_codegen_mainfunc(
                r"
    int a = 4;
    int b = 9;
    int c = 0;
    switch (a ? b : 7) {
        case 0:
            return 5;
        case 7:
            c = 1;
        case 9:
            c = 2;
        case 1:
            c = c + 4;
    }
    return c;
            ",
                6,
            );
        });

        test!(goto_middle_of_case {
            test_codegen_mainfunc(
                r"
    int a = 0;
    // a goto statement can jump to any point in a switch statement, including the middle of a case
    goto mid_case;
    switch (4) {
        case 4:
            a = 5;
        mid_case:
            a = a + 1;
            return a;
    }
    return 100;
            ",
                1,
            );
        });

        test!(assign_in_condition {
            test_codegen_mainfunc(
                r"
    int a = 0;
    switch (a = 1) {
        case 0:
            return 10;
        case 1:
            a = a * 2;
            break;
        default:
            a = 99;
    }
    return a;
            ",
                2,
            );
        });

        test!(loop_break_in_switch {
            test_codegen_mainfunc(
                r"
    int cond = 10;
    switch (cond) {
        case 1:
            return 0;
        case 10:
            for (int i = 0; i < 5; i = i + 1) {
                cond = cond - 1;
                if (cond == 8)
                    // make sure this breaks out of loop,
                    // not switch
                    break;
            }
            return 123;
        default:
            return 2;
    }
    return 3;
            ",
                123,
            );
        });

        test!(switch_break_in_loop {
            test_codegen_mainfunc(
                r"
    int acc = 0;
    int ctr = 0;
    for (int i = 0; i < 10; i = i + 1)  {
        // make sure break statements here break out of switch but not loop
        switch(i) {
            case 0:
                acc = 2;
                break;
            case 1:
                acc = acc * 3;
                break;
            case 2:
                acc = acc * 4;
                break;
            default:
                acc = acc + 1;
        }
        ctr = ctr + 1;
    }

    return ctr == 10 && acc == 31;
            ",
                1,
            );
        });

        test!(loop_in_switch_continue {
            test_codegen_mainfunc(
                r"
    switch(4) {
        case 0:
            return 0;
        case 4: {
            int acc = 0;
            // make sure we can use continue inside a loop
            // inside a switch
            for (int i = 0; i < 10; i = i + 1) {
                if (i % 2)
                    continue;
                acc = acc + 1;
            }
            return acc;
        }
    }
    return 0;
            ",
                5,
            );
        });

        test!(switch_in_loop_continue {
            test_codegen_mainfunc(
                r"
    int sum = 0;
    for (int i = 0; i < 10; i = i + 1) {
        switch(i % 2) {
            // make sure continue in switch in loop is permitted
            case 0: continue;
            default: sum = sum + 1;
        }
    }
    return sum;
            ",
                5,
            );
        });

        test!(skip_initializer {
            test_codegen_mainfunc(
                r"
    int a = 3;
    int b = 0;
    switch(a) {
        // a is in scope but we skip its initializer
        int a = (b = 5);
    case 3:
        a = 4;
        b = b + a;
    }

    // make sure case was executed but initializer (b = 5) was not
    return a == 3 && b == 4;
            ",
                1,
            );
        });

        test!(nested_cases {
            test_codegen_mainfunc(
                r"
    int switch1 = 0;
    int switch2 = 0;
    int switch3 = 0;
    switch(3) {
        case 0: return 0;
        case 1: if (0) {
            case 3: switch1 = 1; break;
        }
        default: return 0;
    }
    switch(4) {
        case 0: return 0;
        if (1) {
            return 0;
        } else {
            case 4: switch2 = 1; break;
        }
        default: return 0;
    }
    switch (5) {
        for (int i = 0; i < 10; i = i + 1) {
            switch1 = 0;
            case 5: switch3 = 1; break;
            default: return 0;
        }
    }

    return (switch1 && switch2 && switch3);
            ",
                1,
            );
        });

        test!(nested_switch {
            test_codegen_mainfunc(
                r"
    int a = 0;
    // outer switch will execute default, not nested 'case 0'
    switch(a) {
        case 1:
            switch(a) {
                case 0: return 0;
                default: return 0;
            }
        default: a = 2;
    }
    return a;
            ",
                2,
            );
        });

        test!(outer_and_inner {
            test_codegen_mainfunc(
                r"
    // a switch statement cannot jump to cases in a nested switch statement;
    // here we execute both outer and inner cases
    switch(3) {
        case 0:
            return 1;
        case 3: {
            switch(4) {
                case 3: return 2;
                case 4: return 3;
                default: return 4;
            }
        }
        case 4: return 5;
        default: return 6;
    }
            ",
                3,
            );
        });

        mod fail {
            use super::*;

            test!(case_declaration {
                test_codegen_mainfunc_failure(
                    r"
    switch(3) {
        case 3:
            int i = 0;
            return i;
    }
    return 0;
                ",
                );
            });

            test!(goto_case_label {
                test_codegen_mainfunc_failure(
                    r"
    goto 3;
    switch (3) {
        case 3: return 0;
    }
                ",
                );
            });

            test!(missing_condition_parentheses {
                test_codegen_mainfunc_failure(
                    r"
    switch 3 {
        case 3: return 0;
    }
                ",
                );
            });

            test!(missing_condition {
                test_codegen_mainfunc_failure(
                    r"
    switch {
        return 0;
    }
                ",
                );
            });

            test!(continue_in_case {
                test_codegen_mainfunc_failure(
                    r"
    int a = 3;
    switch(a + 1) {
        case 0:
            // continue can only break out of loops, not switch statements
            continue;
        default: a = 1;
    }
    return a;
                ",
                );
            });

            test!(case_outside {
                test_codegen_mainfunc_failure(
                    r"
    for (int i = 0; i < 10; i = i + 1) {
        // case statements can only appear inside switch statements
        case 0: return 1;
    }
    return 9;
                ",
                );
            });

            test!(default_continue {
                test_codegen_mainfunc_failure(
                    r"
    int a = 3;
    switch(a + 1) {
        case 0:
            a = 1;
        // make sure the pass that labels loops and checks for invalid
        // break/continue statements traverses default statements
        default: continue;
    }
    return a;
                ",
                );
            });

            test!(default_outside {
                test_codegen_mainfunc_failure(
                    r"
    {
        // case statements can only appear inside switch statements
        default: return 0;
    }
                ",
                );
            });

            test!(different_cases_same_scope {
                test_codegen_mainfunc_failure(
                    r"
    int a = 1;
    switch (a) {
        case 1:;
            int b = 10;
            break;

        case 2:;
            // invalid redefinition, because we're in the same scope
            // as declaration of b above
            int b = 11;
            break;

        default:
            break;
    }
    return 0;
                ",
                );
            });

            test!(duplicate_case_constants {
                test_codegen_mainfunc_failure(
                    r"
    switch(4) {
        case 5: return 0;
        case 4: return 1;
        case 5: return 0; // duplicate of previous case 5
        default: return 2;
    }
                ",
                );
            });

            test!(duplicate_case_expressions {
                test_codegen_mainfunc_failure(
                    r"
    switch(4) {
        case 5: return 0;
        case 4: return 1;
        case ((4 * 100 - 395) << 1) >> 1: return 0; // duplicate of previous case 5
        default: return 2;
    }
                ",
                );
            });

            test!(duplicate_case_in_labeled_switch {
                test_codegen_mainfunc_failure(
                    r"
    // make sure our validation of switch statements also traverses labeled
    // statements
    int a = 0;
label:
    switch (a) {
        case 1:
        case 1:
            break;
    }
    return 0;
                ",
                );
            });

            test!(nested_duplicate_case {
                test_codegen_mainfunc_failure(
                    r"
    int a = 10;
    switch (a) {
        case 1: {
            if(1) {
                case 1: // duplicate of previous 'case 1'
                return 0;
            }
        }
    }
    return 0;
                ",
                );
            });

            test!(duplicate_default {
                test_codegen_mainfunc_failure(
                    r"
    int a = 0;
    switch(a) {
        case 0: return 0;
        default: return 1;
        case 2: return 2;
        // can't have two default statements in same enclosing switch
        default: return 2;
    }
                ",
                );
            });

            test!(nested_duplicate_default {
                test_codegen_mainfunc_failure(
                    r"
    int a = 10;
    switch (a) {
        case 1:
        for (int i = 0; i < 10; i = i + 1) {
            continue;
            while(1)
            default:;
        }
        case 2:
        return 0;
        default:;
    }
    return 0;
                ",
                );
            });

            test!(duplicate_label_in_default {
                test_codegen_mainfunc_failure(
                    r"
        int a = 1;
label:

    switch (a) {
        case 1:
            return 0;
        default:
        label:
            return 1;
    }
    return 0;
                ",
                );
            });

            test!(duplicate_variable {
                test_codegen_mainfunc_failure(
                    r"
    int a = 1;
    switch (a) {
        // variable resolution must process this even though it's not reachable;
        // it still declares the variable/brings it into scope
        int b = 2;
        case 0:
            a = 3;
            int b = 2;  // error - duplicate declaration
    }
    return 0;
                ",
                );
            });

            test!(non_constant_case {
                test_codegen_mainfunc_failure(
                    r"
    int a = 3;
    switch(a + 1) {
        case 0: return 0;
        case a: return 1; // case statement values must be constant
        case 1: return 2;
    }
                ",
                );
            });

            test!(undeclared_variable_in_case {
                test_codegen_mainfunc_failure(
                    r"
    int a = 10;
    switch (a) {
        case 1:
            return b;
            break;

        default:
            break;
    }
    return 0;
                ",
                );
            });

            test!(undeclared_variable_in_default {
                test_codegen_mainfunc_failure(
                    r"
    int a = 10;
    switch (a) {
        case 1:
            break;

        default:
            return b;
            break;
    }
    return 0;
                ",
                );
            });

            test!(undeclared_label_in_case {
                test_codegen_mainfunc_failure(
                    r"
    int a = 3;
    switch (a) {
        case 1: goto foo;
        default: return 0;
    }
    return 0;
                ",
                );
            });

            test!(undeclared_label_in_default {
                test_codegen_mainfunc_failure(
                    r"
    int a = 3;
    switch (a) {
        default: goto foo;
        case 1: return 0;
    }
    return 0;
                ",
                );
            });
        }
    }

    #[test]
    #[ignore]
    fn wrong_func_arg_count() {
        validate_error_count(
            r"int blah(int x, int y)
{
    return 5;
}

int main() {
    while (blah()) { }
    do { blah(); } while (blah());
    if (blah()) { blah(); } else { blah(); }
    int x = blah();
    x = blah();
    for (int y = blah(); blah(); blah()) { blah(); }
    for (blah(); blah(); blah()) { blah(); }
    blah();

    int x = -blah();
    x = blah(blah(10), blah(), blah());
    return blah() + blah();
}",
            24,
        );
    }

    #[test]
    #[ignore]
    fn recursive_function() {
        codegen_run_and_check_exit_code(
            r"
int sigma(int x) {
    if (x == 0) {
        return 0;
    }

    return x + sigma(x - 1);
}

int main() {
    return sigma(10);
}
",
            55,
        );
    }

    #[test]
    #[ignore]
    fn nested_function_arg_counts() {
        codegen_run_and_check_exit_code(
            r"
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
",
            56,
        );
    }

    #[test]
    #[ignore]
    fn nested_block_variable_allocation() {
        codegen_run_and_check_exit_code(
            r"
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
",
            21,
        );
    }

    #[test]
    #[ignore]
    fn parameter_redefinition() {
        test_codegen_mainfunc_failure(
            r"int blah(int x)
{
    int x;
    return 5;
}

int main() {
    return 1;
}",
        );

        test_codegen_mainfunc_failure(
            r"int blah(int x)
{
    {
        int x;
        return 5;
    }
}

int main() {
    return 1;
}",
        );
    }
}
