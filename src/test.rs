use {
    crate::*,
    std::{io::Read, path},
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

    fn run_and_check_exit_code_and_output(
        exe_path: &str,
        (expected_exit_code, expected_stdout_opt): (i32, Option<&str>),
    ) -> bool {
        let mut result = true;

        let mut child = Command::new(exe_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect(&format!("failed to run {}", exe_path));

        let actual_exit_code = child
            .wait()
            .expect("child process must exit")
            .code()
            .expect("all processes must have exit code");

        assert_eq!(expected_exit_code, actual_exit_code);
        if expected_exit_code != actual_exit_code {
            result = false;
        }

        if let Some(expected_stdout) = expected_stdout_opt {
            let mut output = String::new();
            let _ = child
                .stdout
                .unwrap()
                .read_to_string(&mut output)
                .expect("failed to get child output");

            assert_eq!(expected_stdout, &output);
            if expected_stdout != &output {
                result = false;
            }
        }

        result
    }

    fn codegen_run_and_check_exit_code_and_output_or_compile_failure(
        input: &str,
        extra_link_args: &[String],
        expected_result_opt: Option<(i32, Option<&str>)>,
    ) -> bool {
        let mut result = true;

        let args = LcArgs {
            input_path: String::new(),
            output_path: Some(format!("test_{}.exe", generate_random_string(8))),
            extra_link_args: Vec::from(extra_link_args),
            mode: Mode::All,
            verbose: true,
            should_keep_intermediate_files: false,
            should_compile_only: false,
        };

        let compile_result = compile_and_link(&args, input, true, expected_result_opt.is_some());
        if compile_result.is_ok() {
            let exe_path_abs = path::absolute(args.output_path.as_ref().unwrap()).unwrap();
            let exe_path_str = exe_path_abs.to_str().unwrap();
            let mut pdb_path = exe_path_abs.clone();
            pdb_path.set_extension("pdb");

            if let Some(expected_result) = expected_result_opt {
                if run_and_check_exit_code_and_output(exe_path_str, expected_result) {
                    std::fs::remove_file(&exe_path_abs);
                    std::fs::remove_file(&pdb_path);
                } else {
                    result = false;
                }
            } else {
                assert!(false, "compile succeeded but expected failure");
                result = false;
            }
        } else {
            println!("compile failed! {:?}", compile_result);
            assert!(expected_result_opt.is_none());
            if expected_result_opt.is_some() {
                result = false;
            }
        }

        result
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
        let _ = codegen_run_and_check_exit_code_and_output_or_compile_failure(
            input,
            &[],
            Some((expected_exit_code, None)),
        );
    }

    fn codegen_run_and_expect_compile_failure(input: &str) {
        let _ = codegen_run_and_check_exit_code_and_output_or_compile_failure(input, &[], None);
    }

    fn codegen_run_and_check_exit_code_and_output(
        input: &str,
        expected_exit_code: i32,
        expected_stdout: &str,
    ) {
        let _ = codegen_run_and_check_exit_code_and_output_or_compile_failure(
            input,
            &[],
            Some((expected_exit_code, Some(expected_stdout))),
        );
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

    fn test_library(client_code: &str, library_code: &str, expected_exit_code: i32) {
        full_test_library(client_code, library_code, expected_exit_code, None)
    }

    fn test_library_with_output(
        client_code: &str,
        library_code: &str,
        expected_exit_code: i32,
        expected_stdout: &str,
    ) {
        full_test_library(
            client_code,
            library_code,
            expected_exit_code,
            Some(expected_stdout),
        )
    }

    fn full_test_library(
        client_code: &str,
        library_code: &str,
        expected_exit_code: i32,
        expected_stdout_opt: Option<&str>,
    ) {
        fn compile_with_standard_compiler(
            code: &str,
            temp_dir: &Path,
            output_filename: &str,
            extra_link_args_opt: Option<&[String]>,
        ) -> PathBuf {
            let code_path = temp_dir.join("code.c");
            let output_path = temp_dir.join(output_filename);

            std::fs::write(&code_path, code)
                .map_err(format_io_err)
                .unwrap();

            let mut args = vec![];

            args.push(String::from("/Zi"));
            args.push(String::from(
                path::absolute(code_path)
                    .unwrap()
                    .to_str()
                    .expect("failed to format code path as string"),
            ));

            // If link args were supplied, even empty ones, then it means we're intended to link and produce an
            // executable. Otherwise only compile to an object file.
            if let Some(extra_link_args) = extra_link_args_opt {
                args.push(format!("/Fe{}", output_filename));

                args.push(String::from("/link"));
                args.push(String::from("/nodefaultlib:libcmt"));
                args.push(String::from("msvcrt.lib"));
                for arg in extra_link_args.iter() {
                    args.push(arg.clone());
                }
            } else {
                args.push(String::from("/c"));
                args.push(format!("/Fo{}", output_filename));
            }

            let mut command = Command::new("cl.exe");

            command
                .args(&args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .current_dir(&temp_dir);

            debug!(
                "compile_with_standard_compiler command: {}",
                format_command_args(&command)
            );

            let status = command.status().expect("failed to run cl.exe");

            if !status.success() {
                error!("cl failed with {:?}", status);
            }

            path::absolute(output_path).unwrap()
        }

        // First compile the library with the standard compiler and compile the client code with this compiler.
        let temp_dir_name = format!("testrun_{}", generate_random_string(8));
        let temp_dir = Path::new(&temp_dir_name);
        std::fs::create_dir_all(&temp_dir);

        let obj_path = compile_with_standard_compiler(library_code, &temp_dir, "library.obj", None);

        if codegen_run_and_check_exit_code_and_output_or_compile_failure(
            client_code,
            &[String::from(obj_path.to_str().unwrap())],
            Some((expected_exit_code, expected_stdout_opt)),
        ) {
            debug!("cleaning up temp dir {}", temp_dir.to_string_lossy());
            std::fs::remove_dir_all(&temp_dir);
        } else {
            return;
        }

        let temp_dir_name = format!("testrun_{}", generate_random_string(8));
        let temp_dir = Path::new(&temp_dir_name);
        std::fs::create_dir_all(&temp_dir);

        // Now compile the library code with our compiler and the client with the standard compiler and make sure this
        // works too.
        let args = LcArgs {
            input_path: String::new(),
            output_path: Some(String::from(
                path::absolute(temp_dir.join("library.obj"))
                    .unwrap()
                    .to_str()
                    .expect("could not convert path to str"),
            )),
            extra_link_args: vec![],
            mode: Mode::All,
            verbose: true,
            should_keep_intermediate_files: false,
            should_compile_only: true,
        };

        assert!(compile_and_link(&args, library_code, true, true).is_ok());

        let exe_path = compile_with_standard_compiler(
            client_code,
            &temp_dir,
            "output.exe",
            Some(&vec![args.output_path.unwrap().clone()]),
        );

        if run_and_check_exit_code_and_output(
            exe_path.to_str().expect("could not convert path to str"),
            (expected_exit_code, expected_stdout_opt),
        ) {
            debug!("cleaning up temp dir {}", temp_dir.to_string_lossy());
            std::fs::remove_dir_all(&temp_dir);
        }
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

        fn test_invalid_identifier(ident: &str) {
            test_codegen_mainfunc_failure(&format!("int {0}; return {0};", &ident));
        }

        test!(keywords_as_var_identifier {
            for keyword in &KEYWORDS {
                test_invalid_identifier(keyword);
            }
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

    mod functions {
        use super::*;

        test!(nested_function_arg_counts {
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
        });

        test!(nested_block_variable_allocation {
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
        });

        test!(compound_assign_function_result {
            codegen_run_and_check_exit_code(
                r"
int foo(void) {
    return 2;
}

int main(void) {
    int x = 3;
    x -= foo();
    return x;
}
    ",
                1,
            );
        });

        test!(preserve_ecx_bitwise_shift {
            codegen_run_and_check_exit_code(
                r"
/* Make sure we don't clobber argument passed in ECX register by
 * performing a bitwise shift operation that uses that register */

int x(int a, int b, int c, int d, int e, int f) {
    return a == 1 && b == 2 && c == 3 && d == 4 && e == 5 && f == 6;
}

int main(void) {
    int a = 4;
    return x(1, 2, 3, 4, 5, 24 >> (a / 2));
}
    ",
                1,
            );
        });

        test!(label_reuse_in_functions {
            codegen_run_and_check_exit_code(
                r"
/* The same label can be used in multiple functions */
int foo(void) {
    goto label;
    return 0;
    label:
        return 5;
}

int main(void) {
    goto label;
    return 0;
    label:
        return foo();
}
    ",
                5,
            );
        });

        test!(label_same_as_function_name {
            codegen_run_and_check_exit_code(
                r"
/* The same identifier can be used
 * in the same scope as both a function name and a label */
int foo(void) {
    goto foo;
    return 0;
    foo:
        return 1;
}

int main(void) {
    return foo();
}
    ",
                1,
            );
        });

        test!(label_naming_scheme {
            codegen_run_and_check_exit_code(
                r##"
// We need to transform labels to avoid conflicts between identical labels
// in different functions. This test case tries to catch a few
// obvious-but-unsafe naming schemes: transforming label "lbl" in function "fun"
// into "lblfun," "lbl_fun", "funlbl" or "fun_lbl".
// Here we just want to see that the program compiles successfully (rather than
// hitting an assembler error b/c of duplicate labels in assembly).

int main(void) {
    // If we combine function name and label with no separator,
    // this will conflict with "label" in the "main_" function below
    // (both will be main_label)
    // If we combine function name and label with a _ separator, they'll still conflict;
    // both will be main__label
    _label:

    // If we add function name to end of label, this will conflict with "label"
    // in _main below (both will be label_main or label__main)
    label_:
    return 0;
}

int main_(void) {
    label:
    return 0;
}

int _main(void) {
    label: return 0;
}
    "##,
                0,
            );
        });

        test!(stack_alignment {
        const STACK_CHECK_ASM_CODE: &'static str = r##"
INCLUDELIB msvcrt.lib

.DATA
.CODE

check_stack_alignment_even PROC
    mov rax,rsp

    ; Get the low 4 bits. If any of these are set, then it's not a multiple of 16.
    and rax,15

    ; But we're checking *after* the call instruction, so rsp is expected to be off by one pointer size--the return address.
    sub rax,8

    ; At this point, if it was 16-byte aligned plus one pointer size, the return value is 0.
    ret
check_stack_alignment_even ENDP

check_stack_alignment_odd = check_stack_alignment_even
PUBLIC check_stack_alignment_odd

END
"##;

                let temp_dir_name = format!("testrun_{}", generate_random_string(8));
                let temp_dir = Path::new(&temp_dir_name);
                std::fs::create_dir_all(&temp_dir);

                let stack_check_asm_path = path::absolute(temp_dir.join("stack_check.asm")).unwrap();
                let stack_check_obj_path = path::absolute(temp_dir.join("stack_check.obj")).unwrap();

                let _ = std::fs::write(&stack_check_asm_path, STACK_CHECK_ASM_CODE).unwrap();

                let exit_code = assemble_and_link(
                    &stack_check_asm_path,
                    stack_check_obj_path.to_str().unwrap(),
                    None,
                    true,
                    &temp_dir,
                )
                .expect("programs should always have an exit code");

                assert_eq!(exit_code, 0);

                codegen_run_and_check_exit_code_and_output_or_compile_failure(
                    r"
/* Call functions with both even and odd numbers of stack arguments,
 * to make sure the stack is correctly aligned in both cases.
 */

int check_stack_alignment_even(int a, int b, int c, int d, int e, int f, int g, int h);
int check_stack_alignment_odd(int a, int b, int c, int d, int e, int f, int g, int h, int i);

int check_even_arguments(int a, int b, int c, int d, int e, int f, int g, int h) {
    return a == 1 &&
           b == 2 &&
           c == 3 &&
           d == 4 &&
           e == 5 &&
           f == 6 &&
           g == 7 &&
           h == 8;
}

int check_odd_arguments(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
    return a == 1 &&
           b == 2 &&
           c == 3 &&
           d == 4 &&
           e == 5 &&
           f == 6 &&
           g == 7 &&
           h == 8 &&
           i == 9;
}

int main(void) {
    /* Allocate an argument on the stack, to check that
     * we properly account for already-allocated stack space
     * when deciding how much padding to add
     */
    int x = 3;
    if (check_stack_alignment_even(1, 2, 3, 4, 5, 6, 7, 8) != 0) { return 0; }
    if (check_even_arguments(1, 2, 3, 4, 5, 6, 7, 8) != 1) { return 0; }
    if (check_stack_alignment_odd(1, 2, 3, 4, 5, 6, 7, 8, 9) != 0) { return 0; }
    if (check_odd_arguments(1, 2, 3, 4, 5, 6, 7, 8, 9) != 1) { return 0; }
    // return x to make sure it hasn't been clobbered
    return x;
}
    ",
                    &[String::from(stack_check_obj_path.to_str().unwrap())],
                    Some((3, None)),
                );

                std::fs::remove_dir_all(&temp_dir);
            });

        mod no_args {
            use super::*;

            test!(multiple_declarations {
                codegen_run_and_check_exit_code(
                    r"
int main(void) {
    int f(void);
    int f(void);
    return f();
}

int f(void) {
    return 3;
}
        ",
                    3,
                );
            });

            test!(no_return_value {
                codegen_run_and_check_exit_code(
                    r"
int foo(void) {
    /* It's legal for a non-void function to not return a value.
     * If the caller tries to use the value of the function, the result is undefined.
     */
    int x = 1;
}

int main(void) {
    /* This is well-defined because we call foo but don't use its return value */
    foo();
    return 3;
}
        ",
                    3,
                );
            });

            test!(function_unary_precedence {
                codegen_run_and_check_exit_code(
                    r"
int three(void) {
    return 3;
}

int main(void) {
    /* The function call operator () is higher precedence
     * than unary prefix operators
     */
    return !three();
}
        ",
                    0,
                );
            });

            test!(call_in_expression {
                codegen_run_and_check_exit_code(
                    r"
int bar(void) {
    return 9;
}

int foo(void) {
    return 2 * bar();
}

int main(void) {
    /* Use multiple function calls in an expression,
     * make sure neither overwrites the other's return value in EAX */
    return foo() + bar() / 3;
}
        ",
                    21,
                );
            });

            test!(variable_shadows_function {
                codegen_run_and_check_exit_code(
                    r"
int main(void) {
    int foo(void);

    int x = foo();
    if (x > 0) {
        int foo  = 3;
        x = x + foo;
    }
    return x;
}

int foo(void) {
    return 4;
}
        ",
                    7,
                );
            });
        }

        mod args_in_registers {
            use super::*;

            test!(dont_clobber_edx {
                codegen_run_and_check_exit_code(
                    r"
/* Make sure we don't clobber argument passed in EDX register by
 * performing a division operation that uses that register */

int x(int a, int b, int c, int d, int e, int f) {
    return a == 1 && b == 2 && c == 3 && d == 4 && e == 5 && f == 6;
}

int main(void) {
    int a = 4;
    return x(1, 2, 3, 4, 5, 24 / a);
}
        ",
                    1,
                );
            });

            test!(expression_args {
                codegen_run_and_check_exit_code(
                    r"
int sub(int a, int b) {
    /* Make sure arguments are passed in the right order
     * (we can test this with subtraction since a - b  != b - a)
     */
    return a - b;
}

int main(void) {
    /* Make sure we can evaluate expressions passed as arguments */
    int sum = sub(1 + 2, 1);
    return sum;
}
        ",
                    2,
                );
            });

            test!(recursive {
                codegen_run_and_check_exit_code(
                    r"
int fib(int n) {
    if (n == 0 || n == 1) {
        return n;
    } else {
        return fib(n - 1) + fib(n - 2);
    }
}

int main(void) {
    int n = 6;
    return fib(n);
}
        ",
                    8,
                );
            });

            test!(forward_decl_change_param_names {
                codegen_run_and_check_exit_code(
                    r"
int foo(int a, int b);

int main(void) {
    return foo(2, 1);
}

/* Multiple declarations of a function
 * can use different parameter names
 */
int foo(int x, int y){
    return x - y;
}
        ",
                    1,
                );
            });

            test!(hello_world {
                codegen_run_and_check_exit_code_and_output(
                    r"
int putchar(int c);

int main(void) {
    putchar(72);
    putchar(101);
    putchar(108);
    putchar(108);
    putchar(111);
    putchar(44);
    putchar(32);
    putchar(87);
    putchar(111);
    putchar(114);
    putchar(108);
    putchar(100);
    putchar(33);
}
        ",
                    0,
                    "Hello, World!",
                );
            });

            test!(params_are_preserved {
                codegen_run_and_check_exit_code(
                    r"
/* Make sure that calling another function doesn't clobber
 * arguments to the current function passed in the same registers
 */

int g(int w, int x, int y, int z) {
    if (w == 2 && x == 4 && y == 6 && z == 8)
        return 1;
    return 0;
}

int f(int a, int b, int c, int d) {
    int result = g(a * 2, b * 2, c * 2, d * 2);
    return (result == 1 && a == 1 && b == 2 && c == 3 && d == 4);

}

int main(void) {
    return f(1, 2, 3, 4);
}
        ",
                    1,
                );
            });

            test!(param_shadows_function_name {
                codegen_run_and_check_exit_code(
                    r"
int a(void) {
    return 1;
}

int b(int a) {
    return a;
}

int main(void) {
    return a() + b(2);
}
        ",
                    3,
                );
            });

            test!(param_shadows_own_function_name {
                codegen_run_and_check_exit_code(
                    r"
int a(int a) {
    return a * 2;
}

int main(void) {
    return a(1);
}
        ",
                    2,
                );
            });

            test!(param_shadows_local_variable {
                codegen_run_and_check_exit_code(
                    r"
int main(void) {
    int a = 10;
    // a function declaration is a separate scope,
    // so parameter 'a' doesn't conflict with variable 'a' above
    int f(int a);
    return f(a);
}

int f(int a) {
    return a * 2;
}
        ",
                    20,
                );
            });
        }

        mod stack_args {
            use super::*;

            test!(call_library_function {
                codegen_run_and_check_exit_code_and_output(
                    r"
int putchar(int c);

/* Make sure we can correctly manage calling conventions from the callee side
 * (by accessing parameters, including parameters on the stack) and the caller side
 * (by calling a standard library function) in the same function
 */
int foo(int a, int b, int c, int d, int e, int f, int g, int h) {
    putchar(h);
    return a + g;
}

int main(void) {
    return foo(1, 2, 3, 4, 5, 6, 7, 65);
}
        ",
                    8,
                    "A",
                );
            });

            test!(many_parameters {
                codegen_run_and_check_exit_code(
                    r"
int foo(int a, int b, int c, int d, int e, int f, int g, int h) {
    return (a == 1 && b == 2 && c == 3 && d == 4 && e == 5
            && f == 6 && g == 7 && h == 8);
}

int main(void) {
    return foo(1, 2, 3, 4, 5, 6, 7, 8);
}
        ",
                    1,
                );
            });

            test!(stack_leaks {
                codegen_run_and_check_exit_code(
                    r"
/* Make sure stack arguments are deallocated correctly after returning from a function call; also test passing variables as stack arguments */

int lots_of_args(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, int m, int n, int o) {
    return l + o;
}

int main(void) {
    int ret = 0;
    for (int i = 0; i < 10000000; i = i + 1) {
        ret = lots_of_args(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ret, 13, 14, 15);
    }
    return ret == 150000000;
}
        ",
                    1,
                );
            });
        }

        mod libraries {
            use super::*;

            test!(addition {
                test_library(
                    r"
int add(int x, int y);

int main(void) {
    return add(1, 2);
}
        ",
                    r"
int add(int x, int y) {
    return x + y;
}
        ",
                    3,
                );
            });

            // TODO knutaf fix this
            test!(many_args {
                test_library(
                    r"
int fib(int a);

int multiply_many_args(int a, int b, int c, int d, int e, int f, int g, int h);

int main(void) {
    int x = fib(4); // 3
    // at least until we implement optimizations, seven will have other values
    // adjacent to it in memory, which we'll push onto the stack when we pass it as an arg;
    // this tests that the caller will just look at 7 and not the junk bytes next to it
    int seven = 7;
    int eight = fib(6);
    int y = multiply_many_args(x, 2, 3, 4, 5, 6, seven, eight);
    if (x != 3) {
        return 1;
    }
    if (y != 589680) {
        return 2;
    }
    return x + (y % 256);
}
        ",
                    r"
int fib(int n) {
    if (n == 0 || n == 1) {
        return n;
    } else {
        return fib(n - 1) + fib(n - 2);
    }
}

int multiply_many_args(int a, int b, int c, int d, int e, int f, int g, int h) {

    return a * b * c * d * e * f * fib(g) * fib(h);
}
        ",
                    115,
                );
            });

            test!(std_library_call {
                test_library_with_output(
                    r"
int incr_and_print(int c);

int main(void) {
    incr_and_print(70);
    return 0;
}
        ",
                    r"
int putchar(int c);

int incr_and_print(int b) {
    return putchar(b + 2);
}
        ",
                    0,
                    "H",
                );
            });

            test!(division_clobber_rdx {
                test_library(
                    r"
int f(int a, int b, int c, int d);

int main(void) {
    return f(10, 2, 100, 4);
}
        ",
                    r"
/* Division requires us to use the RDX register;
 * make sure this doesn't clobber the argument passed
 * in this register
 */
int f(int a, int b, int c, int d) {
    // perform division
    int x = a / b;
    // make sure everything has the right value
    if (a == 10 && b == 2 && c == 100 && d == 4 && x == 5)
        return 1;
    return 0;
}
        ",
                    1,
                );
            });

            test!(local_stack_variables {
                test_library(
                    r"
int f(int reg1, int reg2, int reg3, int reg4, int reg5, int reg6,
    int stack1, int stack2, int stack3);

int main(void) {
    return f(1, 2, 3, 4, 5, 6, -1, -2, -3);
}
        ",
                    r"
/* Make sure a called function can correctly access variables on the stack */
int f(int reg1, int reg2, int reg3, int reg4, int reg5, int reg6,
    int stack1, int stack2, int stack3) {
    int x = 10;
    // make sure every variable has the right value
    if (reg1 == 1 && reg2 == 2 && reg3 == 3 && reg4 == 4 && reg5 == 5
        && reg6 == 6 && stack1 == -1 && stack2 == -2 && stack3 == -3
        && x == 10) {
        // make sure we can update the value of one argument
        stack2 = 100;
        return stack2;
    }
    return 0;
}
        ",
                    100,
                );
            });
        }

        mod fail {
            use super::*;

            test!(wrong_func_arg_count {
                validate_error_count(
                    r"int blah(int x, int y)
        {
            return 5;
        }

        int main() {
            while (blah()) { }
            do { } while (blah());
            do { blah(); } while (1);
            if (blah()) { } else { }
            if (1) { blah(); } else { }
            if (1) { } else { blah(); }
            int x = blah();
            x = blah();
            x = -blah();
            for (int y = blah(); ; ) { }
            for (int y = 1; blah(); ) { }
            for (int y = 1; ; blah()) { }
            for (int y = 1; ; ) { blah(); }
            blah();
            x = blah(blah(10), blah());
            return blah() + blah();
        }",
                    18,
                );
            });

            test!(goto_across_functions {
                test_codegen_mainfunc_failure(
                    r"
int foo(void) {
    label:
        return 0;
}

int main(void) {
    /* You can't goto a label in another function */
    goto label;
    return 1;
}
                ",
                );
            });

            test!(goto_function {
                test_codegen_mainfunc_failure(
                    r"
int foo(void) {
    return 3;
}

int main(void) {
    /* You can't use a function name as a goto label */
    goto foo;
    return 3;
}
                ",
                );
            });

            mod invalid_declarations {
                use super::*;

                test!(assign_to_func_call {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    // a function call is not an lvalue
    // NOTE: in later chapters we'll detect this during type checking
    // rather than identifier resolution
    x() = 1;
    return 0;
}
                    ",
                    );
                });

                test!(compound_assign_to_func_call {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    // a function call is not an lvalue
    // NOTE: in later chapters we'll detect this during type checking
    // rather than identifier resolution
    x() += 1;
    return 0;
}
                    ",
                    );
                });

                test!(decrement_func_call {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    // a function call is not an lvalue, so we can't decrement it
    x()--;
}
                    ",
                    );
                });

                test!(increment_func_call {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    // a function call is not an lvalue, so we can't increment it
    ++x();
}
                    ",
                    );
                });

                test!(duplicate_param_name_in_declaration {
                    test_codegen_mainfunc_failure(
                        r"
/* Duplicate parameter names are illegal in function declarations
   as well as definitions */
int foo(int a, int a);

int main(void) {
    return foo(1, 2);
}

int foo(int a, int b) {
    return a + b;
}
                    ",
                    );
                });

                test!(duplicate_param_name_in_definition {
                    test_codegen_mainfunc_failure(
                        r"
/* It's illegal for multiple parameters to a function to have the same name */
int foo(int a, int a) {
    return a;
}

int main(void) {
    return foo(1, 2);
}
                    ",
                    );
                });

                test!(nested_function {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    /* Nested function definitions are not permitted */
    int foo(void) {
        return 1;
    }
    return foo();
}
                    ",
                    );
                });

                test!(redefine_func_as_var {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    /* It's illegal to declare an identifier with external linkage and
     * no linkage in the same scope. Here, the function declaration foo
     * has external linkage and the variable declaration has no linkage.
     * The types here also conflict, but our implementation will catch
     * the linkage error before this gets to the type checker.
     */
    int foo(void);
    int foo = 1;
    return foo;
}

int foo(void) {
    return 1;
}
                    ",
                    );
                });

                test!(redefine_var_as_func {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    /* It's illegal to declare an identifier with external linkage and
     * no linkage in the same scope. Here, the function declaration foo
     * has external linkage and the variable declaration has no linkage.
     * The types here also conflict, but our implementation will catch
     * the linkage error before this gets to the type checker.
     */
    int foo = 1;
    int foo(void);
    return foo;
}

int foo(void) {
    return 1;
}
                    ",
                    );
                });

                test!(redefine_parameter {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a) {
    /* A function's parameter list and its body are in the same scope,
     * so redeclaring a here is illegal. */
    int a = 5;
    return a;
}

int main(void) {
    return foo(3);
}
                    ",
                    );
                });

                test!(undeclared_func {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    /* You must declare a function before using it */
    return foo(3);
}

int foo(int a) {
    return 1;
}
                    ",
                    );
                });

                test!(wrong_param_name {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a);

int main(void) {
    return foo(3);
}

int foo(int x) {
    /* Only the parameter names from this definition are in scope.
     * Parameter names from earlier declarations of foo aren't!
     */
    return a;
}
                    ",
                    );
                });

                test!(call_label {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    int x = 1;
    a:
    x = x + 1;
    a(); // can't call a label like a function
    return x;

}
                    ",
                    );
                });
            }

            mod invalid_parse {
                use super::*;

                test!(call_non_identifier {
                    test_codegen_mainfunc_failure(
                        r"
/* You can only call a function, not a constant.
   Our implementation will reject this during parsing.
   Because the C grammar permits this declaration,
   some compilers may reject it during type checking.
*/

int main(void) {
    return 1();
}
                    ",
                    );
                });

                test!(parameters_wrong_closing_delimiter {
                    test_codegen_mainfunc_failure(
                        r"
/* Make sure a parameter list ends with a closing ) and not some other character;
 * this is a regression test for a bug in the reference implementation */

int foo(int x, int y} { return x + y; }

int main(void) { return 0;}
                    ",
                    );
                });

                test!(arguments_wrong_closing_delimiter {
                    test_codegen_mainfunc_failure(
                        r"
/* Make sure a argument list ends with a closing ) and not some other character;
 * this is a regression test for a bug in the reference implementation */

int foo(int x, int y) {
    return x + y;
}

int main(void) { return foo(1, 2};}
                    ",
                    );
                });

                test!(parameter_declaration_in_call {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a) {
    return 0;
}

int main(void) {
    /* A function argument must be an expression, not a declaration */
    return foo(int a);
}
                    ",
                    );
                });

                test!(return_type_function {
                    test_codegen_mainfunc_failure(
                        r"
/* You cannot declare a function that returns a function.
   Our implementation will reject this during parsing.
   Because the C grammar permits this declaration,
   some compilers may reject it during type checking.
*/
int foo(void)(void);

int main(void) {
    return 0;
}
                    ",
                    );
                });

                test!(declaration_in_for_loop {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    /* Function declarations aren't permitted in for loop headers. */
    for (int f(void); ; ) {
        return 0;
    }
}
                    ",
                    );
                });

                test!(function_with_initializer {
                    test_codegen_mainfunc_failure(
                        r"
/* You can't declare a function with an initializer.
   Our implementation will reject this during parsing.
   Because the C grammar permits this declaration,
   some compilers may reject it during type checking.
*/
int foo(void) = 3;

int main(void) {
    return 0;
}
                    ",
                    );
                });

                test!(arguments_trailing_comma {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a, int b, int c) {
    return a + b + c;
}

int main(void) {
    /* Trailing commas aren't permitted in argument lists */
    return foo(1, 2, 3,);
}
                    ",
                    );
                });

                test!(parameters_trailing_comma {
                    test_codegen_mainfunc_failure(
                        r"
/* Trailing commas aren't permitted in parameter lists */
int foo(int a,) {
    return a + 1;
}

int main(void) {
    return foo(4);
}
                    ",
                    );
                });

                test!(parameters_unclosed_parentheses {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a, int b {
    return 0;
}

int main(void) {
    return 0;
}
                    ",
                    );
                });

                test!(arguments_unclosed_parentheses {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a, int b) {
    return 0;
}

int main(void) {
    return foo(1, 2;
}
                    ",
                    );
                });

                test!(parameter_default_value {
                    test_codegen_mainfunc_failure(
                        r"
/* Variable initializers aren't permitted in parameter lists */
int bad_params(int a = 3) {
    return 1;
}

int main(void) {
    return 0;
}
                    ",
                    );
                });
            }

            mod invalid_types {
                use super::*;

                test!(assign_func_to_variable {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);
int main(void) {
    int a = 10;
    a = x;
    return 0;
}
                    ",
                    );
                });

                test!(assign_to_func {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    int x(void);
    x = 3;
    return 0;
}
                    ",
                    );
                });

                test!(call_variable {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    int x = 0;
    /* x isn't a function, so you can't call it */
    return x();
}
                    ",
                    );
                });

                test!(conflicting_declaration {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a);

int main(void) {
    return 5;
}

/* The forward declaration and definition of 'foo' conflict
 * (different numbers of parameters)
 */
int foo(int a, int b) {
    return 4;
}
                    ",
                    );
                });

                test!(conflicting_local_declaration {
                    test_codegen_mainfunc_failure(
                        r"
int bar(void);

int main(void) {
    /* Two local declarations of foo in 'main' and 'bar' conflict -
     * different numbers of parameters
     */
    int foo(int a);
    return bar() + foo(1);
}

int bar(void) {
    int foo(int a, int b);
    return foo(1, 2);
}
                    ",
                    );
                });

                test!(divide_by_function {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    int a = 10 / x;
    return 0;
}
                    ",
                    );
                });

                test!(multiple_definitions {
                    test_codegen_mainfunc_failure(
                        r"
/* Function 'foo' is defined twice */
int foo(void){
    return 3;
}

int main(void) {
    return foo();
}

int foo(void){
    return 4;
}
                    ",
                    );
                });

                test!(multiple_definitions_with_local {
                    test_codegen_mainfunc_failure(
                        r"
/* Function 'foo' is defined twice */
int foo(void){
    return 3;
}

int main(void) {
    // after seeing this declaration, we should still remember that
    // foo was defined earlier
    int foo(void);
    return foo();
}

int foo(void){
    return 4;
}
                    ",
                    );
                });

                test!(too_few_args {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a, int b) {
    return a + 1;
}

int main(void) {
    /* foo is called with too many arguments */
    return foo(1);
}
                    ",
                    );
                });

                test!(too_many_args {
                    test_codegen_mainfunc_failure(
                        r"
int foo(int a) {
    return a + 1;
}

int main(void) {
    /* foo is called with too many arguments */
    return foo(1, 2);
}
                    ",
                    );
                });

                test!(bitwise_op_function {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    x >> 2;
    return 0;
}
                    ",
                    );
                });

                test!(compound_assign_function {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    x += 3;
    return 0;
}
                    ",
                    );
                });

                test!(compound_assign_function_rhs {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    int a = 3;
    a += x;
    return 0;
}
                    ",
                    );
                });

                test!(postfix_increment_function {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void) {
    x++;
    return 0;
}
                    ",
                    );
                });

                test!(prefix_decrement_function {
                    test_codegen_mainfunc_failure(
                        r"
int x(void);

int main(void){
    --x;
    return 0;
}
                    ",
                    );
                });

                test!(switch_on_function {
                    test_codegen_mainfunc_failure(
                        r"
int main(void) {
    int f(void);
    switch (f)
        return 0;
}
                    ",
                    );
                });
            }
        }
    }

    mod padding {
        use super::*;

        test!(simple {
            for num in 0..=16 {
                info!(num, align = 16, result = pad_to_alignment(num, 16));
                assert_eq!(pad_to_alignment(num, 16), 16);
            }

            for num in 17..=32 {
                info!(num, align = 16, result = pad_to_alignment(num, 16));
                assert_eq!(pad_to_alignment(num, 16), 32);
            }
        });
    }
}
