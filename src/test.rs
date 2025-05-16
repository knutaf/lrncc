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

            // Trim whitespace because it might include newlines that don't convey properly in the test file.
            let expected_stdout = expected_stdout.trim();
            let output = output.trim();

            assert_eq!(expected_stdout, output);
            if expected_stdout != output {
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
            Ok((_ast, _global_tracking)) => {
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

    fn test_codegen_mainfunc_failure(body: &str) {
        codegen_run_and_expect_compile_failure(&format!("int main() {{\n{}\n}}", body))
    }

    fn test_library(client_code: &str, library_code: &str, expected_result: (i32, Option<&str>)) {
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
            Some(expected_result),
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
            expected_result,
        ) {
            debug!("cleaning up temp dir {}", temp_dir.to_string_lossy());
            std::fs::remove_dir_all(&temp_dir);
        }
    }

    // No callers in this file, but it's used by the autogenerated tests from build.rs.
    fn run_test_from_file(filename: &str) {
        lazy_static! {
            static ref MODE_REGEX: Regex =
                Regex::new(r"^// Mode: (\w+)$").expect("failed to compile regex");
            static ref EXITCODE_REGEX: Regex =
                Regex::new(r"^// ExitCode: (-?\w+)$").expect("failed to compile regex");
            static ref OUTPUT_REGEX: Regex =
                Regex::new(r"^// Output: (.+)$").expect("failed to compile regex");
        }

        fn try_parse_mode(line: &str) -> Option<&str> {
            let Some((_, [mode_str])) = MODE_REGEX.captures_iter(line).map(|c| c.extract()).next()
            else {
                return None;
            };

            Some(mode_str)
        }

        fn parse_success_result<'l>(
            filename: &str,
            mode_str: &str,
            lines: &mut impl Iterator<Item = &'l str>,
        ) -> (i32, Option<&'l str>) {
            assert!(mode_str == "success" || mode_str == "success_and_output");

            // When this function is called, the next line should contain the ExitCode.
            let line = lines.next().unwrap();
            let (_, [exit_code_str]) = EXITCODE_REGEX
                .captures_iter(line)
                .map(|c| c.extract())
                .next()
                .expect(&format!(
                    "Failed to find ExitCode in test file {}!",
                    filename
                ));
            let expected_exit_code = parse_int::parse::<i32>(exit_code_str).expect(&format!(
                "Failed to parse ExitCode {} as a string, in test file {}!",
                exit_code_str, filename
            ));

            // The "success_and_output" mode also has the Output specified on the next line.
            let output_opt = if mode_str == "success_and_output" {
                let line = lines.next().unwrap();
                let (_, [output_str]) = OUTPUT_REGEX
                    .captures_iter(line)
                    .map(|c| c.extract())
                    .next()
                    .expect(&format!("Failed to find Output in test file {}!", filename));
                Some(output_str)
            } else {
                None
            };

            (expected_exit_code, output_opt)
        }

        let contents = std::fs::read_to_string(filename).unwrap();

        // Each test file is a regular C program, but starts with commented lines that tell the test harness about how
        // to run it and other configuration.
        //
        // The first line must contain Mode: <success|success_and_output|fail|library>
        // For failure tests, we'll try compiling it and expect compile failure. For success, we'll compile and run it.
        // When success, the next line must have ExitCode: <num>
        // When success_and_output, the next line has ExitCode and the line after that has Output.
        //
        // We'll check for the expected exit code and output when the program runs.
        let mut lines = contents.lines();
        let line = lines.next().unwrap();

        let mode_str =
            try_parse_mode(line).expect(&format!("Failed to find Mode in test file {}!", filename));

        match mode_str {
            "fail" => {
                codegen_run_and_expect_compile_failure(&contents);
            }
            "success" | "success_and_output" => {
                let expected_result = parse_success_result(filename, mode_str, &mut lines);
                let _ = codegen_run_and_check_exit_code_and_output_or_compile_failure(
                    &contents,
                    &[],
                    Some(expected_result),
                );
            }
            "library" => {
                // The "library" type test contains two code portions:
                // - the "library", which is compiled into a lib file
                // - the "client", which links with the lib file and accesses symbols exported by it
                let mut library_contents = String::new();

                // Starting directly after the first Mode declares library, the subsequent lines are part of the library
                // code.
                let mode_str = loop {
                    let line = lines.next().unwrap();

                    // Keep accumulating lines until the next Mode.
                    let Some(mode_str) = try_parse_mode(line) else {
                        library_contents.push_str(line);
                        library_contents.push_str("\r\n");
                        continue;
                    };

                    break mode_str;
                };

                // The next mode needs to be a regular success type, and indicates the start of the client code.
                match mode_str {
                    "success" | "success_and_output" => {
                        let expected_result = parse_success_result(filename, mode_str, &mut lines);

                        let mut client_contents = String::new();
                        for line in lines {
                            client_contents.push_str(line);
                            client_contents.push_str("\r\n");
                        }

                        test_library(&client_contents, &library_contents, expected_result);
                    }
                    _ => panic!("Invalid Mode {} for library test", mode_str),
                }
            }
            _ => {
                panic!("unknown test mode {} in test file {}", mode_str, filename);
            }
        }
    }

    // Include all the tests generated by build.rs.
    mod generated {
        use super::*;
        include!(concat!(env!("OUT_DIR"), "/test.rs"));
    }

    mod fail {
        use super::*;

        fn test_invalid_identifier(ident: &str) {
            test_codegen_mainfunc_failure(&format!("int {0}; return {0};", &ident));
        }

        test!(keywords_as_var_identifier {
            for keyword in &KEYWORDS {
                test_invalid_identifier(keyword);
            }
        });
    }

    mod functions {
        use super::*;

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
