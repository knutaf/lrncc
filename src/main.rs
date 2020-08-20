use std::env;
use std::fmt;
use std::process::Command;

#[macro_use] extern crate lazy_static;
extern crate regex;
use regex::Regex;

trait AstToString {
    fn ast_to_string(&self, indent_levels : u32) -> String;

    fn get_indent_string(indent_levels : u32) -> String {
        let mut result = String::new();
        for _ in 0 .. indent_levels {
            result += "    ";
        }
        result
    }
}

#[derive(PartialEq, Clone, Debug)]
struct AstProgram<'a> {
    main_function : AstFunction<'a>,
}

#[derive(PartialEq, Clone, Debug)]
struct AstFunction<'a> {
    name : &'a str,
    body : AstStatement,
}

#[derive(PartialEq, Clone, Debug)]
enum AstStatement {
    Return(AstExpression),
}

#[derive(PartialEq, Clone, Debug)]
enum AstUnaryOperator {
    Negation,
    BitwiseNot,
    LogicalNot,
}

#[derive(PartialEq, Clone, Debug)]
enum AstExpression {
    Constant(u32),
    UnaryOperator(AstUnaryOperator, Box<AstExpression>),
}

impl<'a> AstToString for AstProgram<'a> {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        format!("{}", self.main_function.ast_to_string(0))
    }
}

impl<'a> fmt::Display for AstProgram<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ast_to_string(0))
    }
}

impl<'a> AstToString for AstFunction<'a> {
    fn ast_to_string(&self, indent_levels : u32) -> String {
        format!("{}FUNC {}:\n{}", Self::get_indent_string(indent_levels), self.name, self.body.ast_to_string(indent_levels + 1))
    }
}

impl AstToString for AstStatement {
    fn ast_to_string(&self, indent_levels : u32) -> String {
        if let AstStatement::Return(expr) = self {
            format!("{}return {}", Self::get_indent_string(indent_levels), expr.ast_to_string(indent_levels + 1))
        } else {
            format!("{}err {:?}", Self::get_indent_string(indent_levels), self)
        }
    }
}

impl AstToString for AstExpression {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        if let AstExpression::Constant(val) = self {
            format!("{}", val)
        } else {
            format!("err {:?}", self)
        }
    }
}

fn lex_next_token<'a>(input : &'a str)  -> Result<(&'a str, &'a str), String> {
    lazy_static! {
        static ref TOKEN_REGEXES : Vec<regex::Regex> = vec![
            Regex::new(r"^\{").expect("failed to compile regex"),
            Regex::new(r"^\}").expect("failed to compile regex"),
            Regex::new(r"^\(").expect("failed to compile regex"),
            Regex::new(r"^\)").expect("failed to compile regex"),
            Regex::new(r"^;").expect("failed to compile regex"),
            Regex::new(r"^-").expect("failed to compile regex"),
            Regex::new(r"^~").expect("failed to compile regex"),
            Regex::new(r"^!").expect("failed to compile regex"),
            Regex::new(r"^[a-zA-Z]\w*").expect("failed to compile regex"),
            Regex::new(r"^[0-9]+").expect("failed to compile regex"),
        ];
    }

    for r in TOKEN_REGEXES.iter() {
        if let Some(mat) = r.find(input) {
            let range = mat.range();
            //println!("match: {}, {}", range.start, range.end);
            return Ok(input.split_at(range.end));
        }
    }

    Err(format!("unrecognized token starting at {}", input))
}

fn lex_all_tokens<'a>(input : &'a str) -> Result<Vec<&'a str>, String> {
    let mut tokens : Vec<&'a str> = vec![];

    let mut remaining_input = input.trim();
    while remaining_input.len() > 0 {
        match lex_next_token(&remaining_input) {
            Ok(split) => {
                //println!("[{}], [{}]", split.0, split.1);
                tokens.push(split.0);
                //println!("token: {}", split.0);
                remaining_input = split.1.trim();
            },
            Err(msg) => return Err(msg)
        }
    }

    Ok(tokens)
}

fn parse_program<'a>(remaining_tokens : &[&'a str]) -> Result<AstProgram<'a>, String> {
    parse_function(remaining_tokens).and_then(|(function, remaining_tokens)| {
        if remaining_tokens.len() == 0 {
            Ok(AstProgram {
                main_function: function,
            })
        } else {
            Err(format!("extra tokens after main function end: {:?}", remaining_tokens))
        }
    })
}

fn parse_function<'a, 'b>(remaining_tokens : &'b [&'a str]) -> Result<(AstFunction<'a>, &'b [&'a str]), String> {
    if remaining_tokens.len() >= 5 {
        let (tokens, remaining_tokens) = remaining_tokens.split_at(5);
        if tokens[0] == "int" &&
           tokens[2] == "(" &&
           tokens[3] == ")" &&
           tokens[4] == "{" {
            let name = tokens[1];

            parse_statement(remaining_tokens).and_then(|(body, remaining_tokens)| {
                if remaining_tokens.len() >= 1 {
                    let (tokens, remaining_tokens) = remaining_tokens.split_at(1);
                    if tokens[0] == "}" {
                        Ok((AstFunction {
                            name,
                            body,
                        }, remaining_tokens))
                    } else {
                        Err(format!("function body missing closing brace"))
                    }
                } else {
                    Err(format!("not enough tokens for function body closing brace"))
                }
            })
        } else {
            Err(format!("failed to find function declaration"))
        }
    } else {
        Err(format!("not enough tokens for function declaration"))
    }
}

fn parse_statement<'a, 'b>(remaining_tokens : &'b [&'a str]) -> Result<(AstStatement, &'b [&'a str]), String> {
    if remaining_tokens.len() >= 1 {
        let (tokens, remaining_tokens) = remaining_tokens.split_at(1);
        if tokens[0] == "return" {
            parse_expression(remaining_tokens).and_then(|(expr, remaining_tokens)| {
                if remaining_tokens.len() >= 1 {
                    let (tokens, remaining_tokens) = remaining_tokens.split_at(1);
                    if tokens[0] == ";" {
                        Ok((AstStatement::Return(expr), remaining_tokens))
                    } else {
                        Err(format!("statement must end with semicolon. instead ends with {}", tokens[0]))
                    }
                } else {
                    Err(format!("not enough tokens for ending semicolon"))
                }
            })
        } else {
            Err(format!("unrecognized statement starting with {}", tokens[0]))
        }
    } else {
        Err(format!("not enough tokens for statement"))
    }
}

fn parse_expression<'a, 'b>(remaining_tokens : &'b [&'a str]) -> Result<(AstExpression, &'b [&'a str]), String> {
    if remaining_tokens.len() >= 1 {
        let (tokens, remaining_tokens) = remaining_tokens.split_at(1);
        if let Ok(integer_literal) = tokens[0].parse::<u32>() {
            Ok((AstExpression::Constant(integer_literal), remaining_tokens))
        } else {
            let operator_result = match tokens[0] {
                "-" => Ok((AstUnaryOperator::Negation, remaining_tokens)),
                "~" => Ok((AstUnaryOperator::BitwiseNot, remaining_tokens)),
                "!" => Ok((AstUnaryOperator::LogicalNot, remaining_tokens)),
                _ => Err(format!("unknown operator {}", tokens[0]))
            };

            operator_result.and_then(|(operator, remaining_tokens)| {
                parse_expression(remaining_tokens).and_then(|(inner_expression, remaining_tokens)| {
                    Ok((AstExpression::UnaryOperator(operator, Box::new(inner_expression)), remaining_tokens))
                })
            })
        }
    } else {
        Err(format!("not enough tokens for expression"))
    }
}

fn get_register_name(register_name : &str, width : u32) -> String {
    match width {
        8 => format!("{}l", register_name),
        16 => format!("{}x", register_name),
        32 => format!("e{}x", register_name),
        64 => format!("r{}x", register_name),
        _ => panic!("unexpected register name"),
    }
}

fn generate_expression_code(ast_expression : &AstExpression, target_register : &str) -> Result<String, String> {
    let reg64 = get_register_name(target_register, 64);
    let reg8 = get_register_name(target_register, 8);

    match ast_expression {
        AstExpression::Constant(val) => Ok(format!("    mov {},{}", reg64, val)),
        AstExpression::UnaryOperator(operator, box_expr) => {
            generate_expression_code(box_expr, target_register).and_then(|inner_expr_code| {
                match operator {
                    AstUnaryOperator::Negation => Ok(format!("{}\n    neg {}", inner_expr_code, reg64)),
                    AstUnaryOperator::BitwiseNot => Ok(format!("{}\n    not {}", inner_expr_code, reg64)),
                    AstUnaryOperator::LogicalNot => Ok(format!("{}\n    cmp {},0\n    mov {},0\n    sete {}", inner_expr_code, reg64, reg64, reg8)),
                }
            })
        },
    }
}

fn generate_statement_code(ast_statement : &AstStatement) -> Result<String, String> {
    if let AstStatement::Return(expr) = ast_statement {
        generate_expression_code(expr, "a").and_then(|expr_code| {
            Ok(format!("{}\n    ret", expr_code))
        })
    } else {
        Err(format!("unsupported statement {:?}", ast_statement))
    }
}

fn generate_function_code(ast_function : &AstFunction) -> Result<String, String> {
    generate_statement_code(&ast_function.body).and_then(|function_code| {
        Ok(format!("{} PROC\n{}\n{} ENDP", ast_function.name, function_code, ast_function.name))
    })
}

fn generate_program_code(ast_program : &AstProgram) -> Result<String, String> {
    const header : &str =
r"INCLUDELIB msvcrt.lib
.DATA

.CODE
start:
";
    const footer : &str =
r"END
";

    generate_function_code(&ast_program.main_function).and_then(|main_code| {
        Ok(String::from(header) + &main_code + "\n" + footer)
    })
}

fn compile_and_link(code : &str, exe_path : &str) {
    const temp_path : &str = "temp.asm";

    std::fs::write(temp_path, &code);

    let status = Command::new("ml64.exe")
        .args(&[&format!("/Fe{}", exe_path), temp_path])
        .status()
        .expect("failed to run ml64.exe");

    println!("assembly status: {:?}", status);
    if status.success() {
        std::fs::remove_file(temp_path);
        std::fs::remove_file("temp.obj");
        std::fs::remove_file("mllink$.lnk");
    }
}

fn main() {
    let args : Vec<String> = env::args().collect();
    println!("loading {}", args[1]);
    let input = std::fs::read_to_string(&args[1]).unwrap();
    //println!("input: {}", input);

    match lex_all_tokens(&input) {
        Ok(tokens) => {
            for token in tokens.iter() {
                println!("{}", token);
            }

            println!();

            match parse_program(&tokens) {
                Ok(program) => {
                    println!("AST:\n{}\n", program);

                    match generate_program_code(&program) {
                        Ok(code) => {
                            println!("code:\n{}", code);
                            compile_and_link(&code, &args[2])
                        }
                        Err(msg) => println!("{}", msg),
                    }
                },
                Err(msg) => println!("{}", msg)
            }
        },
        Err(msg) => println!("{}", msg)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lex_simple() {
        let input =
r"int main() {
    return 2;
}";
        assert_eq!(lex_all_tokens(&input), Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"]));
    }

    #[test]
    fn lex_no_whitespace() {
        let input =
r"int main(){return 2;}";
        assert_eq!(lex_all_tokens(&input), Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"]));
    }

    #[test]
    fn lex_negative() {
        assert_eq!(lex_all_tokens("int main() { return -1; }"), Ok(vec!["int", "main", "(", ")", "{", "return", "-", "1", ";", "}"]));
    }

    #[test]
    fn lex_bitwise_not() {
        assert_eq!(lex_all_tokens("int main() { return ~1; }"), Ok(vec!["int", "main", "(", ")", "{", "return", "~", "1", ";", "}"]));
    }

    #[test]
    fn lex_logical_not() {
        assert_eq!(lex_all_tokens("int main() { return !1; }"), Ok(vec!["int", "main", "(", ")", "{", "return", "!", "1", ";", "}"]));
    }

    fn test_parse_simple(value : u32) {
        let input = format!(
r"int main() {{
    return {};
}}", value);

        assert_eq!(
            parse_program(&lex_all_tokens(&input).unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression::Constant(value)
                    )
                },
            })
        );
    }

    fn test_parse_failure(input : &str) {
        assert!(parse_program(&lex_all_tokens(input).unwrap()).is_err())
    }

    #[test]
    fn parse_return_0() {
        test_parse_simple(0)
    }

    #[test]
    fn parse_return_2() {
        test_parse_simple(2)
    }

    #[test]
    fn parse_return_multi_digit() {
        test_parse_simple(12345)
    }

    #[test]
    fn parse_error_missing_open_paren() {
        test_parse_failure("int main) { return 1; }")
    }

    #[test]
    fn parse_error_missing_close_paren() {
        test_parse_failure("int main( { return 1; }")
    }

    #[test]
    fn parse_error_missing_retval() {
        test_parse_failure("int main() { return; }")
    }

    #[test]
    fn parse_error_missing_close_brace() {
        test_parse_failure("int main() { return;")
    }

    #[test]
    fn parse_error_missing_statement_semicolon() {
        test_parse_failure("int main() { return }")
    }

    #[test]
    fn parse_error_missing_statement_missing_space() {
        test_parse_failure("int main() { return0; }")
    }

    #[test]
    fn parse_error_missing_statement_return_wrong_case() {
        test_parse_failure("int main() { RETURN 0; }")
    }

    #[test]
    fn parse_error_extra_token() {
        test_parse_failure("int main() { return 0; }}")
    }

    fn test_parse_single_unary_operator(token : &str, operator : AstUnaryOperator) {
        assert_eq!(
            parse_program(&lex_all_tokens(&format!("int main() {{ return {}1; }}", token)).unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression::UnaryOperator(
                            operator,
                            Box::new(AstExpression::Constant(1))
                        )
                    )
                },
            })
        );
    }

    #[test]
    fn parse_single_unary_operators() {
        test_parse_single_unary_operator("-", AstUnaryOperator::Negation);
        test_parse_single_unary_operator("~", AstUnaryOperator::BitwiseNot);
        test_parse_single_unary_operator("!", AstUnaryOperator::LogicalNot);
    }

    #[test]
    fn test_parse_multi_unary_operators() {
        assert_eq!(
            parse_program(&lex_all_tokens("int main() { return -~!1; }").unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression::UnaryOperator(
                            AstUnaryOperator::Negation,
                            Box::new(AstExpression::UnaryOperator(
                                AstUnaryOperator::BitwiseNot,
                                Box::new(AstExpression::UnaryOperator(
                                    AstUnaryOperator::LogicalNot,
                                    Box::new(AstExpression::Constant(1))
                                ))
                            ))
                        )
                    )
                },
            })
        );
    }
}
