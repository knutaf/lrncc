use std::io::prelude::*;

#[macro_use] extern crate lazy_static;
extern crate regex;
use regex::Regex;

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
enum AstExpression {
    Constant(u32),
}

pub fn read_all_stdin() -> String {
    let mut contents = String::new();
    std::io::stdin().read_to_string(&mut contents).expect("failed to read input from stdin");
    contents.trim().to_string()
}

fn lex_next_token<'a>(input : &'a str)  -> Option<(&'a str, &'a str)> {
    lazy_static! {
        static ref TOKEN_REGEXES : Vec<regex::Regex> = vec![
            Regex::new(r"^\{").expect("failed to compile regex"),
            Regex::new(r"^\}").expect("failed to compile regex"),
            Regex::new(r"^\(").expect("failed to compile regex"),
            Regex::new(r"^\)").expect("failed to compile regex"),
            Regex::new(r"^;").expect("failed to compile regex"),
            Regex::new(r"^[a-zA-Z]\w*").expect("failed to compile regex"),
            Regex::new(r"^[0-9]+").expect("failed to compile regex"),
        ];
    }

    for r in TOKEN_REGEXES.iter() {
        if let Some(mat) = r.find(input) {
            let range = mat.range();
            //println!("match: {}, {}", range.start, range.end);
            return Some(input.split_at(range.end));
        }
    }

    return None;
}

// TODO should use error instead of option
fn lex_all_tokens<'a>(input : &'a str) -> Option<Vec<&'a str>> {
    let mut tokens : Vec<&'a str> = vec![];

    let mut remaining_input = input.trim();
    while remaining_input.len() > 0 {
        if let Some(split) = lex_next_token(&remaining_input) {
            //println!("[{}], [{}]", split.0, split.1);
            tokens.push(split.0);
            //println!("token: {}", split.0);
            remaining_input = split.1.trim();
        } else {
            println!("Unrecognized token starting at {}", remaining_input);
            return None;
        }
    }

    Some(tokens)
}

fn parse_program<'a>(remaining_tokens : &[&'a str]) -> Option<AstProgram<'a>> {
    // TODO: verify no remaining tokens
    if let Some((function, _remaining_tokens)) = parse_function(remaining_tokens) {
        Some(AstProgram {
            main_function: function,
        })
    } else {
        println!("failed parse_function");
        None
    }
}

fn parse_function<'a, 'b>(remaining_tokens : &'b [&'a str]) -> Option<(AstFunction<'a>, &'b [&'a str])> {
    if remaining_tokens.len() >= 5 {
        let (tokens, remaining_tokens) = remaining_tokens.split_at(5);
        if tokens[0] == "int" &&
           tokens[2] == "(" &&
           tokens[3] == ")" &&
           tokens[4] == "{" {
            let name = tokens[1];

            if let Some((body, remaining_tokens)) = parse_statement(remaining_tokens) {
                Some((AstFunction {
                    name,
                    body,
                }, remaining_tokens))
                // TODO parse closing brace
            } else {
                println!("failed parse_statement");
                None
            }
        }
        else {
            println!("failed fn header");
            None
        }
    } else {
        println!("failed fn tokens len");
        None
    }
}

fn parse_statement<'a, 'b>(remaining_tokens : &'b [&'a str]) -> Option<(AstStatement, &'b [&'a str])> {
    if remaining_tokens.len() >= 1 {
        let (tokens, remaining_tokens) = remaining_tokens.split_at(1);
        if tokens[0] == "return" {
            if let Some((expr, remaining_tokens)) = parse_expression(remaining_tokens) {
                if remaining_tokens.len() >= 1 {
                    let (tokens, remaining_tokens) = remaining_tokens.split_at(1);
                    if tokens[0] == ";" {
                        Some((AstStatement::Return(expr), remaining_tokens))
                    } else {
                        println!("failed semicolon");
                        None
                    }
                } else {
                    println!("failed semicolon len");
                    None
                }
            } else {
                println!("failed parse_expression");
                None
            }
        } else {
            println!("failed statement return");
            None
        }
    } else {
        println!("failed statement len 1");
        None
    }
}

fn parse_expression<'a, 'b>(remaining_tokens : &'b [&'a str]) -> Option<(AstExpression, &'b [&'a str])> {
    if remaining_tokens.len() >= 1 {
        let (tokens, remaining_tokens) = remaining_tokens.split_at(1);
        if let Ok(integer_literal) = tokens[0].parse::<u32>() {
            Some((AstExpression::Constant(integer_literal), remaining_tokens))
        } else {
            println!("failed parse u32 {}", tokens[0]);
            None
        }
    } else {
        println!("failed tokens len");
        None
    }
}

fn main() {
    let input = read_all_stdin();
    //println!("input: {}", input);

    if let Some(tokens) = lex_all_tokens(&input) {
        for token in tokens.iter() {
            println!("{}", token);
        }

        parse_program(&tokens);
    } else {
        println!("parse error!");
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
        assert_eq!(lex_all_tokens(&input), Some(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"]));
    }

    #[test]
    fn parse_simple() {
        let input =
r"int main() {
    return 2;
}";
        assert_eq!(
            parse_program(&lex_all_tokens(&input).unwrap()),
            Some(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression::Constant(2)
                    )
                },
            })
        );
    }
}
