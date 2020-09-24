use std::env;
use std::fmt;
use std::process::*;
use std::path::*;
use std::collections::HashMap;
use std::ops::Deref;

#[macro_use] extern crate lazy_static;
extern crate regex;
use regex::Regex;

extern crate rand;
use rand::{thread_rng, Rng};
use rand::distributions::Alphanumeric;

const VARIABLE_SIZE : u32 = 8;

fn generate_random_string(len : usize) -> String {
    thread_rng()
        .sample_iter(&Alphanumeric)
        .take(len)
        .collect()
}

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

// A wrapper around a slice of tokens with convenience functions useful for parsing.
#[derive(PartialEq, Clone, Debug)]
struct Tokens<'i, 't>(&'t [&'i str]);

impl<'i, 't> Tokens<'i, 't> {
    fn consume_tokens(&self, num_tokens : usize) -> Result<(Tokens<'i, 't>, Tokens<'i, 't>), String> {
        if self.0.len() >= num_tokens {
            let (tokens, remaining_tokens) = self.0.split_at(num_tokens);
            Ok((Tokens(tokens), Tokens(remaining_tokens)))
        } else {
            Err(format!("could not find {} more token(s)", num_tokens))
        }
    }

    fn consume_expected_next_token(&mut self, expected_token : &str) -> Result<&mut Self, String> {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        if tokens[0] == expected_token {
            *self = remaining_tokens;
            Ok(self)
        } else {
            Err(format!("expected next token \"{}\" but found \"{}\"", expected_token, tokens[0]))
        }
    }

    fn consume_next_token(&mut self) -> Result<&'i str, String> {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;
        *self = remaining_tokens;
        Ok(tokens[0])
    }

    fn consume_variable_name(&mut self) -> Result<&'i str, String> {
        fn is_token_variable_name(token : &str) -> bool {
            lazy_static! {
                static ref VAR_REGEX : Regex = Regex::new(r"^[a-zA-Z]\w*$").expect("failed to compile regex");
            }

            VAR_REGEX.find(token).is_some()
        }

        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        if is_token_variable_name(tokens[0]) {
            *self = remaining_tokens;
            Ok(tokens[0])
        } else {
            Err(format!("token \"{}\" is not a variable name", tokens[0]))
        }
    }

    fn consume_and_parse_next_token<T>(&mut self) -> Result<T, String>
    where T : std::str::FromStr {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        let result = tokens[0].parse::<T>().or(Err(format!("token \"{}\" can't be parsed", tokens[0])));

        if result.is_ok() {
            *self = remaining_tokens;
        }

        result
    }
}

impl<'i, 't> Deref for Tokens<'i, 't> {
    type Target = &'t [&'i str];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

trait BinaryOperatorCodeGenerator {
    fn generate_code(&self, global_state : &mut CodegenGlobalState, rhs_code : &str) -> String;
}

#[derive(PartialEq, Clone, Debug)]
struct AstProgram<'i> {
    main_function : AstFunction<'i>,
}

#[derive(PartialEq, Clone, Debug)]
struct AstFunction<'i> {
    name : &'i str,
    body : Vec<AstBlockItem>,
}

#[derive(PartialEq, Clone, Debug)]
enum AstBlockItem {
    Declaration(AstDeclaration),
    Statement(AstStatement),
}

#[derive(PartialEq, Clone, Debug)]
enum AstDeclaration {
    DeclareVar(String, Option<AstExpression>),
}

// TODO: use a string slice instead of a string
#[derive(PartialEq, Clone, Debug)]
enum AstStatement {
    Return(AstExpression),
    Expression(AstExpression),
    Conditional(AstExpression, Box<AstStatement>, Option<Box<AstStatement>>),
}

#[derive(PartialEq, Clone, Debug)]
enum AstLogicalOrExpressionBinaryOperator {
    Or,
}

#[derive(PartialEq, Clone, Debug)]
enum AstLogicalAndExpressionBinaryOperator {
    And,
}

#[derive(PartialEq, Clone, Debug)]
enum AstEqualityExpressionBinaryOperator {
    Equals,
    NotEquals,
}

#[derive(PartialEq, Clone, Debug)]
enum AstRelationalExpressionBinaryOperator {
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
}

#[derive(PartialEq, Clone, Debug)]
enum AstAdditiveExpressionBinaryOperator {
    Plus,
    Minus,
}

#[derive(PartialEq, Clone, Debug)]
enum AstTermBinaryOperator {
    Multiply,
    Divide,
}

#[derive(PartialEq, Clone, Debug)]
enum AstUnaryOperator {
    Negation,
    BitwiseNot,
    LogicalNot,
}

#[derive(PartialEq, Clone, Debug)]
enum AstFactor {
    Constant(u32),
    Variable(String), // TODO use a string slice
    UnaryOperator(AstUnaryOperator, Box<AstFactor>),
    Expression(Box<AstExpression>),
}

#[derive(PartialEq, Clone, Debug)]
struct AstBinaryOperation<TOperator, TRhs> {
    operator : TOperator,
    rhs : TRhs
}

#[derive(PartialEq, Clone, Debug)]
enum AstExpression {
    Assign(String, Box<AstExpression>),
    Or(AstLogicalOrExpression),
}

type AstLogicalOrExpressionBinaryOperation = AstBinaryOperation<AstLogicalOrExpressionBinaryOperator, AstLogicalAndExpressionBinaryOperation>;
type AstLogicalAndExpressionBinaryOperation = AstBinaryOperation<AstLogicalAndExpressionBinaryOperator, AstEqualityExpressionBinaryOperation>;
type AstEqualityExpressionBinaryOperation = AstBinaryOperation<AstEqualityExpressionBinaryOperator, AstRelationalExpressionBinaryOperation>;
type AstRelationalExpressionBinaryOperation = AstBinaryOperation<AstAdditiveExpressionBinaryOperator, AstAdditiveExpressionBinaryOperation>;
type AstAdditiveExpressionBinaryOperation = AstBinaryOperation<AstAdditiveExpressionBinaryOperator, AstTerm>;
type AstTermBinaryOperation = AstBinaryOperation<AstTermBinaryOperator, AstFactor>;

#[derive(PartialEq, Clone, Debug)]
struct AstExpressionLevel<TOperator, TInner> {
    inner : TInner,
    binary_ops : Vec<AstBinaryOperation<TOperator, TInner>>,
}

type AstLogicalOrExpression = AstExpressionLevel<AstLogicalOrExpressionBinaryOperator, AstLogicalAndExpression>;
type AstLogicalAndExpression = AstExpressionLevel<AstLogicalAndExpressionBinaryOperator, AstEqualityExpression>;
type AstEqualityExpression = AstExpressionLevel<AstEqualityExpressionBinaryOperator, AstRelationalExpression>;
type AstRelationalExpression = AstExpressionLevel<AstRelationalExpressionBinaryOperator, AstAdditiveExpression>;
type AstAdditiveExpression = AstExpressionLevel<AstAdditiveExpressionBinaryOperator, AstTerm>;
type AstTerm = AstExpressionLevel<AstTermBinaryOperator, AstFactor>;

struct CodegenGlobalState {
    next_label : u32,
}

#[derive(Clone)]
struct CodegenFunctionState {
    variables : HashMap<String, u32>,
    next_offset : u32,
}

impl<TOperator, TInner> AstExpressionLevel<TOperator, TInner> {
    fn new(inner : TInner) -> AstExpressionLevel<TOperator, TInner> {
        AstExpressionLevel {
            inner,
            binary_ops : vec![],
        }
    }
}

impl CodegenGlobalState {
    fn new() -> CodegenGlobalState {
        CodegenGlobalState {
            next_label : 0,
        }
    }

    fn consume_jump_label(&mut self) -> u32 {
        self.next_label += 1;
        self.next_label - 1
    }
}

impl CodegenFunctionState {
    fn new() -> CodegenFunctionState {
        CodegenFunctionState {
            variables : HashMap::new(),
            next_offset : VARIABLE_SIZE,
        }
    }

    fn add_var(&mut self, name : &str) -> bool {
        if !self.variables.contains_key(name) {
            self.variables.insert(String::from(name), self.next_offset);
            self.next_offset += VARIABLE_SIZE;
            true
        } else {
            false
        }
    }

    fn get_var_offset(&self, name : &str) -> Option<u32> {
        self.variables.get(name).map(|value| {
            *value
        })
    }
}

impl<'i> AstToString for AstProgram<'i> {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        format!("{}", self.main_function.ast_to_string(0))
    }
}

impl<'i> fmt::Display for AstProgram<'i> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ast_to_string(0))
    }
}

impl<'i> AstToString for AstFunction<'i> {
    fn ast_to_string(&self, indent_levels : u32) -> String {
        let mut body_str = String::new();
        for statement in &self.body {
            body_str += "\n";
            body_str += &statement.ast_to_string(indent_levels + 1);
        }

        format!("{}FUNC {}:{}", Self::get_indent_string(indent_levels), self.name, &body_str)
    }
}

impl AstToString for AstBlockItem {
    fn ast_to_string(&self, indent_levels : u32) -> String {
        match self {
            AstBlockItem::Statement(statement) => statement.ast_to_string(indent_levels),
            AstBlockItem::Declaration(declaration) => declaration.ast_to_string(indent_levels),
        }
    }
}

impl AstToString for AstDeclaration {
    fn ast_to_string(&self, indent_levels : u32) -> String {
        if let AstDeclaration::DeclareVar(name, expr_opt) = self {
            if let Some(expr) = expr_opt {
                format!("{}int {} = {};", Self::get_indent_string(indent_levels), &name, expr.ast_to_string(indent_levels + 1))
            } else {
                format!("{}int {};", Self::get_indent_string(indent_levels), &name)
            }
        } else {
            format!("{}err {:?}", Self::get_indent_string(indent_levels), self)
        }
    }
}

impl AstToString for AstStatement {
    fn ast_to_string(&self, indent_levels : u32) -> String {
        match self {
            AstStatement::Return(expr) => format!("{}return {};", Self::get_indent_string(indent_levels), expr.ast_to_string(indent_levels + 1)),
            AstStatement::Expression(expr) => format!("{}{};", Self::get_indent_string(indent_levels), expr.ast_to_string(indent_levels + 1)),
            AstStatement::Conditional(expr, positive, negative_opt) => {
                let mut result = format!("{}if ({})\n{}", Self::get_indent_string(indent_levels), expr.ast_to_string(0), positive.ast_to_string(indent_levels + 1));
                if let Some(negative) = negative_opt {
                    result += &format!("\n{}else\n{}", Self::get_indent_string(indent_levels), negative.ast_to_string(indent_levels + 1));
                }
                result
            },
        }
    }
}

impl AstToString for AstExpression {
    fn ast_to_string(&self, indent_levels : u32) -> String {
        match self {
            AstExpression::Assign(name, expr) => format!("{} = {}", name, expr.ast_to_string(indent_levels)),
            AstExpression::Or(expr) => expr.ast_to_string(indent_levels),
        }
    }
}

impl AstToString for AstLogicalOrExpressionBinaryOperator {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        String::from(match self {
            AstLogicalOrExpressionBinaryOperator::Or => "||",
        })
    }
}

impl AstToString for AstLogicalAndExpressionBinaryOperator {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        String::from(match self {
            AstLogicalAndExpressionBinaryOperator::And => "&&",
        })
    }
}

impl AstToString for AstEqualityExpressionBinaryOperator {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        String::from(match self {
            AstEqualityExpressionBinaryOperator::Equals => "==",
            AstEqualityExpressionBinaryOperator::NotEquals => "!=",
        })
    }
}

impl AstToString for AstRelationalExpressionBinaryOperator {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        String::from(match self {
            AstRelationalExpressionBinaryOperator::LessThan => "<",
            AstRelationalExpressionBinaryOperator::GreaterThan => ">",
            AstRelationalExpressionBinaryOperator::LessThanEqual => "<=",
            AstRelationalExpressionBinaryOperator::GreaterThanEqual => ">=",
        })
    }
}

impl AstToString for AstAdditiveExpressionBinaryOperator {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        String::from(match self {
            AstAdditiveExpressionBinaryOperator::Plus => "+",
            AstAdditiveExpressionBinaryOperator::Minus => "-",
        })
    }
}

impl AstToString for AstTermBinaryOperator {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        String::from(match self {
            AstTermBinaryOperator::Multiply => "*",
            AstTermBinaryOperator::Divide => "/",
        })
    }
}

impl AstToString for AstUnaryOperator {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        String::from(match self {
            AstUnaryOperator::Negation => "-",
            AstUnaryOperator::BitwiseNot => "~",
            AstUnaryOperator::LogicalNot => "!",
        })
    }
}

impl AstToString for AstFactor {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        match self {
            AstFactor::Constant(val) => format!("{}", val),
            AstFactor::Variable(name) => name.clone(),
            AstFactor::UnaryOperator(operator, factor) => {
                format!("{}{}", operator.ast_to_string(0), factor.ast_to_string(0))
            },
            AstFactor::Expression(expr) => format!("({})", expr.ast_to_string(0)),
        }
    }
}

impl std::str::FromStr for AstUnaryOperator {
    type Err = String;
    fn from_str(s : &str) -> Result<Self, Self::Err> {
        match s {
            "-" => Ok(AstUnaryOperator::Negation),
            "~" => Ok(AstUnaryOperator::BitwiseNot),
            "!" => Ok(AstUnaryOperator::LogicalNot),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl std::str::FromStr for AstLogicalOrExpressionBinaryOperator {
    type Err = String;
    fn from_str(s : &str) -> Result<Self, Self::Err> {
        match s {
            "||" => Ok(AstLogicalOrExpressionBinaryOperator::Or),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl std::str::FromStr for AstLogicalAndExpressionBinaryOperator {
    type Err = String;
    fn from_str(s : &str) -> Result<Self, Self::Err> {
        match s {
            "&&" => Ok(AstLogicalAndExpressionBinaryOperator::And),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl std::str::FromStr for AstEqualityExpressionBinaryOperator {
    type Err = String;
    fn from_str(s : &str) -> Result<Self, Self::Err> {
        match s {
            "==" => Ok(AstEqualityExpressionBinaryOperator::Equals),
            "!=" => Ok(AstEqualityExpressionBinaryOperator::NotEquals),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl std::str::FromStr for AstRelationalExpressionBinaryOperator {
    type Err = String;
    fn from_str(s : &str) -> Result<Self, Self::Err> {
        match s {
            "<" => Ok(AstRelationalExpressionBinaryOperator::LessThan),
            ">" => Ok(AstRelationalExpressionBinaryOperator::GreaterThan),
            "<=" => Ok(AstRelationalExpressionBinaryOperator::LessThanEqual),
            ">=" => Ok(AstRelationalExpressionBinaryOperator::GreaterThanEqual),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl std::str::FromStr for AstAdditiveExpressionBinaryOperator {
    type Err = String;
    fn from_str(s : &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(AstAdditiveExpressionBinaryOperator::Plus),
            "-" => Ok(AstAdditiveExpressionBinaryOperator::Minus),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl std::str::FromStr for AstTermBinaryOperator {
    type Err = String;
    fn from_str(s : &str) -> Result<Self, Self::Err> {
        match s {
            "*" => Ok(AstTermBinaryOperator::Multiply),
            "/" => Ok(AstTermBinaryOperator::Divide),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl BinaryOperatorCodeGenerator for AstLogicalOrExpressionBinaryOperator {
    fn generate_code(&self, global_state : &mut CodegenGlobalState, rhs_code : &str) -> String {
        let label = global_state.consume_jump_label();
        format!("\n    cmp rax,0\n    mov rax,0\n    setne al\n    jne _j{}\n{}\n    cmp rax,0\n    mov rax,0\n    setne al\n    _j{}:", label, rhs_code, label)
    }
}

impl BinaryOperatorCodeGenerator for AstLogicalAndExpressionBinaryOperator {
    fn generate_code(&self, global_state : &mut CodegenGlobalState, rhs_code : &str) -> String {
        let label = global_state.consume_jump_label();
        format!("\n    cmp rax,0\n    je _j{}\n{}\n    cmp rax,0\n    mov rax,0\n    setne al\n    _j{}:", label, rhs_code, label)
    }
}

impl BinaryOperatorCodeGenerator for AstEqualityExpressionBinaryOperator {
    fn generate_code(&self, global_state : &mut CodegenGlobalState, rhs_code : &str) -> String {
        format!("\n    push rax\n{}", rhs_code) +
        &match &self {
            AstEqualityExpressionBinaryOperator::Equals => format!("\n    pop rcx\n    cmp rcx,rax\n    mov rax,0\n    sete al"),
            AstEqualityExpressionBinaryOperator::NotEquals => format!("\n    pop rcx\n    cmp rcx,rax\n    mov rax,0\n    setne al"),
        }
    }
}

impl BinaryOperatorCodeGenerator for AstRelationalExpressionBinaryOperator {
    fn generate_code(&self, global_state : &mut CodegenGlobalState, rhs_code : &str) -> String {
        format!("\n    push rax\n{}", rhs_code) +
        &match &self {
            AstRelationalExpressionBinaryOperator::LessThan => format!("\n    pop rcx\n    cmp rcx,rax\n    mov rax,0\n    setl al"),
            AstRelationalExpressionBinaryOperator::GreaterThan => format!("\n    pop rcx\n    cmp rcx,rax\n    mov rax,0\n    setg al"),
            AstRelationalExpressionBinaryOperator::LessThanEqual => format!("\n    pop rcx\n    cmp rcx,rax\n    mov rax,0\n    setle al"),
            AstRelationalExpressionBinaryOperator::GreaterThanEqual => format!("\n    pop rcx\n    cmp rcx,rax\n    mov rax,0\n    setge al"),
        }
    }
}

impl BinaryOperatorCodeGenerator for AstAdditiveExpressionBinaryOperator {
    fn generate_code(&self, global_state : &mut CodegenGlobalState, rhs_code : &str) -> String {
        format!("\n    push rax\n{}", rhs_code) +
        &match &self {
            AstAdditiveExpressionBinaryOperator::Plus => format!("\n    pop rcx\n    add rax,rcx"),
            AstAdditiveExpressionBinaryOperator::Minus => format!("\n    pop rcx\n    sub rcx,rax\n    mov rax,rcx"),
        }
    }
}

impl BinaryOperatorCodeGenerator for AstTermBinaryOperator {
    fn generate_code(&self, global_state : &mut CodegenGlobalState, rhs_code : &str) -> String {
        format!("\n    push rax\n{}", rhs_code) +
        &match &self {
            AstTermBinaryOperator::Multiply => format!("\n    mov rcx,rax\n    pop rax\n    imul rax,rcx"),
            AstTermBinaryOperator::Divide => format!("\n    mov rcx,rax\n    pop rax\n    cdq\n    idiv ecx"),
        }
    }
}

impl<TOperator, TRhs> AstToString for AstBinaryOperation<TOperator, TRhs>
    where TOperator : AstToString, TRhs : AstToString {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        format!("{} {}", self.operator.ast_to_string(0), self.rhs.ast_to_string(0))
    }
}

impl<TOperator, TInner> AstToString for AstExpressionLevel<TOperator, TInner>
    where TOperator : AstToString, TInner : AstToString {
    fn ast_to_string(&self, _indent_levels : u32) -> String {
        let format_binary_ops = || -> String {
            let mut result = String::new();
            for binop in &self.binary_ops {
                result += " ";
                result += &binop.ast_to_string(0);
            }
            result
        };

        self.inner.ast_to_string(0) + &format_binary_ops()
    }
}

fn lex_next_token<'i>(input : &'i str)  -> Result<(&'i str, &'i str), String> {
    lazy_static! {
        static ref TOKEN_REGEXES : Vec<regex::Regex> = vec![
            Regex::new(r"^&&").expect("failed to compile regex"),
            Regex::new(r"^\|\|").expect("failed to compile regex"),
            Regex::new(r"^==").expect("failed to compile regex"),
            Regex::new(r"^!=").expect("failed to compile regex"),
            Regex::new(r"^<=").expect("failed to compile regex"),
            Regex::new(r"^>=").expect("failed to compile regex"),
            Regex::new(r"^\{").expect("failed to compile regex"),
            Regex::new(r"^\}").expect("failed to compile regex"),
            Regex::new(r"^\(").expect("failed to compile regex"),
            Regex::new(r"^\)").expect("failed to compile regex"),
            Regex::new(r"^;").expect("failed to compile regex"),
            Regex::new(r"^-").expect("failed to compile regex"),
            Regex::new(r"^~").expect("failed to compile regex"),
            Regex::new(r"^!").expect("failed to compile regex"),
            Regex::new(r"^\+").expect("failed to compile regex"),
            Regex::new(r"^/").expect("failed to compile regex"),
            Regex::new(r"^\*").expect("failed to compile regex"),
            Regex::new(r"^<").expect("failed to compile regex"),
            Regex::new(r"^>").expect("failed to compile regex"),
            Regex::new(r"^=").expect("failed to compile regex"),
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

fn lex_all_tokens<'i>(input : &'i str) -> Result<Vec<&'i str>, String> {
    let mut tokens : Vec<&'i str> = vec![];

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

fn parse_program<'i, 't>(mut tokens : Tokens<'i, 't>) -> Result<AstProgram<'i>, String> {
    let function = parse_function(&mut tokens)?;
    if tokens.0.len() == 0 {
        Ok(AstProgram {
            main_function: function,
        })
    } else {
        Err(format!("extra tokens after main function end: {:?}", tokens))
    }
}

fn parse_function<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstFunction<'i>, String> {
    let mut tokens = original_tokens.clone();

    tokens.consume_expected_next_token("int")?;
    let name = tokens.consume_next_token()?;
    tokens.consume_expected_next_token("(")?;
    tokens.consume_expected_next_token(")")?;
    tokens.consume_expected_next_token("{")?;

    // Parse out all the block items possible.
    let mut block_items = vec![];
    loop {
        if let Ok(block_item) = parse_block_item(&mut tokens) {
            block_items.push(block_item);
        } else {
            break;
        }
    }

    tokens.consume_expected_next_token("}")?;

    *original_tokens = tokens;
    Ok(AstFunction {
        name,
        body : block_items,
    })
}

fn parse_block_item<'i, 't>(tokens : &mut Tokens<'i, 't>) -> Result<AstBlockItem, String> {
    if let Ok(declaration) = parse_declaration(tokens) {
        Ok(AstBlockItem::Declaration(declaration))
    } else {
        parse_statement(tokens).map(|statement| {
            AstBlockItem::Statement(statement)
        })
    }
}

fn parse_declaration<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstDeclaration, String> {
    let mut tokens = original_tokens.clone();

    tokens.consume_expected_next_token("int")?;
    let var_name = tokens.consume_next_token()?;

    let mut expr_opt = None;
    if tokens.consume_expected_next_token("=").is_ok() {
        expr_opt = Some(parse_expression(&mut tokens)?);
    }

    tokens.consume_expected_next_token(";")?;

    *original_tokens = tokens;
    Ok(AstDeclaration::DeclareVar(String::from(var_name), expr_opt))
}

fn parse_statement<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstStatement, String> {
    let mut tokens = original_tokens.clone();

    let statement;
    if tokens.consume_expected_next_token("return").is_ok() {
        statement = AstStatement::Return(parse_expression(&mut tokens)?);
        tokens.consume_expected_next_token(";")?;
    } else if tokens.consume_expected_next_token("if").is_ok() {
        tokens.consume_expected_next_token("(")?;
        let expr = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;
        let positive = parse_statement(&mut tokens)?;

        let negative;
        if tokens.consume_expected_next_token("else").is_ok() {
            negative = Some(Box::new(parse_statement(&mut tokens)?));
        } else {
            negative = None;
        }

        statement = AstStatement::Conditional(expr, Box::new(positive), negative);
    } else {
        statement = AstStatement::Expression(parse_expression(&mut tokens)?);
        tokens.consume_expected_next_token(";")?;
    }

    *original_tokens = tokens;
    Ok(statement)
}

fn parse_factor<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstFactor, String> {
    let mut tokens = original_tokens.clone();

    if let Ok(integer_literal) = tokens.consume_and_parse_next_token::<u32>() {
        *original_tokens = tokens;
        Ok(AstFactor::Constant(integer_literal))
    } else if tokens.consume_expected_next_token("(").is_ok() {
        let inner = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;
        *original_tokens = tokens;
        Ok(AstFactor::Expression(Box::new(inner)))
    } else if let Ok(operator) = tokens.consume_and_parse_next_token::<AstUnaryOperator>() {
        let inner = parse_factor(&mut tokens)?;
        *original_tokens = tokens;
        Ok(AstFactor::UnaryOperator(operator, Box::new(inner)))
    } else {
        let variable_name = tokens.consume_variable_name()?;
        *original_tokens = tokens;
        Ok(AstFactor::Variable(String::from(variable_name)))
    }
}

fn parse_expression_level<'i, 't, TOperator, TInner>(
    original_tokens : &mut Tokens<'i, 't>,
    parse_inner : fn(&mut Tokens<'i, 't>) -> Result<TInner, String>
    )
    -> Result<AstExpressionLevel<TOperator, TInner>, String>
    where TOperator : std::str::FromStr
{
    let mut tokens = original_tokens.clone();

    let inner1 = parse_inner(&mut tokens)?;
    let mut expr = AstExpressionLevel::<TOperator, TInner>::new(inner1);

    while let Ok(operator) = tokens.consume_and_parse_next_token::<TOperator>() {
        let rhs = parse_inner(&mut tokens)?;

        expr.binary_ops.push(AstBinaryOperation {
            operator,
            rhs,
        });
    }

    *original_tokens = tokens;
    Ok(expr)
}

fn parse_assignment_expression<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    let variable_name = tokens.consume_variable_name()?;
    tokens.consume_expected_next_token("=")?;

    let expr = parse_expression(&mut tokens)?;
    *original_tokens = tokens;
    Ok(AstExpression::Assign(String::from(variable_name), Box::new(expr)))
}

fn parse_expression<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    if let Ok(assignment) = parse_assignment_expression(original_tokens) {
        Ok(assignment)
    } else {
        parse_logical_or_expression(original_tokens).map(|expr| {
            AstExpression::Or(expr)
        })
    }
}

fn parse_logical_or_expression<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstLogicalOrExpression, String> {
    parse_expression_level::<AstLogicalOrExpressionBinaryOperator, AstLogicalAndExpression>(original_tokens, parse_logical_and_expression)
}

fn parse_logical_and_expression<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstLogicalAndExpression, String> {
    parse_expression_level::<AstLogicalAndExpressionBinaryOperator, AstEqualityExpression>(original_tokens, parse_equality_expression)
}

fn parse_equality_expression<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstEqualityExpression, String> {
    parse_expression_level::<AstEqualityExpressionBinaryOperator, AstRelationalExpression>(original_tokens, parse_relational_expression)
}

fn parse_relational_expression<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstRelationalExpression, String> {
    parse_expression_level::<AstRelationalExpressionBinaryOperator, AstAdditiveExpression>(original_tokens, parse_additive_expression)
}

fn parse_additive_expression<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstAdditiveExpression, String> {
    parse_expression_level::<AstAdditiveExpressionBinaryOperator, AstTerm>(original_tokens, parse_term)
}

fn parse_term<'i, 't>(original_tokens : &mut Tokens<'i, 't>) -> Result<AstTerm, String> {
    parse_expression_level::<AstTermBinaryOperator, AstFactor>(original_tokens, parse_factor)
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

fn generate_factor_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_factor : &AstFactor) -> Result<String, String> {
    match ast_factor {
        AstFactor::Constant(val) => Ok(format!("    mov rax,{}", val)),
        AstFactor::Variable(name) => {
            func_state.get_var_offset(&name).ok_or(format!("unknown variable {}", name)).and_then(|offset| {
                Ok(format!("    mov rax,[rbp-{}]", offset))
            })
        },
        AstFactor::UnaryOperator(operator, box_factor) => {
            generate_factor_code(global_state, func_state, box_factor).and_then(|inner_factor_code| {
                match operator {
                    AstUnaryOperator::Negation => Ok(format!("{}\n    neg rax", inner_factor_code)),
                    AstUnaryOperator::BitwiseNot => Ok(format!("{}\n    not rax", inner_factor_code)),
                    AstUnaryOperator::LogicalNot => Ok(format!("{}\n    cmp rax,0\n    mov rax,0\n    sete al", inner_factor_code)),
                }
            })
        },
        AstFactor::Expression(box_expr) => generate_expression_code(global_state, func_state, &box_expr),
    }
}

fn generate_expression_level_code<TOperator, TInner>(
    global_state : &mut CodegenGlobalState,
    func_state : &CodegenFunctionState,
    expr : &AstExpressionLevel<TOperator, TInner>,
    generate_inner : fn(&mut CodegenGlobalState, &CodegenFunctionState, &TInner) -> Result<String, String>
    ) -> Result<String, String>
    where TOperator : BinaryOperatorCodeGenerator {
    generate_inner(global_state, func_state, &expr.inner).and_then(|mut code| {
        for binop in &expr.binary_ops {
            let inner_code_result = generate_inner(global_state, func_state, &binop.rhs);
            if let Ok(inner_code) = inner_code_result {
                code += &binop.operator.generate_code(global_state, &inner_code);
            } else {
                return inner_code_result;
            }
        }

        Ok(code)
    })
}

fn generate_expression_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_node : &AstExpression) -> Result<String, String> {
    match ast_node {
        AstExpression::Or(expr) => {
            generate_logical_or_expression_code(global_state, func_state, expr)
        },
        AstExpression::Assign(name, expr) => {
            generate_expression_code(global_state, func_state, expr).and_then(|expr_code| {
                func_state.get_var_offset(&name).ok_or(format!("unknown variable {}", name)).and_then(|offset| {
                    Ok(format!("{}\n    mov [rbp-{}],rax", expr_code, offset))
                })
            })
        },
    }
}

fn generate_logical_or_expression_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_node : &AstLogicalOrExpression) -> Result<String, String> {
    generate_expression_level_code(global_state, func_state, ast_node, generate_logical_and_expression_code)
}

fn generate_logical_and_expression_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_node : &AstLogicalAndExpression) -> Result<String, String> {
    generate_expression_level_code(global_state, func_state, ast_node, generate_equality_expression_code)
}

fn generate_equality_expression_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_node : &AstEqualityExpression) -> Result<String, String> {
    generate_expression_level_code(global_state, func_state, ast_node, generate_relational_expression_code)
}

fn generate_relational_expression_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_node : &AstRelationalExpression) -> Result<String, String> {
    generate_expression_level_code(global_state, func_state, ast_node, generate_additive_expression_code)
}

fn generate_additive_expression_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_node : &AstAdditiveExpression) -> Result<String, String> {
    generate_expression_level_code(global_state, func_state, ast_node, generate_term_code)
}

fn generate_term_code(global_state : &mut CodegenGlobalState, func_state : &CodegenFunctionState, ast_node : &AstTerm) -> Result<String, String> {
    generate_expression_level_code(global_state, func_state, ast_node, generate_factor_code)
}

fn generate_block_item_code(global_state : &mut CodegenGlobalState, func_state : &mut CodegenFunctionState, ast_block_item : &AstBlockItem) -> Result<String, String> {
    match ast_block_item {
        AstBlockItem::Statement(statement) => generate_statement_code(global_state, func_state, statement),
        AstBlockItem::Declaration(declaration) => generate_declaration_code(global_state, func_state, declaration),
    }
}

fn generate_declaration_code(global_state : &mut CodegenGlobalState, func_state : &mut CodegenFunctionState, ast_declaration : &AstDeclaration) -> Result<String, String> {
    match ast_declaration {
        AstDeclaration::DeclareVar(name, expr_opt) => {
            if !func_state.add_var(&name) {
                return Err(format!("variable {} already defined", name));
            }

            let mut code = String::new();
            if let Some(expr) = expr_opt {
                let result = generate_expression_code(global_state, func_state, expr);
                if let Ok(expr_code) = result {
                    code += &expr_code;
                } else {
                    return result;
                }
            }

            // The assignment expression (or junk data, if no expression was used) is in rax and should be stored at the
            // variable's location on the stack.
            code += "\n    push rax";
            Ok(code)
        },
    }
}

fn generate_statement_code(global_state : &mut CodegenGlobalState, func_state : &mut CodegenFunctionState, ast_statement : &AstStatement) -> Result<String, String> {
    match ast_statement {
        AstStatement::Return(expr) => {
            generate_expression_code(global_state, func_state, expr).and_then(|expr_code| {
                Ok(format!("{}\n    mov rsp,rbp\n    pop rbp\n    ret", expr_code))
            })
        },
        AstStatement::Expression(expr) => {
            generate_expression_code(global_state, func_state, expr)
        },
        _ => Err(format!("unsupported statement {:?}", ast_statement)),
    }
}

fn generate_function_code(global_state : &mut CodegenGlobalState, ast_function : &AstFunction) -> Result<String, String> {
    let mut func_state = CodegenFunctionState::new();
    let mut code = format!("{} PROC\n    push rbp\n    mov rbp,rsp", ast_function.name);

    for block_item in &ast_function.body {
        let result = generate_block_item_code(global_state, &mut func_state, block_item);
        if let Ok(block_item_code) = result {
            code += "\n";
            code += &block_item_code;
        } else {
            return result;
        }
    }

    // Add a default return of 0 in case the code in the function body didn't put a return statement.
    Ok(code + &format!("\n    mov rsp,rbp\n    pop rbp\n    ret\n{} ENDP", ast_function.name))
}

fn generate_program_code(ast_program : &AstProgram) -> Result<String, String> {
    const HEADER : &str =
r"INCLUDELIB msvcrt.lib
.DATA

.CODE
start:
";
    const FOOTER : &str =
r"END
";

    let mut codegen_state = CodegenGlobalState::new();
    generate_function_code(&mut codegen_state, &ast_program.main_function).and_then(|main_code| {
        Ok(String::from(HEADER) + &main_code + "\n" + FOOTER)
    })
}

fn assemble_and_link(code : &str, exe_path : &str, should_suppress_output : bool) -> Option<i32> {
    let temp_dir = Path::new(&format!("testrun_{}", generate_random_string(8))).to_path_buf();
    std::fs::create_dir_all(&temp_dir);

    let asm_path = temp_dir.join("code.asm");
    let exe_temp_output_path = temp_dir.join("output.exe");
    let pdb_temp_output_path = temp_dir.join("output.pdb");

    std::fs::write(&asm_path, &code);

    let mut command = Command::new("ml64.exe");
    let args = ["/Zi", "/Feoutput.exe", "code.asm", "/link", "/pdb:output.pdb"];
    println!("ml64.exe {} {}", args[0], args[1]);
    command.args(&args);
    command.current_dir(&temp_dir);

    if should_suppress_output {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }

    let status = command.status()
    .expect("failed to run ml64.exe");

    println!("assembly status: {:?}", status);
    if status.success() {
        std::fs::rename(&exe_temp_output_path, &Path::new(exe_path));

        let mut pdb_path = Path::new(exe_path).to_path_buf();
        pdb_path.set_extension("pdb");
        std::fs::rename(&pdb_temp_output_path, &pdb_path);
        println!("cleaning up temp dir {}", temp_dir.to_string_lossy());
        std::fs::remove_dir_all(&temp_dir);
    }

    status.code()
}

fn compile_and_link(input : &str, output_exe : &str, should_suppress_output : bool) -> Result<i32, String> {
    lex_all_tokens(&input).and_then(|tokens| {
        for token in tokens.iter() {
            println!("{}", token);
        }

        println!();

        parse_program(Tokens(&tokens)).and_then(|ast| {
            println!("AST:\n{}\n", ast);

            generate_program_code(&ast).and_then(|asm| {
                println!("assembly:\n{}", asm);
                let exit_code = assemble_and_link(&asm, output_exe, should_suppress_output).expect("programs should always have an exit code");
                println!("assemble status: {:?}", exit_code);
                Ok(exit_code)
            })
        })
    })
}

fn main() {
    let args : Vec<String> = env::args().collect();
    println!("loading {}", args[1]);
    let input = std::fs::read_to_string(&args[1]).unwrap();
    //println!("input: {}", input);

    let exit_code = match compile_and_link(&input, &args[2], false) {
        Ok(inner_exit_code) => inner_exit_code,
        Err(msg) => {
            println!("error! {}", msg);
            1
        },
    };

    std::process::exit(exit_code)
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

    /* TODO re-enable parse tests when the AST is a bit more stable
    fn make_factor_expression(factor : AstFactor) -> AstExpression {
        AstExpression::new(AstTerm::new(factor))
    }

    fn make_constant_term(value : u32) -> AstTerm {
        AstTerm {
            inner : AstFactor::Constant(value),
            binary_ops : vec![],
        }
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
                        make_factor_expression(AstFactor::Constant(value))
                    )
                },
            })
        );
    }

    fn test_parse_failure(input : &str) {
        assert!(parse_program(&lex_all_tokens(input).unwrap()).is_err());
    }

    #[test]
    fn parse_return_0() {
        test_parse_simple(0);
    }

    #[test]
    fn parse_return_2() {
        test_parse_simple(2);
    }

    #[test]
    fn parse_return_multi_digit() {
        test_parse_simple(12345);
    }

    #[test]
    fn parse_error_missing_open_paren() {
        test_parse_failure("int main) { return 1; }");
    }

    #[test]
    fn parse_error_missing_close_paren() {
        test_parse_failure("int main( { return 1; }");
    }

    #[test]
    fn parse_error_missing_retval() {
        test_parse_failure("int main() { return; }");
    }

    #[test]
    fn parse_error_missing_close_brace() {
        test_parse_failure("int main() { return;");
    }

    #[test]
    fn parse_error_missing_statement_semicolon() {
        test_parse_failure("int main() { return }");
        test_parse_failure("int main() { return 5 }");
        test_parse_failure("int main() { return !5 }");
    }

    #[test]
    fn parse_error_missing_statement_missing_space() {
        test_parse_failure("int main() { return0; }");
    }

    #[test]
    fn parse_error_missing_statement_return_wrong_case() {
        test_parse_failure("int main() { RETURN 0; }");
    }

    #[test]
    fn parse_error_extra_token() {
        test_parse_failure("int main() { return 0; }}");
    }

    #[test]
    fn parse_error_missing_unary_operand() {
        test_parse_failure("int main() { return !; }}");
        test_parse_failure("int main() { return !-~; }}");
    }

    #[test]
    fn parse_error_unary_operand_misorder() {
        test_parse_failure("int main() { return 5!; }}");
        test_parse_failure("int main() { return 5-; }}");
        test_parse_failure("int main() { return 5~; }}");
    }

    fn test_parse_single_unary_operator(token : &str, operator : AstUnaryOperator) {
        assert_eq!(
            parse_program(&lex_all_tokens(&format!("int main() {{ return {}1; }}", token)).unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        make_factor_expression(AstFactor::UnaryOperator(operator, Box::new(AstFactor::Constant(1))))
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
                        make_factor_expression(
                            AstFactor::UnaryOperator(
                                AstUnaryOperator::Negation,
                                Box::new(AstFactor::UnaryOperator(
                                    AstUnaryOperator::BitwiseNot,
                                    Box::new(AstFactor::UnaryOperator(
                                        AstUnaryOperator::LogicalNot,
                                        Box::new(AstFactor::Constant(1))
                                    ))
                                ))
                            )
                        )
                    )
                },
            })
        );
    }

    #[test]
    fn test_parse_single_binary_expression_operation() {
        assert_eq!(
            parse_program(&lex_all_tokens("int main() { return 3 + 4; }").unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression {
                            inner : make_constant_term(3),
                            binary_ops : vec![
                                AstAdditiveExpressionBinaryOperation {
                                    operator : AstAdditiveExpressionBinaryOperator::Plus,
                                    rhs : make_constant_term(4),
                                },
                            ],
                        }
                    )
                },
            })
        );
    }

    #[test]
    fn test_parse_multi_binary_expression_operation() {
        assert_eq!(
            parse_program(&lex_all_tokens("int main() { return 3 + 4 - 5; }").unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression {
                            inner : make_constant_term(3),
                            binary_ops : vec![
                                AstAdditiveExpressionBinaryOperation {
                                    operator : AstAdditiveExpressionBinaryOperator::Plus,
                                    rhs : make_constant_term(4),
                                },
                                AstAdditiveExpressionBinaryOperation {
                                    operator : AstAdditiveExpressionBinaryOperator::Minus,
                                    rhs : make_constant_term(5),
                                },
                            ],
                        }
                    )
                },
            })
        );
    }

    #[test]
    fn test_parse_single_binary_term_operation() {
        assert_eq!(
            parse_program(&lex_all_tokens("int main() { return 3 * 4; }").unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression::new(
                            AstTerm {
                                inner : AstFactor::Constant(3),
                                binary_ops : vec![
                                    AstTermBinaryOperation {
                                        operator : AstTermBinaryOperator::Multiply,
                                        rhs : AstFactor::Constant(4)
                                    },
                                ],
                            },
                        )
                    )
                },
            })
        );
    }

    #[test]
    fn test_parse_multi_binary_term_operation() {
        assert_eq!(
            parse_program(&lex_all_tokens("int main() { return 3 * 4 / 5; }").unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression::new(
                            AstTerm {
                                inner : AstFactor::Constant(3),
                                binary_ops : vec![
                                    AstTermBinaryOperation {
                                        operator : AstTermBinaryOperator::Multiply,
                                        rhs : AstFactor::Constant(4)
                                    },
                                    AstTermBinaryOperation {
                                        operator : AstTermBinaryOperator::Divide,
                                        rhs : AstFactor::Constant(5)
                                    },
                                ],
                            },
                        )
                    )
                },
            })
        );
    }

    #[test]
    fn test_parse_multi_binary_expression_and_term_operations() {
        assert_eq!(
            parse_program(&lex_all_tokens("int main() { return 3 * 4 / 5 + 6 * 7 / 8 - 11 / 10 * 9; }").unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression {
                            inner : AstTerm {
                                inner : AstFactor::Constant(3),
                                binary_ops : vec![
                                    AstTermBinaryOperation {
                                        operator : AstTermBinaryOperator::Multiply,
                                        rhs : AstFactor::Constant(4)
                                    },
                                    AstTermBinaryOperation {
                                        operator : AstTermBinaryOperator::Divide,
                                        rhs : AstFactor::Constant(5)
                                    },
                                ],
                            },
                            binary_ops : vec![
                                AstAdditiveExpressionBinaryOperation {
                                    operator : AstAdditiveExpressionBinaryOperator::Plus,
                                    rhs : AstTerm {
                                        inner : AstFactor::Constant(6),
                                        binary_ops : vec![
                                            AstTermBinaryOperation {
                                                operator : AstTermBinaryOperator::Multiply,
                                                rhs : AstFactor::Constant(7)
                                            },
                                            AstTermBinaryOperation {
                                                operator : AstTermBinaryOperator::Divide,
                                                rhs : AstFactor::Constant(8)
                                            },
                                        ],
                                    },
                                },
                                AstAdditiveExpressionBinaryOperation {
                                    operator : AstAdditiveExpressionBinaryOperator::Minus,
                                    rhs : AstTerm {
                                        inner : AstFactor::Constant(11),
                                        binary_ops : vec![
                                            AstTermBinaryOperation {
                                                operator : AstTermBinaryOperator::Divide,
                                                rhs : AstFactor::Constant(10)
                                            },
                                            AstTermBinaryOperation {
                                                operator : AstTermBinaryOperator::Multiply,
                                                rhs : AstFactor::Constant(9)
                                            },
                                        ],
                                    },
                                },
                            ],
                        }
                    )
                },
            })
        );
    }

    #[test]
    fn test_parse_binary_operator_grouping() {
        assert_eq!(
            parse_program(&lex_all_tokens("int main() { return 3 * (4 + 5); }").unwrap()),
            Ok(AstProgram {
                main_function: AstFunction {
                    name: "main",
                    body: AstStatement::Return(
                        AstExpression {
                            inner : AstTerm {
                                inner : AstFactor::Constant(3),
                                binary_ops : vec![
                                    AstTermBinaryOperation {
                                        operator : AstTermBinaryOperator::Multiply,
                                        rhs : AstFactor::Expression(
                                             Box::new(AstExpression {
                                                 inner : make_constant_term(4),
                                                 binary_ops : vec![
                                                     AstAdditiveExpressionBinaryOperation {
                                                         operator : AstAdditiveExpressionBinaryOperator::Plus,
                                                         rhs : make_constant_term(5)
                                                     },
                                                 ],
                                             })
                                        )
                                    },
                                ],
                            },
                            binary_ops : vec![]
                        }
                    )
                },
            })
        );
    }
    */

    fn codegen_run_and_check_exit_code_or_compile_failure(input : &str, expected_result : Option<i32>) {
        let exe_name = format!("test_{}.exe", generate_random_string(8));
        let mut pdb_path = Path::new(&exe_name).to_path_buf();
        pdb_path.set_extension("pdb");

        let compile_result = compile_and_link(input, &exe_name, true);

        if compile_result.is_ok() {
            if let Some(expected_exit_code) = expected_result {
                let actual_exit_code = Command::new(&exe_name)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .expect("failed to run test.exe")
                    .code()
                    .expect("all processes must have exit code");

                assert_eq!(expected_exit_code, actual_exit_code);
                std::fs::remove_file(exe_name);
                std::fs::remove_file(&pdb_path);
            } else {
                assert!(false, "compile succeeded but expected failure");
            }
        } else {
            assert!(expected_result.is_none());
        }
    }

    fn codegen_run_and_check_exit_code(input : &str, expected_exit_code : i32) {
        codegen_run_and_check_exit_code_or_compile_failure(input, Some(expected_exit_code))
    }

    fn codegen_run_and_expect_compile_failure(input : &str) {
        codegen_run_and_check_exit_code_or_compile_failure(input, None)
    }

    fn test_codegen_expression(expression : &str, expected_exit_code : i32) {
        codegen_run_and_check_exit_code(&format!("int main() {{ return {}; }}", expression), expected_exit_code);
    }

    fn test_codegen_mainfunc(body : &str, expected_exit_code : i32) {
        codegen_run_and_check_exit_code(&format!("int main() {{ {} }}", body), expected_exit_code);
    }

    fn test_codegen_mainfunc_failure(body : &str) {
        codegen_run_and_expect_compile_failure(&format!("int main() {{ {} }}", body))
    }

    #[test]
    fn test_codegen_unary_operator() {
        test_codegen_expression("-5", -5);
    }

    #[test]
    fn test_codegen_expression_binary_operation() {
        test_codegen_expression("5 + 6", 11);
    }

    #[test]
    fn test_codegen_expression_negative_divide() {
        test_codegen_expression("-110 / 10", -11);
    }

    #[test]
    fn test_codegen_expression_negative_multiply() {
        test_codegen_expression("10 * -11", -110);
    }

    #[test]
    fn test_codegen_expression_factors_and_terms() {
        test_codegen_expression("(1 + 2 + 3 + 4) * (10 - 21)", -110);
    }

    #[test]
    fn test_codegen_relational_lt() {
        test_codegen_expression("1234 < 1234", 0);
        test_codegen_expression("1234 < 1235", 1);
    }

    #[test]
    fn test_codegen_relational_gt() {
        test_codegen_expression("1234 > 1234", 0);
        test_codegen_expression("1234 > 1233", 1);
    }

    #[test]
    fn test_codegen_relational_le() {
        test_codegen_expression("1234 <= 1234", 1);
        test_codegen_expression("1234 <= 1233", 0);
    }

    #[test]
    fn test_codegen_relational_ge() {
        test_codegen_expression("1234 >= 1234", 1);
        test_codegen_expression("1234 >= 1235", 0);
    }

    #[test]
    fn test_codegen_equality_eq() {
        test_codegen_expression("1234 == 1234", 1);
        test_codegen_expression("1234 == 1235", 0);
    }

    #[test]
    fn test_codegen_equality_ne() {
        test_codegen_expression("1234 != 1234", 0);
        test_codegen_expression("1234 != 1235", 1);
    }

    #[test]
    fn test_codegen_logical_and() {
        test_codegen_expression("0 && 1 && 2", 0);
        test_codegen_expression("5 && 6 && 7", 1);
        test_codegen_expression("5 && 6 && 0", 0);
    }

    #[test]
    fn test_codegen_logical_or() {
        test_codegen_expression("0 || 0 || 1", 1);
        test_codegen_expression("1 || 0 || 0", 1);
        test_codegen_expression("0 || 0 || 0", 0);
    }

    #[test]
    fn test_codegen_operator_precedence() {
        test_codegen_expression("-1 * -2 + 3 >= 5 == 1 && (6 - 6) || 7", 1);
    }

    #[test]
    fn test_codegen_var_use() {
        test_codegen_mainfunc("int x = 5; int y = 6; int z; x = 1; z = 3; return x + y + z;", 10);
    }

    #[test]
    fn test_codegen_assign_expr() {
        test_codegen_mainfunc("int x = 5; int y = x = 3 + 1; return x + y;", 8);
    }

    #[test]
    fn test_codegen_duplicate_variable() {
        test_codegen_mainfunc_failure("int x = 5; int x = 4; return x;");
    }

    #[test]
    fn test_codegen_unknown_variable() {
        test_codegen_mainfunc_failure("return x;");
    }
}
