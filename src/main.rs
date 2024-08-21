#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate regex;

use {
    clap::Parser,
    rand::distributions::Alphanumeric,
    rand::{thread_rng, Rng},
    regex::Regex,
    std::{collections::HashMap, env, fmt, ops::Deref, path::*, process::*},
};

const VARIABLE_SIZE: u32 = 8;
const RCX_SP_OFFSET: u32 = VARIABLE_SIZE * 1;
const RDX_SP_OFFSET: u32 = VARIABLE_SIZE * 2;
const R8_SP_OFFSET: u32 = VARIABLE_SIZE * 3;
const R9_SP_OFFSET: u32 = VARIABLE_SIZE * 4;

fn generate_random_string(len: usize) -> String {
    thread_rng()
        .sample_iter(&Alphanumeric)
        .map(char::from)
        .take(len)
        .collect()
}

// Convenience method for formatting an io::Error to String.
fn format_io_err(err: std::io::Error) -> String {
    format!("{}: {}", err.kind(), err)
}

trait AstToString {
    fn ast_to_string(&self, indent_levels: u32) -> String;

    fn get_indent_string(indent_levels: u32) -> String {
        let mut result = String::new();
        for _ in 0..indent_levels {
            result += "    ";
        }
        result
    }
}

// A wrapper around a slice of tokens with convenience functions useful for parsing.
#[derive(PartialEq, Clone, Debug)]
struct Tokens<'i, 't>(&'t [&'i str]);

#[derive(PartialEq, Clone, Debug)]
struct AstProgram<'i> {
    functions: Vec<AstFunction<'i>>,
}

#[derive(PartialEq, Clone, Debug)]
struct AstFunction<'i> {
    name: &'i str,
    parameters: Vec<String>,
    body_opt: Option<Vec<AstBlockItem>>,
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
    Expression(Option<AstExpression>),
    Conditional(AstExpression, Box<AstStatement>, Option<Box<AstStatement>>),
    Compound(Box<Vec<AstBlockItem>>),
    For(
        Option<AstExpression>,
        AstExpression,
        Option<AstExpression>,
        Box<AstStatement>,
    ),
    ForDecl(
        AstDeclaration,
        AstExpression,
        Option<AstExpression>,
        Box<AstStatement>,
    ),
    While(AstExpression, Box<AstStatement>),
    DoWhile(AstExpression, Box<AstStatement>),
    Break,
    Continue,
}

#[derive(PartialEq, Clone, Debug)]
enum AstUnaryOperator {
    Negation,
    BitwiseNot,
    LogicalNot,
}

#[derive(PartialEq, Clone, Debug)]
enum AstBinaryOperator {
    Or,
    And,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
    Plus,
    Minus,
    Multiply,
    Divide,
}

#[derive(PartialEq, Clone, Debug)]
enum AstExpression {
    Constant(u32),
    Variable(String), // TODO use a string slice
    UnaryOperator(AstUnaryOperator, Box<AstExpression>),
    BinaryOperator(AstBinaryOperator, Box<AstExpression>, Box<AstExpression>),
    Assign(String, Box<AstExpression>),
    Conditional(Box<AstExpression>, Box<AstExpression>, Box<AstExpression>),
    FuncCall(String, Vec<AstExpression>),
}

#[derive(Clone, PartialEq)]
struct CodegenGlobalState {
    next_label: u32,
}

#[derive(Clone, PartialEq, Eq)]
struct StackVariable {
    offset_from_base: i32,
    block_id: u32,
}

#[derive(Clone)]
struct CodegenBlockState {
    variables: HashMap<String, StackVariable>,
    next_local_offset_from_base: i32,
    next_arg_to_func_offset_from_base: u32,
    next_func_call_arg_offset_from_sp: u32,
    lowest_local_offet_from_base: i32,
    frame_size: Option<u32>,
    block_id: u32,
    break_label: Option<u32>,
    continue_label: Option<u32>,
}

impl<'i, 't> Tokens<'i, 't> {
    fn consume_tokens(
        &self,
        num_tokens: usize,
    ) -> Result<(Tokens<'i, 't>, Tokens<'i, 't>), String> {
        if self.0.len() >= num_tokens {
            let (tokens, remaining_tokens) = self.0.split_at(num_tokens);
            Ok((Tokens(tokens), Tokens(remaining_tokens)))
        } else {
            Err(format!("could not find {} more token(s)", num_tokens))
        }
    }

    fn consume_expected_next_token(&mut self, expected_token: &str) -> Result<&mut Self, String> {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        if tokens[0] == expected_token {
            *self = remaining_tokens;
            //println!("consumed expected next token {}", expected_token);
            Ok(self)
        } else {
            //println!("expected next token \"{}\" but found \"{}\"", expected_token, tokens[0]);
            Err(format!(
                "expected next token \"{}\" but found \"{}\"",
                expected_token, tokens[0]
            ))
        }
    }

    fn consume_next_token(&mut self) -> Result<&'i str, String> {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;
        *self = remaining_tokens;
        Ok(tokens[0])
    }

    fn consume_identifier(&mut self) -> Result<&'i str, String> {
        fn is_token_identifier(token: &str) -> bool {
            lazy_static! {
                static ref IDENT_REGEX: Regex =
                    Regex::new(r"^[a-zA-Z]\w*$").expect("failed to compile regex");
            }

            IDENT_REGEX.find(token).is_some()
        }

        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        if is_token_identifier(tokens[0]) {
            *self = remaining_tokens;
            //println!("consumed identifier {}", tokens[0]);
            Ok(tokens[0])
        } else {
            Err(format!("token \"{}\" is not a variable name", tokens[0]))
        }
    }

    fn consume_and_parse_next_token<T>(&mut self) -> Result<T, String>
    where
        T: std::str::FromStr,
    {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        let result = tokens[0]
            .parse::<T>()
            .or(Err(format!("token \"{}\" can't be parsed", tokens[0])));

        if result.is_ok() {
            *self = remaining_tokens;
        }

        result
    }

    fn consume_and_parse_next_binary_operator(
        &mut self,
        required_precedence_level: u32,
    ) -> Result<AstBinaryOperator, String> {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        let operator = tokens[0].parse::<AstBinaryOperator>()?;
        let operator_level = operator.get_precedence_level();
        assert!(operator_level <= MAX_EXPRESSION_LEVEL);
        if operator_level == required_precedence_level {
            *self = remaining_tokens;
            Ok(operator)
        } else {
            Err(format!(
                "required precedence level {} doesn't match operator {} precedence level {}",
                required_precedence_level, tokens[0], operator_level
            ))
        }
    }
}

impl<'i, 't> Deref for Tokens<'i, 't> {
    type Target = &'t [&'i str];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'i> AstProgram<'i> {
    fn lookup_function_definition(&'i self, name: &str) -> Option<&'i AstFunction<'i>> {
        for func in &self.functions {
            if func.name == name {
                return Some(func);
            }
        }

        return None;
    }

    fn collect_all_expressions(&self) -> Vec<&AstExpression> {
        let mut expressions = vec![];
        for func in &self.functions {
            // Only definitions can have statements in them that need to be scanned for expressions.
            if !func.is_definition() {
                continue;
            }

            for block_item in func.body_opt.as_ref().unwrap().iter() {
                Self::add_expressions_from_block_item(block_item, &mut expressions);
            }
        }

        expressions
    }

    fn add_expressions_from_block_item<'p>(
        block_item: &'p AstBlockItem,
        expressions: &mut Vec<&'p AstExpression>,
    ) {
        match block_item {
            AstBlockItem::Declaration(decl) => {
                Self::add_expressions_from_declaration(decl, expressions);
            }
            AstBlockItem::Statement(stmt) => {
                Self::add_expressions_from_statement(stmt, expressions);
            }
        }
    }

    fn add_expressions_from_declaration<'p>(
        declaration: &'p AstDeclaration,
        expressions: &mut Vec<&'p AstExpression>,
    ) {
        if let AstDeclaration::DeclareVar(_, Some(expr)) = declaration {
            Self::add_expressions_from_expression(expr, expressions);
        }
    }

    fn add_expressions_from_statement<'p>(
        statement: &'p AstStatement,
        expressions: &mut Vec<&'p AstExpression>,
    ) {
        match statement {
            AstStatement::Return(expr) => Self::add_expressions_from_expression(expr, expressions),
            AstStatement::Expression(expr_opt) => {
                if let Some(expr) = expr_opt {
                    Self::add_expressions_from_expression(expr, expressions);
                }
            }
            AstStatement::Conditional(cond, pos_stmt, neg_stmt_opt) => {
                Self::add_expressions_from_expression(cond, expressions);
                Self::add_expressions_from_statement(&pos_stmt, expressions);
                if let Some(neg_stmt) = neg_stmt_opt {
                    Self::add_expressions_from_statement(neg_stmt, expressions);
                }
            }
            AstStatement::Compound(block_items) => {
                for block_item in block_items.iter() {
                    Self::add_expressions_from_block_item(block_item, expressions);
                }
            }
            AstStatement::ForDecl(decl, cond, post_opt, body) => {
                Self::add_expressions_from_declaration(&decl, expressions);

                Self::add_expressions_from_expression(cond, expressions);

                if let Some(post) = post_opt {
                    Self::add_expressions_from_expression(post, expressions);
                }

                Self::add_expressions_from_statement(body, expressions);
            }
            AstStatement::For(pre_opt, cond, post_opt, body) => {
                if let Some(pre) = pre_opt {
                    Self::add_expressions_from_expression(pre, expressions);
                }

                Self::add_expressions_from_expression(cond, expressions);

                if let Some(post) = post_opt {
                    Self::add_expressions_from_expression(post, expressions);
                }

                Self::add_expressions_from_statement(body, expressions);
            }
            AstStatement::While(cond, body) => {
                Self::add_expressions_from_expression(cond, expressions);
                Self::add_expressions_from_statement(body, expressions);
            }
            AstStatement::DoWhile(cond, body) => {
                Self::add_expressions_from_expression(cond, expressions);
                Self::add_expressions_from_statement(body, expressions);
            }
            AstStatement::Break => {}
            AstStatement::Continue => {}
        }
    }

    fn add_expressions_from_expression<'p>(
        expression: &'p AstExpression,
        expressions: &mut Vec<&'p AstExpression>,
    ) {
        expressions.push(expression);

        match expression {
            AstExpression::Constant(_) | AstExpression::Variable(_) => expressions.push(expression),
            AstExpression::UnaryOperator(_, expr) => {
                Self::add_expressions_from_expression(expr, expressions)
            }
            AstExpression::BinaryOperator(_, expr1, expr2) => {
                Self::add_expressions_from_expression(expr1, expressions);
                Self::add_expressions_from_expression(expr2, expressions);
            }
            AstExpression::Assign(_, expr) => {
                Self::add_expressions_from_expression(expr, expressions)
            }
            AstExpression::Conditional(expr1, expr2, expr3) => {
                Self::add_expressions_from_expression(expr1, expressions);
                Self::add_expressions_from_expression(expr2, expressions);
                Self::add_expressions_from_expression(expr3, expressions);
            }
            AstExpression::FuncCall(_, args) => {
                for arg in args.iter() {
                    Self::add_expressions_from_expression(arg, expressions);
                }
            }
        }
    }
}

const MAX_EXPRESSION_LEVEL: u32 = 6;
impl AstBinaryOperator {
    fn get_precedence_level(&self) -> u32 {
        match self {
            AstBinaryOperator::Or => 6,
            AstBinaryOperator::And => 5,
            AstBinaryOperator::Equals => 4,
            AstBinaryOperator::NotEquals => 4,
            AstBinaryOperator::LessThan => 3,
            AstBinaryOperator::GreaterThan => 3,
            AstBinaryOperator::LessThanEqual => 3,
            AstBinaryOperator::GreaterThanEqual => 3,
            AstBinaryOperator::Plus => 2,
            AstBinaryOperator::Minus => 2,
            AstBinaryOperator::Multiply => 1,
            AstBinaryOperator::Divide => 1,
        }
    }
}

impl<'i> AstFunction<'i> {
    fn is_definition(&self) -> bool {
        self.body_opt.is_some()
    }
}

impl<'i> AstToString for AstProgram<'i> {
    fn ast_to_string(&self, _indent_levels: u32) -> String {
        self.functions
            .iter()
            .map(|function| function.ast_to_string(0))
            .collect::<Vec<String>>()
            .join("\n\n")
    }
}

impl<'i> fmt::Display for AstProgram<'i> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ast_to_string(0))
    }
}

impl<'i> AstToString for AstFunction<'i> {
    fn ast_to_string(&self, indent_levels: u32) -> String {
        fn format_parameter_list(parameters: &Vec<String>) -> String {
            parameters
                .iter()
                .map(|param| format!("int {}", param))
                .collect::<Vec<String>>()
                .join(", ")
        }

        if let Some(body) = self.body_opt.as_ref() {
            let mut body_str = String::new();
            for block_item in body {
                let result = block_item.ast_to_string(indent_levels);
                if result.len() != 0 {
                    body_str += "\n";
                    body_str += &result;
                }
            }

            format!(
                "{}FUNC {}({}):{}",
                Self::get_indent_string(indent_levels),
                self.name,
                &format_parameter_list(&self.parameters),
                &body_str
            )
        } else {
            format!(
                "{}FUNC DECL {}({});",
                Self::get_indent_string(indent_levels),
                self.name,
                &format_parameter_list(&self.parameters)
            )
        }
    }
}

impl AstToString for AstBlockItem {
    fn ast_to_string(&self, indent_levels: u32) -> String {
        match self {
            AstBlockItem::Statement(statement) => statement.ast_to_string(indent_levels + 1),
            AstBlockItem::Declaration(declaration) => declaration.ast_to_string(indent_levels + 1),
        }
    }
}

impl AstToString for AstDeclaration {
    fn ast_to_string(&self, indent_levels: u32) -> String {
        if let AstDeclaration::DeclareVar(name, expr_opt) = self {
            if let Some(expr) = expr_opt {
                format!(
                    "{}int {} = {};",
                    Self::get_indent_string(indent_levels),
                    &name,
                    expr.ast_to_string(indent_levels + 1)
                )
            } else {
                format!("{}int {};", Self::get_indent_string(indent_levels), &name)
            }
        } else {
            format!("{}err {:?}", Self::get_indent_string(indent_levels), self)
        }
    }
}

impl AstToString for AstStatement {
    fn ast_to_string(&self, indent_levels: u32) -> String {
        match self {
            AstStatement::Return(expr) => format!(
                "{}return {};",
                Self::get_indent_string(indent_levels),
                expr.ast_to_string(indent_levels + 1)
            ),
            AstStatement::Expression(Some(expr)) => format!(
                "{}{};",
                Self::get_indent_string(indent_levels),
                expr.ast_to_string(indent_levels + 1)
            ),
            AstStatement::Expression(None) => String::new(),
            AstStatement::Conditional(expr, positive, negative_opt) => {
                let pos_indent_levels = if let AstStatement::Compound(_) = **positive {
                    indent_levels
                } else {
                    indent_levels + 1
                };

                let mut result = format!(
                    "{}if ({})\n{}",
                    Self::get_indent_string(indent_levels),
                    expr.ast_to_string(0),
                    positive.ast_to_string(pos_indent_levels)
                );

                if let Some(negative) = negative_opt {
                    let neg_indent_levels = if let AstStatement::Compound(_) = **negative {
                        indent_levels
                    } else {
                        indent_levels + 1
                    };
                    result += &format!(
                        "\n{}else\n{}",
                        Self::get_indent_string(indent_levels),
                        negative.ast_to_string(neg_indent_levels)
                    );
                }
                result
            }
            AstStatement::Compound(block_items) => {
                let mut result = String::new();
                for block_item in block_items.iter() {
                    if result.len() != 0 {
                        result += "\n";
                    }
                    result += &block_item.ast_to_string(indent_levels);
                }
                result
            }
            AstStatement::For(expr_opt, condition, post_expr_opt, body) => {
                format!(
                    "{}for ({}; {}; {})\n{}",
                    Self::get_indent_string(indent_levels),
                    expr_opt
                        .as_ref()
                        .map_or(String::new(), |expr| { expr.ast_to_string(0) }),
                    condition.ast_to_string(0),
                    post_expr_opt
                        .as_ref()
                        .map_or(String::new(), |expr| { expr.ast_to_string(0) }),
                    body.ast_to_string(indent_levels)
                )
            }
            AstStatement::ForDecl(declaration, condition, post_expr_opt, body) => {
                // Omit the semicolon after the declaration because it's built into the delcaration output itself.
                format!(
                    "{}for ({} {}; {})\n{}",
                    Self::get_indent_string(indent_levels),
                    declaration.ast_to_string(0),
                    condition.ast_to_string(0),
                    post_expr_opt
                        .as_ref()
                        .map_or(String::new(), |expr| { expr.ast_to_string(0) }),
                    body.ast_to_string(indent_levels)
                )
            }
            AstStatement::While(condition, body) => {
                format!(
                    "{}while ({})\n{}",
                    Self::get_indent_string(indent_levels),
                    condition.ast_to_string(0),
                    body.ast_to_string(indent_levels)
                )
            }
            AstStatement::DoWhile(condition, body) => {
                format!(
                    "{}do while ({})\n{}",
                    Self::get_indent_string(indent_levels),
                    condition.ast_to_string(0),
                    body.ast_to_string(indent_levels)
                )
            }
            AstStatement::Break => {
                format!("{}break;", Self::get_indent_string(indent_levels))
            }
            AstStatement::Continue => {
                format!("{}continue;", Self::get_indent_string(indent_levels))
            }
        }
    }
}

impl AstToString for AstUnaryOperator {
    fn ast_to_string(&self, _indent_levels: u32) -> String {
        String::from(match self {
            AstUnaryOperator::Negation => "-",
            AstUnaryOperator::BitwiseNot => "~",
            AstUnaryOperator::LogicalNot => "!",
        })
    }
}

impl AstToString for AstBinaryOperator {
    fn ast_to_string(&self, _indent_levels: u32) -> String {
        String::from(match self {
            AstBinaryOperator::Or => "||",
            AstBinaryOperator::And => "&&",
            AstBinaryOperator::Equals => "==",
            AstBinaryOperator::NotEquals => "!=",
            AstBinaryOperator::LessThan => "<",
            AstBinaryOperator::GreaterThan => ">",
            AstBinaryOperator::LessThanEqual => "<=",
            AstBinaryOperator::GreaterThanEqual => ">=",
            AstBinaryOperator::Plus => "+",
            AstBinaryOperator::Minus => "-",
            AstBinaryOperator::Multiply => "*",
            AstBinaryOperator::Divide => "/",
        })
    }
}

impl AstToString for AstExpression {
    fn ast_to_string(&self, indent_levels: u32) -> String {
        match self {
            AstExpression::Constant(val) => format!("{}", val),
            AstExpression::Variable(name) => name.clone(),
            AstExpression::UnaryOperator(operator, expr) => {
                format!("{}{}", operator.ast_to_string(0), expr.ast_to_string(0))
            }
            AstExpression::Assign(name, expr) => {
                format!("{} = {}", name, expr.ast_to_string(indent_levels))
            }
            AstExpression::BinaryOperator(operator, left, right) => format!(
                "({}) {} ({})",
                left.ast_to_string(0),
                operator.ast_to_string(0),
                right.ast_to_string(0)
            ),
            AstExpression::Conditional(condition, positive, negative) => format!(
                "({}) ? ({}) : ({})",
                condition.ast_to_string(0),
                positive.ast_to_string(0),
                negative.ast_to_string(0)
            ),
            AstExpression::FuncCall(name, args) => {
                let args_string = args
                    .iter()
                    .map(|arg| arg.ast_to_string(0))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({})", name, args_string)
            }
        }
    }
}

impl std::str::FromStr for AstUnaryOperator {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "-" => Ok(AstUnaryOperator::Negation),
            "~" => Ok(AstUnaryOperator::BitwiseNot),
            "!" => Ok(AstUnaryOperator::LogicalNot),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl std::str::FromStr for AstBinaryOperator {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "||" => Ok(AstBinaryOperator::Or),
            "&&" => Ok(AstBinaryOperator::And),
            "==" => Ok(AstBinaryOperator::Equals),
            "!=" => Ok(AstBinaryOperator::NotEquals),
            "<" => Ok(AstBinaryOperator::LessThan),
            ">" => Ok(AstBinaryOperator::GreaterThan),
            "<=" => Ok(AstBinaryOperator::LessThanEqual),
            ">=" => Ok(AstBinaryOperator::GreaterThanEqual),
            "+" => Ok(AstBinaryOperator::Plus),
            "-" => Ok(AstBinaryOperator::Minus),
            "*" => Ok(AstBinaryOperator::Multiply),
            "/" => Ok(AstBinaryOperator::Divide),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl CodegenGlobalState {
    fn new() -> CodegenGlobalState {
        CodegenGlobalState { next_label: 0 }
    }

    fn consume_jump_label(&mut self) -> u32 {
        self.next_label += 1;
        self.next_label - 1
    }
}

impl CodegenBlockState {
    fn new(func: &AstFunction, frame_size: Option<u32>) -> CodegenBlockState {
        let mut block_state = CodegenBlockState {
            variables: HashMap::new(),

            // When entering a new function, the base pointer points to the return address to the caller. The first
            // available location to allocate a new stack variable is rbp-8.
            next_local_offset_from_base: -(VARIABLE_SIZE as i32),

            // The first parameter to the callee is at rbp+8, because rbp+0 is the return address.
            next_arg_to_func_offset_from_base: VARIABLE_SIZE,

            // Arguments to a function called by this one. When adjusting this, first we reserve stack space for
            // the whole function, but ultimately at the point of the call instruction, the first argument is at
            // rsp+0, next is at rsp+8, etc..
            next_func_call_arg_offset_from_sp: 0,

            // Keep track of the lowest local or temp space offset from the base pointer, which represents the size
            // of this stack frame.
            lowest_local_offet_from_base: 0,

            // In order to correctly determine the frame size, code generation needs to make two passes over the
            // contents of a function. In the first pass, it runs through all the variable and parameter tracking
            // logic to determine the frame size needed. The frame size determines the rsp adjustment in the
            // function prologue, so in the second pass the correct rsp offsets of variables and temp locations can
            // be emitted.
            frame_size,

            // Reserve block_id 0 specially for function parameters. All local variables go to block_id 1 or higher.
            block_id: 1,
            break_label: None,
            continue_label: None,
        };

        let mut arg_index = 0;
        for arg_name in func.parameters.iter() {
            block_state.add_param(arg_name);
            arg_index += 1;
        }

        block_state
    }

    fn new_for_scan(func: &AstFunction) -> CodegenBlockState {
        CodegenBlockState::new(func, None)
    }

    fn nest(&self) -> CodegenBlockState {
        let mut nested = self.clone();
        nested.block_id += 1;
        nested
    }

    fn enter_loop(&self, break_label: u32, continue_label: u32) -> CodegenBlockState {
        let mut loop_state = self.clone();
        loop_state.break_label = Some(break_label);
        loop_state.continue_label = Some(continue_label);
        loop_state
    }

    fn consume_variable_slot(&mut self) {
        if self.next_local_offset_from_base < self.lowest_local_offet_from_base {
            self.lowest_local_offet_from_base = self.next_local_offset_from_base;
        }

        self.next_local_offset_from_base -= VARIABLE_SIZE as i32;
    }

    fn update_lowest_local_offset_from_base_from_nested(&mut self, nested: &CodegenBlockState) {
        self.lowest_local_offet_from_base = std::cmp::min(
            self.lowest_local_offet_from_base,
            nested.lowest_local_offet_from_base,
        );
    }

    fn add_stack_var(&mut self, name: &str) -> bool {
        if let Some(var) = self.variables.get(name) {
            if var.block_id == self.block_id || var.block_id == 0 {
                // Cannot declare the same variable twice in the same block, and cannot declare variables in any block
                // that clash with function parameter names.
                return false;
            }
            // Else it was declared in an outer block already, so below we will overwrite it with the current block.
        }

        // This will either insert a new variable or overwrite a previously declared variable.
        self.variables.insert(
            String::from(name),
            StackVariable {
                offset_from_base: self.next_local_offset_from_base,
                block_id: self.block_id,
            },
        );
        self.consume_variable_slot();
        true
    }

    fn add_param(&mut self, name: &str) -> bool {
        if let Some(var) = self.variables.get(name) {
            // Cannot declare the same parameter name twice.
            assert_eq!(var.block_id, 0);
            return false;
        }

        // Always add parameters with block id 0 so we can prevent local variables from clashing with them.
        self.variables.insert(
            String::from(name),
            StackVariable {
                offset_from_base: self.next_arg_to_func_offset_from_base as i32,
                block_id: 0,
            },
        );
        self.next_arg_to_func_offset_from_base += VARIABLE_SIZE;
        true
    }

    fn push_temp(&mut self) -> u32 {
        let offset_from_sp =
            self.translate_offset_from_base_to_offset_from_sp(self.next_local_offset_from_base);
        self.consume_variable_slot();
        offset_from_sp
    }

    fn pop_temp(&mut self) -> u32 {
        self.next_local_offset_from_base += VARIABLE_SIZE as i32;
        let ret =
            self.translate_offset_from_base_to_offset_from_sp(self.next_local_offset_from_base);
        ret
    }

    fn push_arg(&mut self) -> u32 {
        // Arguments to a function call are pushed in reverse order with the first argument at rsp+0. Make sure to
        // reserve temp space so the frame is big enough to hold them.
        self.push_temp();

        let offset_from_sp = self.next_func_call_arg_offset_from_sp;
        self.next_func_call_arg_offset_from_sp += VARIABLE_SIZE;
        offset_from_sp
    }

    fn pop_arg(&mut self) {
        // Every arg to a function also has temp space reserved to make sure the frame size is big enough, so need to
        // free the space along with adjusting where the next arg is.
        self.pop_temp();
        self.next_func_call_arg_offset_from_sp -= VARIABLE_SIZE;
    }

    fn get_var_location_str(&self, name: &str) -> Option<String> {
        let offset_from_base = self.variables.get(name)?.offset_from_base;

        Some(match offset_from_base as u32 {
            RCX_SP_OFFSET => String::from("rcx"),
            RDX_SP_OFFSET => String::from("rdx"),
            R8_SP_OFFSET => String::from("r8"),
            R9_SP_OFFSET => String::from("r9"),
            _ => format!(
                "[rsp+{}]",
                self.translate_offset_from_base_to_offset_from_sp(offset_from_base)
            ),
        })
    }

    fn translate_offset_from_base_to_offset_from_sp(&self, offset_from_base: i32) -> u32 {
        // This assumes the frame size is big enough to hold all of the variables and temp locations needed by this
        // block. The first local variable is at rbp-8, which will turn into a large rsp offset, since it is "far away"
        // from rsp. The last allocated local variable will be closest to rsp. Arguments to this function will be even
        // farther from the rsp, past the rbp.
        ((self.get_frame_size() as i32) + offset_from_base) as u32
    }

    fn get_frame_size(&self) -> u32 {
        // This object is used either to just track and calculate the frame size (in which case self.frame_size is None,
        // and the result of this function is the speculative frame size, until the traversal of the function body is
        // complete), or to actually generate code, in which case the frame size is a fixed value.
        self.frame_size
            .unwrap_or(-self.lowest_local_offet_from_base as u32)
    }
}

fn lex_next_token<'i>(input: &'i str) -> Result<(&'i str, &'i str), String> {
    lazy_static! {
        static ref SKIPPED_TOKEN_REGEXES: Vec<regex::Regex> = vec![
            Regex::new(r"^\#[^\n]*").expect("failed to compile regex"),
            Regex::new(r"^\/\/[^\n]*").expect("failed to compile regex"),
        ];
        static ref TOKEN_REGEXES: Vec<regex::Regex> = vec![
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
            Regex::new(r"^\?").expect("failed to compile regex"),
            Regex::new(r"^:").expect("failed to compile regex"),
            Regex::new(r"^,").expect("failed to compile regex"),
            Regex::new(r"^[a-zA-Z]\w*").expect("failed to compile regex"),
            Regex::new(r"^[0-9]+").expect("failed to compile regex"),
        ];
    }

    for r in SKIPPED_TOKEN_REGEXES.iter() {
        if let Some(mat) = r.find(input) {
            let range = mat.range();
            //println!("match: {}, {}", range.start, range.end);
            return Ok(("", input.split_at(range.end).1));
        }
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

fn lex_all_tokens<'i>(input: &'i str) -> Result<Vec<&'i str>, Vec<String>> {
    let mut tokens: Vec<&'i str> = vec![];

    let mut remaining_input = input.trim();
    while remaining_input.len() > 0 {
        match lex_next_token(&remaining_input) {
            Ok(split) => {
                //println!("[{}], [{}]", split.0, split.1);
                //println!("token: {}", split.0);
                if !split.0.is_empty() {
                    tokens.push(split.0);
                }

                remaining_input = split.1.trim();
            }
            Err(msg) => return Err(vec![msg]),
        }
    }

    Ok(tokens)
}

fn parse_program<'i, 't>(mut tokens: Tokens<'i, 't>) -> Result<AstProgram<'i>, String> {
    let mut functions = vec![];
    while let Ok(function) = parse_function(&mut tokens) {
        functions.push(function);
    }

    if tokens.0.len() == 0 {
        Ok(AstProgram { functions })
    } else {
        Err(format!(
            "extra tokens after main function end: {:?}",
            tokens
        ))
    }
}

fn parse_function<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstFunction<'i>, String> {
    let mut tokens = original_tokens.clone();

    tokens.consume_expected_next_token("int")?;
    let name = tokens.consume_next_token()?;
    tokens.consume_expected_next_token("(")?;

    let mut parameters = vec![];
    while let Ok(parameter_name) = parse_function_parameter(&mut tokens, parameters.len() == 0) {
        parameters.push(String::from(parameter_name));
    }

    tokens.consume_expected_next_token(")")?;

    let block_items_opt;
    if tokens.consume_expected_next_token("{").is_ok() {
        // Parse out all the block items possible.
        let mut block_items = vec![];
        loop {
            let res = parse_block_item(&mut tokens);
            if let Ok(block_item) = res {
                block_items.push(block_item);
            } else {
                break;
            }
        }

        tokens.consume_expected_next_token("}")?;
        block_items_opt = Some(block_items);
    } else {
        tokens.consume_expected_next_token(";")?;
        block_items_opt = None;
    }

    *original_tokens = tokens;
    Ok(AstFunction {
        name,
        parameters,
        body_opt: block_items_opt,
    })
}

fn parse_function_parameter<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
    is_first_parameter: bool,
) -> Result<&'i str, String> {
    let mut tokens = original_tokens.clone();

    if !is_first_parameter {
        tokens.consume_expected_next_token(",")?;
    }

    tokens.consume_expected_next_token("int")?;

    let var_name = tokens.consume_next_token()?;
    *original_tokens = tokens;
    Ok(var_name)
}

fn parse_block_item<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<AstBlockItem, String> {
    if let Ok(declaration) = parse_declaration(tokens) {
        Ok(AstBlockItem::Declaration(declaration))
    } else {
        parse_statement(tokens).map(|statement| AstBlockItem::Statement(statement))
    }
}

fn parse_declaration<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<AstDeclaration, String> {
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

fn parse_statement<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstStatement, String> {
    let mut tokens = original_tokens.clone();

    let statement;
    if tokens.consume_expected_next_token(";").is_ok() {
        statement = AstStatement::Expression(None);
    } else if tokens.consume_expected_next_token("return").is_ok() {
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
    } else if tokens.consume_expected_next_token("for").is_ok() {
        tokens.consume_expected_next_token("(")?;

        if tokens.consume_expected_next_token(";").is_ok() {
            let (condition, post_expr_opt, body) = parse_remaining_for_statement(&mut tokens)?;
            statement = AstStatement::For(None, condition, post_expr_opt, Box::new(body));
        } else if let Ok(initial_declaration) = parse_declaration(&mut tokens) {
            let (condition, post_expr_opt, body) = parse_remaining_for_statement(&mut tokens)?;
            statement = AstStatement::ForDecl(
                initial_declaration,
                condition,
                post_expr_opt,
                Box::new(body),
            );
        } else {
            let initial_expression = parse_expression(&mut tokens)?;
            tokens.consume_expected_next_token(";")?;
            let (condition, post_expr_opt, body) = parse_remaining_for_statement(&mut tokens)?;
            statement = AstStatement::For(
                Some(initial_expression),
                condition,
                post_expr_opt,
                Box::new(body),
            );
        }
    } else if tokens.consume_expected_next_token("while").is_ok() {
        tokens.consume_expected_next_token("(")?;
        let condition = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;

        let body = parse_statement(&mut tokens)?;

        statement = AstStatement::While(condition, Box::new(body));
    } else if tokens.consume_expected_next_token("do").is_ok() {
        let body = parse_statement(&mut tokens)?;
        tokens.consume_expected_next_token("while")?;
        tokens.consume_expected_next_token("(")?;
        let condition = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;
        tokens.consume_expected_next_token(";")?;

        statement = AstStatement::DoWhile(condition, Box::new(body));
    } else if tokens.consume_expected_next_token("{").is_ok() {
        let mut block_items = vec![];

        while let Ok(block_item) = parse_block_item(&mut tokens) {
            block_items.push(block_item);
        }

        tokens.consume_expected_next_token("}")?;
        statement = AstStatement::Compound(Box::new(block_items));
    } else if tokens.consume_expected_next_token("break").is_ok() {
        statement = AstStatement::Break;
        tokens.consume_expected_next_token(";")?;
    } else if tokens.consume_expected_next_token("continue").is_ok() {
        statement = AstStatement::Continue;
        tokens.consume_expected_next_token(";")?;
    } else {
        statement = AstStatement::Expression(Some(parse_expression(&mut tokens)?));
        tokens.consume_expected_next_token(";")?;
    }

    *original_tokens = tokens;
    Ok(statement)
}

fn parse_remaining_for_statement<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<(AstExpression, Option<AstExpression>, AstStatement), String> {
    let mut tokens = original_tokens.clone();

    // Empty condition is turned into a constant "true" value.
    let condition;
    if tokens.consume_expected_next_token(";").is_ok() {
        condition = AstExpression::Constant(1);
    } else {
        condition = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(";")?;
    }

    let post_expr_opt;
    if tokens.consume_expected_next_token(")").is_ok() {
        post_expr_opt = None;
    } else {
        post_expr_opt = Some(parse_expression(&mut tokens)?);
        tokens.consume_expected_next_token(")")?;
    }

    let body = parse_statement(&mut tokens)?;

    *original_tokens = tokens;
    Ok((condition, post_expr_opt, body))
}

fn parse_func_call<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    let function_name = tokens.consume_identifier()?;
    tokens.consume_expected_next_token("(")?;

    let mut arguments = vec![];
    loop {
        if tokens.consume_expected_next_token(")").is_ok() {
            break;
        }

        if arguments.len() != 0 {
            tokens.consume_expected_next_token(",")?;
        }

        arguments.push(parse_expression(&mut tokens)?);
    }

    *original_tokens = tokens;
    Ok(AstExpression::FuncCall(
        function_name.to_string(),
        arguments,
    ))
}

fn parse_unary_expression<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    if let Ok(integer_literal) = tokens.consume_and_parse_next_token::<u32>() {
        *original_tokens = tokens;
        Ok(AstExpression::Constant(integer_literal))
    } else if tokens.consume_expected_next_token("(").is_ok() {
        let inner = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;
        *original_tokens = tokens;
        Ok(inner)
    } else if let Ok(operator) = tokens.consume_and_parse_next_token::<AstUnaryOperator>() {
        let inner = parse_unary_expression(&mut tokens)?;
        *original_tokens = tokens;
        Ok(AstExpression::UnaryOperator(operator, Box::new(inner)))
    } else if let Ok(func_call) = parse_func_call(&mut tokens) {
        *original_tokens = tokens;
        Ok(func_call)
    } else {
        let variable_name = tokens.consume_identifier()?;
        *original_tokens = tokens;
        Ok(AstExpression::Variable(String::from(variable_name)))
    }
}

fn parse_binary_operator_level<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
    level: u32,
) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    if level == 0 {
        let expr = parse_unary_expression(&mut tokens)?;
        *original_tokens = tokens;
        Ok(expr)
    } else {
        let mut expr = parse_binary_operator_level(&mut tokens, level - 1)?;

        while let Ok(operator) = tokens.consume_and_parse_next_binary_operator(level) {
            let rhs = parse_binary_operator_level(&mut tokens, level - 1)?;
            expr = AstExpression::BinaryOperator(operator, Box::new(expr), Box::new(rhs));
        }

        *original_tokens = tokens;
        Ok(expr)
    }
}

fn parse_assignment_expression<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    let variable_name = tokens.consume_identifier()?;
    tokens.consume_expected_next_token("=")?;

    let expr = parse_expression(&mut tokens)?;
    *original_tokens = tokens;
    Ok(AstExpression::Assign(
        String::from(variable_name),
        Box::new(expr),
    ))
}

fn parse_expression<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    if let Ok(assignment) = parse_assignment_expression(original_tokens) {
        Ok(assignment)
    } else {
        parse_conditional_expression(original_tokens)
    }
}

fn parse_conditional_expression<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    // Parse logical-or expression
    let expr1 = parse_binary_operator_level(&mut tokens, MAX_EXPRESSION_LEVEL)?;

    if tokens.consume_expected_next_token("?").is_ok() {
        // parse full expression
        let expr2 = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(":")?;
        let expr3 = parse_conditional_expression(&mut tokens)?;

        *original_tokens = tokens;
        Ok(AstExpression::Conditional(
            Box::new(expr1),
            Box::new(expr2),
            Box::new(expr3),
        ))
    } else {
        *original_tokens = tokens;
        Ok(expr1)
    }
}

fn get_register_name(register_name: &str, width: u32) -> String {
    match width {
        8 => format!("{}l", register_name),
        16 => format!("{}x", register_name),
        32 => format!("e{}x", register_name),
        64 => format!("r{}x", register_name),
        _ => panic!("unexpected register name"),
    }
}

fn generate_program_code(ast_program: &AstProgram) -> Result<String, String> {
    const HEADER: &str = r"INCLUDELIB msvcrt.lib
.DATA

.CODE
";
    const FOOTER: &str = r"
END";

    let mut asm = String::from(HEADER);
    let mut codegen_state = CodegenGlobalState::new();

    for function in &ast_program.functions {
        asm += &generate_function_code(&mut codegen_state, function)?;
        asm += "\n";
    }

    asm += FOOTER;
    Ok(asm)
}

fn generate_function_code(
    global_state: &mut CodegenGlobalState,
    ast_function: &AstFunction,
) -> Result<String, String> {
    if let Some(body) = ast_function.body_opt.as_ref() {
        let mut block_state_temp = CodegenBlockState::new_for_scan(ast_function);

        // Compute frame size
        let mut global_state_temp = global_state.clone();
        for block_item in body {
            generate_block_item_code(global_state, &mut block_state_temp, block_item);
        }

        let mut block_state =
            CodegenBlockState::new(ast_function, Some(block_state_temp.get_frame_size()));
        let mut code = format!(
            "{} PROC\n    sub rsp,{}",
            ast_function.name,
            block_state.get_frame_size()
        );

        for block_item in body {
            let result = generate_block_item_code(global_state, &mut block_state, block_item);
            if let Ok(block_item_code) = result {
                code += &block_item_code;
            } else {
                return result;
            }
        }

        // Add a default return of 0 in case the code in the function body didn't put a return statement.
        Ok(code
            + &format!(
                "\n    add rsp,{}\n    ret\n{} ENDP",
                block_state.get_frame_size(),
                ast_function.name
            ))
    } else {
        Ok(format!("EXTERN {} :PROC", ast_function.name))
    }
}

fn generate_block_item_code(
    global_state: &mut CodegenGlobalState,
    block_state: &mut CodegenBlockState,
    ast_block_item: &AstBlockItem,
) -> Result<String, String> {
    match ast_block_item {
        AstBlockItem::Statement(statement) => {
            generate_statement_code(global_state, block_state, statement)
        }
        AstBlockItem::Declaration(declaration) => {
            generate_declaration_code(global_state, block_state, declaration)
        }
    }
}

fn generate_declaration_code(
    global_state: &mut CodegenGlobalState,
    block_state: &mut CodegenBlockState,
    ast_declaration: &AstDeclaration,
) -> Result<String, String> {
    match ast_declaration {
        AstDeclaration::DeclareVar(name, expr_opt) => {
            if !block_state.add_stack_var(&name) {
                return Err(format!("variable {} already defined", name));
            }

            let mut code = String::new();
            if let Some(expr) = expr_opt {
                let result = generate_expression_code(global_state, block_state, expr);
                if let Ok(expr_code) = result {
                    code += &expr_code;

                    // The assignment expression is in rax and should be stored at the variable's location.
                    code += &format!(
                        "\n    mov {},rax ; {} <- rax",
                        block_state.get_var_location_str(&name).unwrap(),
                        &name
                    );
                } else {
                    return result;
                }
            }
            Ok(code)
        }
    }
}

fn generate_statement_code(
    global_state: &mut CodegenGlobalState,
    block_state: &mut CodegenBlockState,
    ast_statement: &AstStatement,
) -> Result<String, String> {
    match ast_statement {
        AstStatement::Return(expr) => {
            let expr_code = generate_expression_code(global_state, block_state, expr)?;
            Ok(format!(
                "{}\n    add rsp,{}\n    ret",
                expr_code,
                block_state.get_frame_size()
            ))
        }
        AstStatement::Expression(Some(expr)) => {
            generate_expression_code(global_state, block_state, expr)
        }
        AstStatement::Expression(None) => Ok(String::new()),
        AstStatement::Conditional(condition, positive, negative_opt) => {
            let after_pos_label = global_state.consume_jump_label();
            let mut code = generate_expression_code(global_state, block_state, condition)?;

            // Check if the conditional expression was true or false.
            code += "\n    cmp rax,0";

            // If it was false, jump to the section after the positive portion.
            code += &format!("\n    je _j{}", after_pos_label);

            // Otherwise, execute the positive section.
            code += "\n    ";
            code += &generate_statement_code(global_state, block_state, positive)?;

            // Check if there is an else clause.
            if let Some(negative) = negative_opt {
                let negative_code = generate_statement_code(global_state, block_state, negative)?;
                let after_neg_label = global_state.consume_jump_label();
                // At the end of the positive section, jump over the negative section so that both aren't executed.
                code += &format!("\n    jmp _j{}", after_neg_label);

                // Start of the negative section.
                code += &format!("\n    _j{}:", after_pos_label);
                code += "\n    ";
                code += &negative_code;
                code += &format!("\n    _j{}:", after_neg_label);
            } else {
                // There was no else clause, so the label after the positive section is the end.
                code += &format!("\n    _j{}:", after_pos_label);
            }

            Ok(code)
        }
        AstStatement::Compound(block_items) => {
            let mut code = String::new();
            let mut inner_block_state = block_state.nest();
            for block_item in block_items.iter() {
                code +=
                    &generate_block_item_code(global_state, &mut inner_block_state, block_item)?;
            }

            block_state.update_lowest_local_offset_from_base_from_nested(&inner_block_state);

            Ok(code)
        }
        AstStatement::For(expr_opt, condition, post_expr_opt, body) => {
            let mut code = String::new();

            // The header of the for loop is in its own scope, for declarations.
            let mut inner_block_state = block_state.nest();

            if let Some(pre_expression) = expr_opt.as_ref() {
                code += &generate_expression_code(
                    global_state,
                    &mut inner_block_state,
                    pre_expression,
                )?;
            }

            code += &generate_remaining_for_loop_code(
                global_state,
                &mut inner_block_state,
                condition,
                body,
                post_expr_opt,
            )?;

            block_state.update_lowest_local_offset_from_base_from_nested(&inner_block_state);

            Ok(code)
        }
        AstStatement::ForDecl(declaration, condition, post_expr_opt, body) => {
            let mut code = String::new();

            // The header of the for loop is in its own scope, for declarations.
            let mut inner_block_state = block_state.nest();
            code += &generate_declaration_code(global_state, &mut inner_block_state, declaration)?;
            code += &generate_remaining_for_loop_code(
                global_state,
                &mut inner_block_state,
                condition,
                body,
                post_expr_opt,
            )?;

            block_state.update_lowest_local_offset_from_base_from_nested(&inner_block_state);

            Ok(code)
        }
        AstStatement::While(condition, body) => {
            let before_loop_condition_label = global_state.consume_jump_label();
            let after_loop_label = global_state.consume_jump_label();

            let mut code = format!("\n    _j{}:", before_loop_condition_label);
            code += &generate_expression_code(global_state, block_state, condition)?;

            // Check if the conditional expression was true or false.
            code += "\n    cmp rax,0";

            // If it was false, jump to after the loop body.
            code += &format!("\n    je _j{}", after_loop_label);

            // Set up the jump labels for break and continue.
            let mut inner_block_state =
                block_state.enter_loop(after_loop_label, before_loop_condition_label);

            // The loop condition was true, so now include the body's code.
            code += "\n    ";
            code += &generate_statement_code(global_state, &mut inner_block_state, body)?;

            // At the end of the loop body, jump back to before the loop condition so it can be evaluated again.
            code += &format!("\n    jmp _j{}", before_loop_condition_label);

            // The label after the end of the loop, so the condition can jump here if false.
            code += &format!("\n    _j{}:", after_loop_label);

            block_state.update_lowest_local_offset_from_base_from_nested(&inner_block_state);

            Ok(code)
        }
        AstStatement::DoWhile(condition, body) => {
            let before_loop_body = global_state.consume_jump_label();
            let before_loop_condition_label = global_state.consume_jump_label();
            let after_loop_label = global_state.consume_jump_label();

            let mut code = format!("\n    _j{}:", before_loop_body);

            // Set up the jump labels for break and continue.
            let mut inner_block_state =
                block_state.enter_loop(after_loop_label, before_loop_condition_label);

            code += &generate_statement_code(global_state, &mut inner_block_state, body)?;

            code += &format!("\n    _j{}:", before_loop_condition_label);

            // After the loop body, check the condition.
            code += &generate_expression_code(global_state, &mut inner_block_state, condition)?;

            // Check if the conditional expression was true or false.
            code += "\n    cmp rax,0";

            // If it was true, jump back to the start of the loop body. Otherwise just continue.
            code += &format!("\n    jne _j{}", before_loop_body);
            code += &format!("\n    _j{}:", after_loop_label);

            block_state.update_lowest_local_offset_from_base_from_nested(&inner_block_state);

            Ok(code)
        }
        AstStatement::Break => {
            if let Some(break_label) = block_state.break_label {
                Ok(format!("\n    jmp _j{}", break_label))
            } else {
                Err(String::from("break statement used outside of loop"))
            }
        }
        AstStatement::Continue => {
            if let Some(continue_label) = block_state.continue_label {
                Ok(format!("\n    jmp _j{}", continue_label))
            } else {
                Err(String::from("continue statement used outside of loop"))
            }
        }
    }
}

fn generate_remaining_for_loop_code(
    global_state: &mut CodegenGlobalState,
    block_state: &mut CodegenBlockState,
    condition: &AstExpression,
    body: &AstStatement,
    post_expr_opt: &Option<AstExpression>,
) -> Result<String, String> {
    let before_loop_condition_label = global_state.consume_jump_label();
    let after_body_label = global_state.consume_jump_label();
    let after_loop_label = global_state.consume_jump_label();

    let mut code = format!("\n    _j{}:", before_loop_condition_label);
    code += &generate_expression_code(global_state, block_state, condition)?;

    // Check if the conditional expression was true or false.
    code += "\n    cmp rax,0";

    // If it was false, jump to after the loop body.
    code += &format!("\n    je _j{}", after_loop_label);

    // Set up the jump labels for break and continue.
    let mut inner_block_state = block_state.enter_loop(after_loop_label, after_body_label);

    // The loop condition was true, so now include the body's code.
    code += "\n    ";
    code += &generate_statement_code(global_state, &mut inner_block_state, body)?;

    // Just before the loop body is a label for the target of a continue statement, which is just before the post
    // expression.
    code += &format!("\n    _j{}:", after_body_label);

    // After the loop body, evaluate the post-expression, if present.
    if let Some(post_expression) = post_expr_opt.as_ref() {
        code += &generate_expression_code(global_state, &mut inner_block_state, post_expression)?;
    }

    // At the end of the loop body, jump back to before the loop condition so it can be evaluated again.
    code += &format!("\n    jmp _j{}", before_loop_condition_label);

    // The label after the end of the loop, so the condition can jump here if false.
    code += &format!("\n    _j{}:", after_loop_label);

    block_state.update_lowest_local_offset_from_base_from_nested(&inner_block_state);

    Ok(code)
}

fn generate_binary_operator_code(
    operator: &AstBinaryOperator,
    global_state: &mut CodegenGlobalState,
    block_state: &mut CodegenBlockState,
    rhs_code: &str,
    lhs_temp_location: &str,
) -> String {
    match operator {
        AstBinaryOperator::Or => {
            let label = global_state.consume_jump_label();
            format!("\n    cmp rax,0\n    mov rax,0\n    setne al\n    jne _j{}\n{}\n    cmp rax,0\n    mov rax,0\n    setne al\n    _j{}:", label, rhs_code, label)
        }
        AstBinaryOperator::And => {
            let label = global_state.consume_jump_label();
            format!("\n    cmp rax,0\n    je _j{}\n{}\n    cmp rax,0\n    mov rax,0\n    setne al\n    _j{}:", label, rhs_code, label)
        }
        _ => {
            // The left side of the operator's code was emitted just before this, and stores the result in rax. Move it
            // to temp stack space so that rax can be repurposed for the right hand side expression.
            let mut code = format!("\n    mov {},rax ; lhs_temp <- rax", lhs_temp_location);

            // Emit the right side code and then the code for the operand that combines it with the left side code from
            // its temp location.
            code += rhs_code;
            code += &match operator {
                AstBinaryOperator::Equals => format!(
                    "\n    mov r10,{}\n    cmp r10,rax\n    mov rax,0\n    sete al",
                    lhs_temp_location
                ),
                AstBinaryOperator::NotEquals => format!(
                    "\n    mov r10,{}\n    cmp r10,rax\n    mov rax,0\n    setne al",
                    lhs_temp_location
                ),
                AstBinaryOperator::LessThan => format!(
                    "\n    mov r10,{}\n    cmp r10,rax\n    mov rax,0\n    setl al",
                    lhs_temp_location
                ),
                AstBinaryOperator::GreaterThan => format!(
                    "\n    mov r10,{}\n    cmp r10,rax\n    mov rax,0\n    setg al",
                    lhs_temp_location
                ),
                AstBinaryOperator::LessThanEqual => format!(
                    "\n    mov r10,{}\n    cmp r10,rax\n    mov rax,0\n    setle al",
                    lhs_temp_location
                ),
                AstBinaryOperator::GreaterThanEqual => format!(
                    "\n    mov r10,{}\n    cmp r10,rax\n    mov rax,0\n    setge al",
                    lhs_temp_location
                ),
                AstBinaryOperator::Plus => {
                    format!("\n    mov r10,{}\n    add rax,r10", lhs_temp_location)
                }
                AstBinaryOperator::Minus => format!(
                    "\n    mov r10,{}\n    sub r10,rax\n    mov rax,r10",
                    lhs_temp_location
                ),
                AstBinaryOperator::Multiply => format!(
                    "\n    mov r10,rax\n    mov rax,{}\n    imul rax,r10",
                    lhs_temp_location
                ),
                AstBinaryOperator::Divide => format!(
                    "\n    mov r10,rax\n    mov rax,{}\n    cdq\n    idiv r10d",
                    lhs_temp_location
                ),
                AstBinaryOperator::Or => panic!("unexpected"),
                AstBinaryOperator::And => panic!("unexpected"),
            };

            code
        }
    }
}

fn generate_expression_code(
    global_state: &mut CodegenGlobalState,
    block_state: &mut CodegenBlockState,
    ast_node: &AstExpression,
) -> Result<String, String> {
    match ast_node {
        AstExpression::Constant(val) => Ok(format!("\n    mov rax,{}", val)),
        AstExpression::Variable(name) => block_state
            .get_var_location_str(&name)
            .ok_or(format!("unknown variable {}", name))
            .map(|location| format!("\n    mov rax,{} ; rax <- {}", &location, &name)),
        AstExpression::UnaryOperator(operator, expr) => generate_expression_code(
            global_state,
            block_state,
            &expr,
        )
        .and_then(|inner_factor_code| match operator {
            AstUnaryOperator::Negation => Ok(format!("{}\n    neg rax", inner_factor_code)),
            AstUnaryOperator::BitwiseNot => Ok(format!("{}\n    not rax", inner_factor_code)),
            AstUnaryOperator::LogicalNot => Ok(format!(
                "{}\n    cmp rax,0\n    mov rax,0\n    sete al",
                inner_factor_code
            )),
        }),
        AstExpression::BinaryOperator(operator, left, right) => {
            // The left hand side is emitted first. Like all expressions, the result is stored in rax. Then we need to
            // allocate temp space to hold that lhs result before performing the rhs.
            let left_code = generate_expression_code(global_state, block_state, &left)?;

            let lhs_temp_offset_from_sp = block_state.push_temp();
            let right_code = generate_expression_code(global_state, block_state, &right)?;

            let result = Ok(left_code
                + &generate_binary_operator_code(
                    &operator,
                    global_state,
                    block_state,
                    &right_code,
                    &format!("[rsp+{}]", lhs_temp_offset_from_sp),
                ));
            block_state.pop_temp();

            result
        }
        AstExpression::Assign(name, expr) => {
            generate_expression_code(global_state, block_state, expr).and_then(|expr_code| {
                block_state
                    .get_var_location_str(&name)
                    .ok_or(format!("unknown variable {}", name))
                    .and_then(|location| {
                        Ok(format!(
                            "{}\n    mov {},rax ; {} <- rax",
                            expr_code, &location, &name
                        ))
                    })
            })
        }
        AstExpression::Conditional(condition, positive, negative) => {
            let after_pos_label = global_state.consume_jump_label();
            let mut code = generate_expression_code(global_state, block_state, condition)?;

            // Check if the conditional expression was true or false.
            code += "\n    cmp rax,0";

            // If it was false, jump to the section after the positive portion.
            code += &format!("\n    je _j{}", after_pos_label);

            // Otherwise, execute the positive section.
            code += "\n    ";
            code += &generate_expression_code(global_state, block_state, positive)?;

            let negative_code = generate_expression_code(global_state, block_state, negative)?;
            let after_neg_label = global_state.consume_jump_label();
            // At the end of the positive section, jump over the negative section so that both aren't executed.
            code += &format!("\n    jmp _j{}", after_neg_label);

            // Start of the negative section.
            code += &format!("\n    _j{}:", after_pos_label);
            code += "\n    ";
            code += &negative_code;
            code += &format!("\n    _j{}:", after_neg_label);

            Ok(code)
        }
        AstExpression::FuncCall(name, args) => {
            let mut code = String::new();

            // Allocate temp space for the expression results as they are computed. Store and remember them because
            // more temp space will be allocated below when copying the parameters into the correct locations for
            // passing to the callee. In other words, store and remember them so that we don't interleave pop with
            // more push calls below.
            let mut arg_sp_offsets = vec![];
            for arg_expr in args.iter() {
                code += &generate_expression_code(global_state, block_state, arg_expr)?;
                let arg_temp_offset_from_sp = block_state.push_temp();
                arg_sp_offsets.push(arg_temp_offset_from_sp);
                code += &format!(
                    "\n    mov [rsp+{}],rax ; temp <- rax",
                    arg_temp_offset_from_sp
                );
            }

            // Must reserve space for the first 4 args, whether they are used or not and even though they're being
            // stored in registers rather than on the stack. This code looks weird because it appears to do nothing
            // (pushes and pops without any interleaving access), but it has a side effect of recording the required
            // frame size.
            for _ in 0..4 {
                block_state.push_arg();
            }

            // Push the arguments onto the stack in reverse order, as required by calling convention.
            for _ in args.iter().rev() {
                // Grab the last temp value location from the stored list. The length is then the index of the arg.
                let arg_temp_offset_from_sp = arg_sp_offsets.pop().unwrap();

                let dest = match arg_sp_offsets.len() {
                    0 => String::from("rcx"),
                    1 => String::from("rdx"),
                    2 => String::from("r8"),
                    3 => String::from("r9"),
                    _ => format!("[rsp+{}]", block_state.push_arg()),
                };

                code += &format!("\n    mov r10,[rsp+{}]", arg_temp_offset_from_sp);
                code += &format!(
                    "\n    mov {},r10 ; arg {} <- temp",
                    dest,
                    arg_sp_offsets.len()
                );
            }

            code += &format!("\n    call {}", name);

            // Pop all the temp values, making sure to pop at least the 4 that are always reserved.
            for _ in 0..std::cmp::max(args.len(), 4) {
                block_state.pop_arg();
            }

            // Pop all the temp expression values used before setting up the parameters.
            for _ in args.iter() {
                block_state.pop_temp();
            }

            Ok(code)
        }
    }
}

fn validate_ast(ast_program: &AstProgram) -> Result<(), Vec<String>> {
    let mut errors = vec![];

    for function in &ast_program.functions {
        let mut num_definitions = 0;
        let mut param_count_opt = None;
        for func in &ast_program.functions {
            if func.name == function.name {
                // Check for multiple definitions of the same function.
                if func.is_definition() {
                    num_definitions += 1;

                    if num_definitions > 1 {
                        errors.push(format!(
                            "found {} definitions of function \"{}\"",
                            num_definitions, function.name
                        ));
                        return Err(errors);
                    }
                }

                // Check for declarations of the same function with different parameter counts.
                if let Some(param_count) = param_count_opt {
                    if param_count != func.parameters.len() {
                        errors.push(format!("function {} re-delcaration with wrong parameter count {}. previously defined with parameter count {}", func.name, func.parameters.len(), param_count));
                        return Err(errors);
                    }
                } else {
                    param_count_opt = Some(func.parameters.len());
                }
            }
        }
    }

    let all_expressions = ast_program.collect_all_expressions();
    for expression in &all_expressions {
        if let AstExpression::FuncCall(name, args) = expression {
            let func = ast_program.lookup_function_definition(&name).unwrap();
            if func.parameters.len() != args.len() {
                errors.push(format!(
                    "function {} called with wrong parameter count {}. Should be {}.",
                    &name,
                    args.len(),
                    func.parameters.len()
                ));
            }
        }
    }

    if errors.len() == 0 {
        Ok(())
    } else {
        Err(errors)
    }
}

fn assemble_and_link(
    code: &str,
    output_exe_path: &str,
    should_suppress_output: bool,
    temp_dir: &Path,
) -> Option<i32> {
    let asm_path = temp_dir.join("code.asm");
    let exe_temp_output_path = temp_dir.join("output.exe");
    let pdb_temp_output_path = temp_dir.join("output.pdb");

    std::fs::write(&asm_path, &code);

    let mut command = Command::new("ml64.exe");
    let args = [
        "/Zi",
        "/Feoutput.exe",
        "code.asm",
        "/link",
        "/pdb:output.pdb",
    ];
    println!("ml64.exe {} {}", args[0], args[1]);
    command.args(&args);
    command.current_dir(&temp_dir);

    if should_suppress_output {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }

    let status = command.status().expect("failed to run ml64.exe");

    println!("assembly status: {:?}", status);
    if status.success() {
        std::fs::rename(&exe_temp_output_path, &Path::new(output_exe_path));

        let mut pdb_path = Path::new(output_exe_path).to_path_buf();
        pdb_path.set_extension("pdb");
        std::fs::rename(&pdb_temp_output_path, &pdb_path);
    }

    status.code()
}

// TODO should return line numbers with errors
fn parse_and_validate<'i>(mode: Mode, input: &'i str) -> Result<AstProgram<'i>, Vec<String>> {
    let tokens = lex_all_tokens(&input)?;
    /*
    for token in tokens.iter() {
        println!("{}", token);
    }

    println!();
    */

    if let Mode::LexOnly = mode {
        return Ok(AstProgram { functions: vec![] });
    }

    // TODO all parsing should return a list of errors, not just one. for now, wrap it in a single error
    let ast = parse_program(Tokens(&tokens)).map_err(|e| vec![e])?;
    println!("AST:\n{}\n", ast);

    validate_ast(&ast)?;

    Ok(ast)
}

fn preprocess(
    input: &str,
    should_suppress_output: bool,
    temp_dir: &Path,
) -> Result<String, String> {
    let preprocessed_output_path = temp_dir.join("input.i");

    let temp_input_path = temp_dir.join("input.c");
    std::fs::write(&temp_input_path, &input);

    let mut command = Command::new("cl.exe");
    let args = ["/P", "/Fiinput.i", "input.c"];
    command.args(&args);
    command.current_dir(&temp_dir);

    println!("preprocess command: {:?}", command);

    if should_suppress_output {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }

    let status = command.status().expect("failed to run cl.exe");

    println!("preprocess status: {:?}", status);
    if status.success() {
        Ok(std::fs::read_to_string(&preprocessed_output_path).map_err(format_io_err)?)
    } else {
        Err(format!("preprocessor failed with {:?}", status))
    }
}

fn compile_and_link(
    args: &LcArgs,
    input: &str,
    should_suppress_output: bool,
) -> Result<i32, String> {
    fn helper(
        args: &LcArgs,
        input: &str,
        should_suppress_output: bool,
        temp_dir: &Path,
    ) -> Result<i32, String> {
        let input = preprocess(input, should_suppress_output, temp_dir)?;

        match parse_and_validate(args.mode, &input) {
            Ok(ast) => match args.mode {
                Mode::All | Mode::CodegenOnly => {
                    let asm = generate_program_code(&ast)?;
                    println!("assembly:\n{}", asm);

                    if let Mode::All = args.mode {
                        let exit_code = assemble_and_link(
                            &asm,
                            &args.output_path.as_ref().unwrap(),
                            should_suppress_output,
                            temp_dir,
                        )
                        .expect("programs should always have an exit code");
                        println!("assemble status: {}", exit_code);

                        if exit_code == 0 {
                            Ok(exit_code)
                        } else {
                            Err(format!("assembler failed with exit code {}", exit_code))
                        }
                    } else {
                        if let Some(output_path) = &args.output_path {
                            std::fs::write(&output_path, &asm).map_err(format_io_err)?;
                        }

                        Ok(0)
                    }
                }
                _ => Ok(0),
            },
            Err(errors) => {
                if errors.len() > 1 {
                    for error_message in errors.iter().skip(1) {
                        println!("Error: {}", error_message);
                    }
                }

                // For now return just the first error
                let error_message = errors.get(0).unwrap();
                Err(error_message.clone())
            }
        }
    }

    let temp_dir_name = format!("testrun_{}", generate_random_string(8));
    let temp_dir = Path::new(&temp_dir_name);
    std::fs::create_dir_all(&temp_dir);
    let ret = helper(args, input, should_suppress_output, temp_dir);
    if ret.is_ok() {
        println!("cleaning up temp dir {}", temp_dir.to_string_lossy());
        std::fs::remove_dir_all(&temp_dir);
    }
    ret
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, clap::ValueEnum)]
enum Mode {
    /// Compile and link executable.
    #[value(name = "all")]
    All,

    /// Lex only. Exit code is zero if successful.
    #[value(name = "lex", alias = "l")]
    LexOnly,

    /// Lex and parse only. Exit code is zero if successful.
    #[value(name = "parse", alias = "p")]
    ParseOnly,

    /// Lex, parse, and codegen only. Exit code is zero if successful.
    #[value(name = "codegen", alias = "c")]
    CodegenOnly,
}

#[derive(clap::Parser, Clone)]
#[command(author, version, about)]
struct LcArgs {
    /// Mode
    #[arg(short = 'm', value_enum, default_value_t = Mode::All)]
    mode: Mode,

    #[arg()]
    input_path: String,

    #[arg()]
    output_path: Option<String>,
}

fn main() {
    let mut args = LcArgs::parse();

    if let Mode::All = args.mode {
        if args.output_path.is_none() {
            println!("Must specify output path!");
            std::process::exit(1)
        }
    }

    println!("loading {}", args.input_path);
    let input = std::fs::read_to_string(&args.input_path).unwrap();

    let exit_code = match compile_and_link(&args, &input, false) {
        Ok(inner_exit_code) => inner_exit_code,
        Err(msg) => {
            println!("error! {}", msg);
            1
        }
    };

    std::process::exit(exit_code)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lex_simple() {
        let input = r"int main() {
    return 2;
}";
        assert_eq!(
            lex_all_tokens(&input),
            Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"])
        );
    }

    #[test]
    fn lex_no_whitespace() {
        let input = r"int main(){return 2;}";
        assert_eq!(
            lex_all_tokens(&input),
            Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"])
        );
    }

    #[test]
    fn lex_negative() {
        assert_eq!(
            lex_all_tokens("int main() { return -1; }"),
            Ok(vec![
                "int", "main", "(", ")", "{", "return", "-", "1", ";", "}"
            ])
        );
    }

    #[test]
    fn lex_bitwise_not() {
        assert_eq!(
            lex_all_tokens("int main() { return ~1; }"),
            Ok(vec![
                "int", "main", "(", ")", "{", "return", "~", "1", ";", "}"
            ])
        );
    }

    #[test]
    fn lex_logical_not() {
        assert_eq!(
            lex_all_tokens("int main() { return !1; }"),
            Ok(vec![
                "int", "main", "(", ")", "{", "return", "!", "1", ";", "}"
            ])
        );
    }

    fn codegen_run_and_check_exit_code_or_compile_failure(
        input: &str,
        expected_result: Option<i32>,
    ) {
        let args = LcArgs {
            input_path: String::new(),
            output_path: Some(format!("test_{}.exe", generate_random_string(8))),
            mode: Mode::All,
        };

        let compile_result = compile_and_link(&args, input, true);
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

    fn test_validate_error_count(input: &str, expected_error_count: usize) {
        match parse_and_validate(Mode::All, input) {
            Ok(ast) => {
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
        codegen_run_and_check_exit_code(&format!("int main() {{ {} }}", body), expected_exit_code);
    }

    fn test_codegen_mainfunc_failure(body: &str) {
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
    fn test_codegen_operator_precedence2() {
        test_codegen_expression("1 * 2 + 3 * 4", 14);
    }

    #[test]
    fn test_codegen_var_use() {
        test_codegen_mainfunc(
            "int x = 5; int y = 6; int z; x = 1; z = 3; return x + y + z;",
            10,
        );
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

    #[test]
    fn test_codegen_if_assign() {
        test_codegen_mainfunc("int x = 5; if (x == 5) x = 4; return x;", 4);
    }

    #[test]
    fn test_codegen_if_else_assign() {
        test_codegen_mainfunc(
            "int x = 5; if (x == 5) x = 4; else x == 6; if (x == 6) x = 7; else x = 8; return x;",
            8,
        );
    }

    #[test]
    fn test_codegen_ternary() {
        test_codegen_mainfunc("return 1 == 1 ? 2 : 3;", 2);
        test_codegen_mainfunc("return 1 == 0 ? 2 : 3;", 3);
    }

    #[test]
    fn test_block_var_declarations() {
        test_codegen_mainfunc("int x = 1; { x = 3; } return x;", 3);
        test_codegen_mainfunc("int x = 1; { int x = 3; } return x;", 1);
        test_codegen_mainfunc("int x = 1; { int y; } int z = 7; return z;", 7);
        test_codegen_mainfunc_failure("int x = 1; { int x = 3; } int x = 5; return x;");
    }

    #[test]
    fn test_while_loop() {
        test_codegen_mainfunc("int x = 1; while (x < 10) x = x + 1; return x;", 10);
    }

    #[test]
    fn test_while_loop_with_break() {
        test_codegen_mainfunc(
            "int x = 1; while (x < 10) { x = x + 1; break; } return x;",
            2,
        );
    }

    #[test]
    fn test_while_loop_with_continue() {
        test_codegen_mainfunc(
            "int x = 1; while (x < 10) { x = x + 1; continue; x = 50; } return x;",
            10,
        );
    }

    #[test]
    fn test_do_while_loop() {
        test_codegen_mainfunc("do { return 1; } while (0); return 2;", 1);
    }

    #[test]
    fn test_do_while_loop_with_break() {
        test_codegen_mainfunc("do { break; return 1; } while (0); return 2;", 2);
    }

    #[test]
    fn test_do_while_loop_with_continue() {
        test_codegen_mainfunc(
            "int x = 20; do { continue; } while ((x = 50) < 10); return x;",
            50,
        );
    }

    #[test]
    fn test_for_loop() {
        test_codegen_mainfunc(
            "int y = 100; for (int i = 0; i < 10; i = i + 1) y = i; return y;",
            9,
        );
        test_codegen_mainfunc(
            "int y = 100; int i = 150; for (int i = 0; i < 10; i = i + 1) { y = i; } return y + i;",
            159,
        );
        test_codegen_mainfunc(
            "int i = 150; for (i = 0; i < 10; i = i + 1) { } return i;",
            10,
        );
    }

    #[test]
    fn test_for_loop_with_break() {
        test_codegen_mainfunc(
            "int i = 150; for (i = 2; i < 10; i = i + 1) { break; i = 20; } return i;",
            2,
        );
        test_codegen_mainfunc(
            "int i = 150; for (i = 2; i < 10; i = i + 1) { if (i == 3) { break; } } return i;",
            3,
        );
    }

    #[test]
    fn test_for_loop_with_continue() {
        test_codegen_mainfunc(
            "int i = 150; for (i = 2; i < 10; i = i + 1) { continue; i = 20; } return i;",
            10,
        );
    }

    #[test]
    fn test_wrong_func_arg_count() {
        test_validate_error_count(
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
    fn test_recursive_function() {
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
    fn test_nested_function_arg_counts() {
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
    fn test_nested_block_variable_allocation() {
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
    fn test_parameter_redefinition() {
        codegen_run_and_expect_compile_failure(
            r"int blah(int x)
{
    int x;
    return 5;
}

int main() {
    return 1;
}",
        );

        codegen_run_and_expect_compile_failure(
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
