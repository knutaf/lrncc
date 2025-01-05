#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate regex;

use {
    clap::Parser,
    rand::distributions::Alphanumeric,
    rand::{thread_rng, Rng},
    regex::Regex,
    std::{
        cell::RefCell, collections::HashMap, fmt, fmt::Display, ops::Deref, path::*, process::*,
    },
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

fn push_error(result: Result<(), String>, errors: &mut Vec<String>) {
    if let Err(error) = result {
        errors.push(error);
    }
}

fn fmt_list<'t, T>(
    f: &mut fmt::Formatter,
    list: impl Iterator<Item = &'t T>,
    joiner: &str,
) -> fmt::Result
where
    T: fmt::Display + 't,
{
    let mut first = true;
    for item in list {
        if !first {
            f.write_str(joiner)?;
        }

        first = false;

        item.fmt(f)?;
    }

    Ok(())
}

/// A wrapper to allow passing an arbitrary formatter for Display, so you don't have to implement Display for every type
/// you want to print.
struct DisplayWrapper<W>
where
    W: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    writer: W,
}

impl<W> fmt::Display for DisplayWrapper<W>
where
    W: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (self.writer)(f)
    }
}

/// Takes a function that is used for fmt-style formatting.
fn display_with<W>(writer: W) -> DisplayWrapper<W>
where
    W: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    DisplayWrapper { writer }
}

fn format_code_and_comment<W1, W2>(
    f: &mut fmt::Formatter,
    code_writer: W1,
    comment_writer: W2,
) -> fmt::Result
where
    W1: Fn(&mut fmt::Formatter) -> fmt::Result,
    W2: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    // This has to be done in two stages because it's not really possible to pass in the padding/alignment information
    // to the formatter. So format them into individual strings and then write them as just strings with the correct
    // alignment.
    let code_output = format!("{}", display_with(code_writer));
    let comment_output = format!("{}", display_with(comment_writer));

    write!(f, "    {:<30} ; {:<50}\n", code_output, comment_output)
}

trait FmtNode {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result;

    fn fmt_nodelist<'t, T>(
        f: &mut fmt::Formatter,
        list: impl Iterator<Item = &'t T>,
        joiner: &str,
        indent_levels: u32,
    ) -> fmt::Result
    where
        T: FmtNode + 't,
    {
        let mut first = true;
        for item in list {
            if !first {
                f.write_str(joiner)?;
            }

            first = false;

            item.fmt_node(f, indent_levels)?;
        }

        Ok(())
    }

    fn write_indent(f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        for _ in 0..indent_levels {
            f.write_str("    ")?;
        }

        Ok(())
    }
}

// A wrapper around a slice of tokens with convenience functions useful for parsing.
#[derive(PartialEq, Clone, Debug)]
struct Tokens<'i, 't>(&'t [&'i str]);

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
                    Regex::new(r"^[a-zA-Z_]\w*$").expect("failed to compile regex");
                static ref KEYWORDS_REGEX: Regex =
                    Regex::new(r"^(?:int|void|return|if|goto|for|break|continue)$")
                        .expect("failed to compile regex");
            }

            IDENT_REGEX.is_match(token) && !KEYWORDS_REGEX.is_match(token)
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
}

impl<'i, 't> Deref for Tokens<'i, 't> {
    type Target = &'t [&'i str];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
struct AstProgram<'i> {
    functions: Vec<AstFunction<'i>>,
    global_tracking_opt: Option<GlobalTracking>,
}

#[derive(Debug, Clone)]
struct AstBlock(Vec<AstBlockItem>);

#[derive(Debug)]
struct AstFunction<'i> {
    name: &'i str,
    parameters: Vec<String>,
    body: AstBlock,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AstIdentifier(String);

#[derive(Debug, Clone)]
enum AstBlockItem {
    Statement(AstStatement),
    Declaration(AstDeclaration),
}

#[derive(Clone, Debug)]
struct AstDeclaration {
    identifier: AstIdentifier,
    initializer_opt: Option<AstExpression>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AstLabel(String);

#[derive(Debug, Clone)]
struct AstStatement {
    labels: Vec<AstLabel>,
    typ: AstStatementType,
}

// TODO: use a string slice instead of a string
#[derive(Clone, Debug)]
enum AstStatementType {
    Return(AstExpression),
    Expr(AstExpression),
    If(AstExpression, Box<AstStatement>, Option<Box<AstStatement>>),
    Goto(AstLabel),
    Compound(Box<AstBlock>),
    For(
        AstForInit,
        Option<AstExpression>,
        Option<AstExpression>,
        Box<AstStatement>,
        Option<AstLabel>,
        Option<AstLabel>,
    ),
    Break(Option<AstLabel>),
    Continue(Option<AstLabel>),
    Null,
}

#[derive(Clone, Debug)]
enum AstForInit {
    Declaration(AstDeclaration),
    Expression(Option<AstExpression>),
}

#[derive(Clone, Debug)]
enum AstExpression {
    Constant(u32),
    UnaryOperator(AstUnaryOperator, Box<AstExpression>),
    BinaryOperator(Box<AstExpression>, AstBinaryOperator, Box<AstExpression>),
    Var(AstIdentifier),
    Assignment(Box<AstExpression>, AstBinaryOperator, Box<AstExpression>),
    Conditional(Box<AstExpression>, Box<AstExpression>, Box<AstExpression>),
}

#[derive(PartialEq, Clone, Debug)]
enum AstUnaryOperator {
    Negation,
    BitwiseNot,
    Not,
    PrefixIncrement,
    PrefixDecrement,
    PostfixIncrement,
    PostfixDecrement,
}

#[derive(PartialEq, Clone, Debug)]
enum AstBinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    Assign,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    ModulusAssign,
    BitwiseAndAssign,
    BitwiseOrAssign,
    BitwiseXorAssign,
    ShiftLeftAssign,
    ShiftRightAssign,
    Conditional,
}

#[derive(Debug)]
struct TacProgram {
    functions: Vec<TacFunction>,
}

// TODO: string slice instead
#[derive(Debug)]
struct TacFunction {
    name: String,
    body: Vec<TacInstruction>,
}

#[derive(Debug, Clone)]
struct TacLabel(String);

#[derive(Debug)]
enum TacInstruction {
    Return(TacVal),
    UnaryOp(TacUnaryOperator, TacVal, TacVar),
    BinaryOp(TacVal, TacBinaryOperator, TacVal, TacVar),
    CopyVal(TacVal, TacVar),
    Jump(TacLabel),
    JumpIfZero(TacVal, TacLabel),
    JumpIfNotZero(TacVal, TacLabel),
    Label(TacLabel),
}

#[derive(Debug, Clone)]
struct TacVar(String);

#[derive(Debug, Clone)]
enum TacVal {
    Constant(u32),
    Var(TacVar),
}

#[derive(Debug)]
enum TacUnaryOperator {
    Negation,
    BitwiseNot,
    Not,
}

#[derive(Debug)]
enum TacBinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}

#[derive(Debug)]
struct AsmProgram {
    functions: Vec<AsmFunction>,
}

// TODO: use str slice
#[derive(Debug)]
struct AsmFunction {
    name: String,
    body: Vec<AsmInstruction>,
}

#[derive(Debug, Clone)]
struct AsmLabel(String);

#[derive(Debug, Clone)]
enum AsmInstruction {
    Mov(AsmVal, AsmLocation),
    UnaryOp(AsmUnaryOperator, AsmLocation),
    BinaryOp(AsmBinaryOperator, AsmVal, AsmLocation),
    Idiv(AsmVal),
    Cdq,
    AllocateStack(u32),
    Ret(u32),
    Cmp(AsmVal, AsmVal),
    SetCc(AsmCondCode, AsmLocation),
    Jmp(AsmLabel),
    JmpCc(AsmCondCode, AsmLabel),
    Label(AsmLabel),
}

#[derive(Debug, Clone)]
enum AsmVal {
    Imm(u32),
    Loc(AsmLocation),
}

#[derive(Debug, Clone)]
enum AsmLocation {
    Reg(&'static str),
    PseudoReg(String),
    RbpOffset(i32, String),
    RspOffset(u32, String),
}

#[derive(Debug, Clone)]
enum AsmUnaryOperator {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
enum AsmBinaryOperator {
    Add,
    Subtract,
    Imul,
    And,
    Or,
    Xor,
    Shl,
    Sar,
}

#[derive(Debug, Clone)]
enum AsmCondCode {
    E,
    NE,
    L,
    LE,
    G,
    GE,
}

#[derive(Debug)]
struct GlobalTracking {
    next_temporary_id: u32,
    next_label_id: u32,
}

#[derive(Debug)]
struct FunctionTracking {
    labels: HashMap<AstLabel, AstLabel>,
}

#[derive(Debug)]
struct BlockTracking<'p> {
    parent_opt: Option<&'p BlockTracking<'p>>,
    variables: HashMap<AstIdentifier, AstIdentifier>,
}

#[derive(Debug)]
struct BreakTracking {
    parent_opt: Option<Box<BreakTracking>>,
    break_label: AstLabel,
    continue_label_opt: Option<AstLabel>,
}

#[derive(Debug)]
struct FuncStackFrame {
    names: HashMap<String, i32>,
    max_base_offset: u32,
}

impl<'i> AstProgram<'i> {
    fn new(functions: Vec<AstFunction<'i>>) -> Self {
        Self {
            functions,
            global_tracking_opt: None,
        }
    }

    fn lookup_function_definition(&'i self, name: &str) -> Option<&'i AstFunction<'i>> {
        self.functions.iter().find(|func| func.name == name)
    }

    fn validate_and_resolve(&mut self) -> Result<(), Vec<String>> {
        self.global_tracking_opt = Some(GlobalTracking::new());

        let mut errors = vec![];
        for func in self.functions.iter_mut() {
            let mut function_tracking = FunctionTracking::new();
            let global_tracking = self.global_tracking_opt.as_mut().unwrap();

            func.validate_and_resolve_variables(global_tracking, &mut errors);

            push_error(
                func.for_each_statement(|ast_statement| {
                    for label in ast_statement.labels.iter_mut() {
                        match function_tracking.add_goto_label(global_tracking, label) {
                            Ok(new_label) => {
                                *label = new_label;
                            }
                            Err(err) => {
                                errors.push(err);
                            }
                        }
                    }

                    Ok(())
                }),
                &mut errors,
            );

            push_error(
                func.for_each_statement(|ast_statement| {
                    if let AstStatementType::Goto(ref mut label) = ast_statement.typ {
                        match function_tracking.resolve_label(label) {
                            Ok(new_label) => {
                                *label = new_label;
                            }
                            Err(err) => {
                                errors.push(err);
                            }
                        }
                    }

                    Ok(())
                }),
                &mut errors,
            );

            // Loop labeling: each loop has two labels associated with it: one that a break statement should jump to
            // (after the end of the loop) and one that a continue statement should jump to (after the end of the body).
            // Because of nesting, the current tracking data needs to store a reference to the containing tracking data,
            // so when the loop is over, we can revert back to the parent.
            //
            // Because we are referencing break_tracking_opt in two closures, the borrow checker doesn't know that we
            // are only referencing it one at a time, so store in a RefCell, to allow for borrowing it in both closures
            // when needed.
            let mut break_tracking_opt = RefCell::new(None);
            push_error(
                func.for_each_statement_and_after(
                    |ast_statement| {
                        match ast_statement.typ {
                            AstStatementType::For(
                                ref _ast_for_init,
                                ref _ast_expr_condition_opt,
                                ref _ast_expr_final_opt,
                                ref _ast_statement_body,
                                ref mut ast_body_end_label_opt,
                                ref mut ast_loop_end_label_opt,
                            ) => {
                                assert!(ast_body_end_label_opt.is_none());
                                assert!(ast_loop_end_label_opt.is_none());

                                // Create new labels for this loop's body end and loop end, to be used in break and continue
                                // statements.
                                *ast_body_end_label_opt =
                                    Some(AstLabel(global_tracking.allocate_label("for_body_end")));
                                *ast_loop_end_label_opt =
                                    Some(AstLabel(global_tracking.allocate_label("for_loop_end")));

                                // Store the pre-existing tracking info into the new tracking info, so we can revert to it
                                // after this loop is done.
                                let parent_opt = break_tracking_opt.borrow_mut().take();
                                *break_tracking_opt.borrow_mut() =
                                    Some(Box::new(BreakTracking::new(
                                        parent_opt,
                                        ast_loop_end_label_opt.as_ref().unwrap().clone(),
                                        ast_body_end_label_opt.clone(),
                                    )));
                            }
                            AstStatementType::Break(ref mut label_opt) => {
                                assert!(label_opt.is_none());

                                let Some(ref break_tracking) = *break_tracking_opt.borrow() else {
                                    errors.push(format!(
                                        "break statement with no containing loop or switch"
                                    ));
                                    return Ok(());
                                };

                                // Write in the correct label to use for breaking in this scope.
                                *label_opt = Some(break_tracking.break_label.clone());
                            }
                            AstStatementType::Continue(ref mut label_opt) => {
                                assert!(label_opt.is_none());

                                // This could happen if the continue statement is found outside of a loop.
                                let Some(ref break_tracking) = *break_tracking_opt.borrow() else {
                                    errors.push(format!(
                                        "continue statement with no containing loop"
                                    ));
                                    return Ok(());
                                };

                                // This could happen if the continue statement is found outside of a loop or anything else
                                // that uses BreakTracking, such as a switch statement.
                                let Some(ref continue_label) = break_tracking.continue_label_opt
                                else {
                                    errors.push(format!(
                                        "continue statement with no containing loop"
                                    ));
                                    return Ok(());
                                };

                                // Write in the correct label to use for continuing in this scope.
                                *label_opt = Some(continue_label.clone());
                            }
                            _ => {}
                        }

                        Ok(())
                    },
                    |ast_statement| {
                        if let AstStatementType::For(
                            _ast_for_init,
                            _ast_expr_condition_opt,
                            _ast_expr_final_opt,
                            _ast_statement_body,
                            _ast_body_end_label_opt,
                            _ast_loop_end_label_opt,
                        ) = &ast_statement.typ
                        {
                            assert!(break_tracking_opt.borrow().is_some());

                            // This for loop is done, so revert the break tracking to its containing one.
                            let parent_opt =
                                break_tracking_opt.borrow_mut().take().unwrap().parent_opt;
                            *break_tracking_opt.borrow_mut() = parent_opt;
                        }

                        Ok(())
                    },
                ),
                &mut errors,
            );
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn to_tac(&mut self) -> Result<TacProgram, String> {
        let mut functions = vec![];
        for func in self.functions.iter() {
            functions.push(func.to_tac(self.global_tracking_opt.as_mut().unwrap())?);
        }

        Ok(TacProgram { functions })
    }
}

impl<'i> FmtNode for AstProgram<'i> {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        Self::fmt_nodelist(f, self.functions.iter(), "\n\n", 0)
    }
}

impl AstBlock {
    fn validate_and_resolve_variables(
        &mut self,
        global_tracking: &mut GlobalTracking,
        block_tracking: &mut BlockTracking,
        errors: &mut Vec<String>,
    ) {
        for block_item in self.0.iter_mut() {
            block_item.validate_and_resolve_variables(global_tracking, block_tracking, errors);
        }
    }

    fn for_each_statement_and_after<F1, F2>(
        &mut self,
        func: &mut F1,
        func_after: &mut F2,
    ) -> Result<(), String>
    where
        F1: FnMut(&mut AstStatement) -> Result<(), String>,
        F2: FnMut(&mut AstStatement) -> Result<(), String>,
    {
        for block_item in self.0.iter_mut() {
            if let AstBlockItem::Statement(ref mut statement) = block_item {
                statement.for_each_statement_and_after(func, func_after)?;
            }
        }

        Ok(())
    }

    fn to_tac(
        &self,
        global_tracking: &mut GlobalTracking,
        instructions: &mut Vec<TacInstruction>,
    ) -> Result<(), String> {
        for block_item in self.0.iter() {
            block_item.to_tac(global_tracking, instructions)?;
        }

        Ok(())
    }
}

impl<'i> AstFunction<'i> {
    fn validate_and_resolve_variables(
        &mut self,
        global_tracking: &mut GlobalTracking,
        errors: &mut Vec<String>,
    ) {
        let mut block_tracking = BlockTracking::new(None);
        self.body
            .validate_and_resolve_variables(global_tracking, &mut block_tracking, errors);
    }

    fn for_each_statement_and_after<F1, F2>(
        &mut self,
        mut func: F1,
        mut func_after: F2,
    ) -> Result<(), String>
    where
        F1: FnMut(&mut AstStatement) -> Result<(), String>,
        F2: FnMut(&mut AstStatement) -> Result<(), String>,
    {
        self.body
            .for_each_statement_and_after(&mut func, &mut func_after)
    }

    fn for_each_statement<F>(&mut self, mut func: F) -> Result<(), String>
    where
        F: FnMut(&mut AstStatement) -> Result<(), String>,
    {
        fn null_func(_statement: &mut AstStatement) -> Result<(), String> {
            Ok(())
        }

        self.body
            .for_each_statement_and_after(&mut func, &mut null_func)
    }

    fn to_tac(&self, global_tracking: &mut GlobalTracking) -> Result<TacFunction, String> {
        let mut body_instructions = vec![];
        self.body.to_tac(global_tracking, &mut body_instructions)?;

        // In C, functions lacking a return value either automatically return 0 (in the case of main), have undefined
        // behavior (if the function's return value is actually used), or the return value doesn't matter (if the return
        // value isn't examined). To handle all three cases, just add an extra "return 0" to the end of every function
        // body.
        body_instructions.push(TacInstruction::Return(TacVal::Constant(0)));

        Ok(TacFunction {
            name: String::from(self.name),
            body: body_instructions,
        })
    }
}

impl<'i> FmtNode for AstFunction<'i> {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;
        write!(f, "FUNC {}(", self.name)?;
        fmt_list(f, self.parameters.iter(), ", ")?;
        writeln!(f, "):")?;
        for block_item in self.body.0.iter() {
            block_item.fmt_node(f, indent_levels + 1)?;
            writeln!(f)?;
        }

        Ok(())
    }
}

impl AstIdentifier {
    fn to_tac(&self) -> TacVar {
        TacVar(self.0.clone())
    }
}

impl AstBlockItem {
    fn validate_and_resolve_variables(
        &mut self,
        global_tracking: &mut GlobalTracking,
        block_tracking: &mut BlockTracking,
        errors: &mut Vec<String>,
    ) {
        match self {
            AstBlockItem::Statement(statement) => {
                statement.validate_and_resolve_variables(global_tracking, block_tracking, errors);
            }
            AstBlockItem::Declaration(declaration) => {
                declaration.validate_and_resolve_variables(global_tracking, block_tracking, errors);
            }
        }
    }

    fn to_tac(
        &self,
        global_tracking: &mut GlobalTracking,
        instructions: &mut Vec<TacInstruction>,
    ) -> Result<(), String> {
        match self {
            AstBlockItem::Statement(statement) => statement.to_tac(global_tracking, instructions),
            AstBlockItem::Declaration(declaration) => {
                declaration.to_tac(global_tracking, instructions)
            }
        }
    }
}

impl FmtNode for AstBlockItem {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        match self {
            AstBlockItem::Statement(statement) => statement.fmt_node(f, indent_levels),
            AstBlockItem::Declaration(declaration) => declaration.fmt_node(f, indent_levels),
        }
    }
}

impl AstDeclaration {
    fn validate_and_resolve_variables(
        &mut self,
        global_tracking: &mut GlobalTracking,
        block_tracking: &mut BlockTracking,
        errors: &mut Vec<String>,
    ) {
        match block_tracking.add_variable(global_tracking, &self.identifier) {
            Ok(identifier) => {
                self.identifier = identifier;
            }
            Err(error) => {
                errors.push(error);
                return;
            }
        }

        if let Some(initializer) = &mut self.initializer_opt {
            push_error(
                initializer.validate_and_resolve_variables(block_tracking),
                errors,
            );
        }
    }

    fn to_tac(
        &self,
        global_tracking: &mut GlobalTracking,
        instructions: &mut Vec<TacInstruction>,
    ) -> Result<(), String> {
        if let Some(initializer) = &self.initializer_opt {
            let tac_val = initializer.to_tac(global_tracking, instructions)?;
            instructions.push(TacInstruction::CopyVal(tac_val, self.identifier.to_tac()));
        }

        Ok(())
    }
}

impl FmtNode for AstDeclaration {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;

        write!(f, "int {}", self.identifier.0)?;

        if let Some(initializer) = &self.initializer_opt {
            write!(f, " = ")?;
            initializer.fmt_node(f, indent_levels + 1)?;
        }

        Ok(())
    }
}

impl AstLabel {
    fn to_tac(&self) -> TacLabel {
        TacLabel(self.0.clone())
    }
}

impl AstStatement {
    fn new(typ: AstStatementType, labels: Vec<AstLabel>) -> Self {
        Self { typ, labels }
    }

    fn validate_and_resolve_variables(
        &mut self,
        global_tracking: &mut GlobalTracking,
        block_tracking: &mut BlockTracking,
        errors: &mut Vec<String>,
    ) {
        match &mut self.typ {
            AstStatementType::Return(expr) | AstStatementType::Expr(expr) => {
                push_error(expr.validate_and_resolve_variables(block_tracking), errors);
            }
            AstStatementType::If(condition_expr, then_statement, else_statement_opt) => {
                push_error(
                    condition_expr.validate_and_resolve_variables(block_tracking),
                    errors,
                );
                then_statement.validate_and_resolve_variables(
                    global_tracking,
                    block_tracking,
                    errors,
                );

                if let Some(else_statement) = else_statement_opt {
                    else_statement.validate_and_resolve_variables(
                        global_tracking,
                        block_tracking,
                        errors,
                    );
                }
            }
            AstStatementType::Goto(_label) => {}
            AstStatementType::Compound(block) => {
                // Create a new scope contained within this one, so that variables can be declared within that shadow
                // the outside ones.
                let mut inner_block = BlockTracking::new(Some(&block_tracking));

                block.validate_and_resolve_variables(global_tracking, &mut inner_block, errors);
            }
            AstStatementType::For(
                ast_for_init,
                ast_expr_condition_opt,
                ast_expr_final_opt,
                ast_statement_body,
                _ast_body_end_label_opt,
                _ast_loop_end_label_opt,
            ) => {
                // The for loop initializer introduces a new scope for its declarations, if any.
                let mut inner_block = BlockTracking::new(Some(&block_tracking));
                ast_for_init.validate_and_resolve_variables(
                    global_tracking,
                    &mut inner_block,
                    errors,
                );

                if let Some(expr) = ast_expr_condition_opt {
                    push_error(
                        expr.validate_and_resolve_variables(&mut inner_block),
                        errors,
                    );
                }

                if let Some(expr) = ast_expr_final_opt {
                    push_error(
                        expr.validate_and_resolve_variables(&mut inner_block),
                        errors,
                    );
                }

                ast_statement_body.validate_and_resolve_variables(
                    global_tracking,
                    &mut inner_block,
                    errors,
                );
            }
            AstStatementType::Break(_label) => {}
            AstStatementType::Continue(_label) => {}
            AstStatementType::Null => {}
        }
    }

    fn for_each_statement_and_after<F1, F2>(
        &mut self,
        func: &mut F1,
        func_after: &mut F2,
    ) -> Result<(), String>
    where
        F1: FnMut(&mut AstStatement) -> Result<(), String>,
        F2: FnMut(&mut AstStatement) -> Result<(), String>,
    {
        (func)(self)?;

        match &mut self.typ {
            AstStatementType::Return(_expr) => {}
            AstStatementType::If(_expr, ref mut then_statement, ref mut else_statement_opt) => {
                then_statement.for_each_statement_and_after(func, func_after)?;

                if let Some(else_statement) = else_statement_opt {
                    else_statement.for_each_statement_and_after(func, func_after)?;
                }
            }
            AstStatementType::Expr(_expr) => {}
            AstStatementType::Goto(_label) => {}
            AstStatementType::Compound(block) => {
                block.for_each_statement_and_after(func, func_after)?;
            }
            AstStatementType::For(
                _ast_for_init,
                _ast_expr_condition_opt,
                _ast_expr_final_opt,
                ast_statement_body,
                _ast_body_end_label_opt,
                _ast_loop_end_label_opt,
            ) => {
                ast_statement_body.for_each_statement_and_after(func, func_after)?;
            }
            AstStatementType::Break(_label) => {}
            AstStatementType::Continue(_label) => {}
            AstStatementType::Null => {}
        }

        (func_after)(self)
    }

    fn to_tac(
        &self,
        global_tracking: &mut GlobalTracking,
        instructions: &mut Vec<TacInstruction>,
    ) -> Result<(), String> {
        // A statement can have one or more labels. Emit all of the labels applied to this statement first.
        for label in self.labels.iter() {
            instructions.push(TacInstruction::Label(label.to_tac()));
        }

        match &self.typ {
            AstStatementType::Return(ast_exp) => {
                let tac_val = ast_exp.to_tac(global_tracking, instructions)?;
                instructions.push(TacInstruction::Return(tac_val));
            }
            AstStatementType::Expr(ast_exp) => {
                let _tac_val = ast_exp.to_tac(global_tracking, instructions)?;
            }
            AstStatementType::If(condition_expr, then_statement, else_statement_opt) => {
                let end_label = TacLabel(global_tracking.allocate_label("if_end"));

                let else_begin_label_opt = if else_statement_opt.is_some() {
                    Some(TacLabel(global_tracking.allocate_label("else_begin")))
                } else {
                    None
                };

                let tac_condition_val = condition_expr.to_tac(global_tracking, instructions)?;

                // If the condition is false, jump either to the beginning of the else, if present, or just the end of
                // the if statement.
                instructions.push(TacInstruction::JumpIfZero(
                    tac_condition_val,
                    else_begin_label_opt.as_ref().unwrap_or(&end_label).clone(),
                ));

                then_statement.to_tac(global_tracking, instructions)?;

                if let Some(else_statement) = else_statement_opt {
                    // After the then-clause is done, if there's an else-clause present, jump over it to the end.
                    instructions.push(TacInstruction::Jump(end_label.clone()));

                    // And then emit the label for the start of the else clause.
                    instructions.push(TacInstruction::Label(else_begin_label_opt.unwrap()));

                    // And then the else clause itself.
                    else_statement.to_tac(global_tracking, instructions)?;
                }

                instructions.push(TacInstruction::Label(end_label));
            }
            AstStatementType::Goto(label) => {
                instructions.push(TacInstruction::Jump(label.to_tac()));
            }
            AstStatementType::Compound(block) => {
                block.to_tac(global_tracking, instructions)?;
            }
            AstStatementType::For(
                ast_for_init,
                ast_expr_condition_opt,
                ast_expr_final_opt,
                ast_statement_body,
                ast_body_end_label_opt,
                ast_loop_end_label_opt,
            ) => {
                // The for loop initializer runs one time, first.
                ast_for_init.to_tac(global_tracking, instructions)?;

                // The for loop condition is evaluated next, and it's the top of the repeated portion, so we need a
                // label to jump to for the beginning of each iteration.
                let condition_label = TacLabel(global_tracking.allocate_label("for_condition"));
                instructions.push(TacInstruction::Label(condition_label.clone()));

                // The for loop condition is actually optional. If it's not present, it'll just fall through to the loop
                // body. If it is present, it's evaluated, and if the condition fails, it jumps to the end of the loop.
                if let Some(ast_condition) = ast_expr_condition_opt {
                    let tac_condition_val = ast_condition.to_tac(global_tracking, instructions)?;

                    instructions.push(TacInstruction::JumpIfZero(
                        tac_condition_val,
                        ast_loop_end_label_opt.as_ref().unwrap().to_tac(),
                    ));
                }

                ast_statement_body.to_tac(global_tracking, instructions)?;

                // After the loop body we need a label that is used for continue statements.
                instructions.push(TacInstruction::Label(
                    ast_body_end_label_opt.as_ref().unwrap().to_tac(),
                ));

                // If the final expression is present, it is evaluated after the end of the loop body.
                if let Some(ast_final_expr) = ast_expr_final_opt {
                    let _ = ast_final_expr.to_tac(global_tracking, instructions)?;
                }

                // It's a loop, so of course we have to jump back to the top after the end of the final expressoin.
                instructions.push(TacInstruction::Jump(condition_label));

                // The loop end label is used for two purposes: jump to it if the loop condition fails; and jump to it
                // from break statements.
                instructions.push(TacInstruction::Label(
                    ast_loop_end_label_opt.as_ref().unwrap().to_tac(),
                ));
            }
            AstStatementType::Break(label_opt) | AstStatementType::Continue(label_opt) => {
                instructions.push(TacInstruction::Jump(label_opt.as_ref().unwrap().to_tac()));
            }
            AstStatementType::Null => {}
        }

        Ok(())
    }
}

impl FmtNode for AstStatement {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;

        for label in self.labels.iter() {
            writeln!(f, "{}:", label.0)?;
            Self::write_indent(f, indent_levels)?;
        }

        match &self.typ {
            AstStatementType::Return(expr) => {
                write!(f, "return ")?;
                expr.fmt_node(f, indent_levels + 1)?;
            }
            AstStatementType::Expr(expr) => {
                expr.fmt_node(f, indent_levels + 1)?;
            }
            AstStatementType::If(condition_expr, then_statement, else_statement_opt) => {
                write!(f, "if (")?;
                condition_expr.fmt_node(f, indent_levels)?;
                writeln!(f, ")")?;
                then_statement.fmt_node(f, indent_levels + 1)?;
                writeln!(f)?;
                Self::write_indent(f, indent_levels)?;

                if let Some(else_statement) = else_statement_opt {
                    writeln!(f, " else {{")?;
                    else_statement.fmt_node(f, indent_levels + 1)?;
                    writeln!(f)?;
                    Self::write_indent(f, indent_levels)?;
                    write!(f, "}}")?;
                }
            }
            AstStatementType::Goto(label) => {
                write!(f, "goto {}", label.0)?;
            }
            AstStatementType::Compound(block) => {
                writeln!(f, "{{")?;

                for block_item in block.0.iter() {
                    block_item.fmt_node(f, indent_levels + 1)?;
                    writeln!(f)?;
                }

                Self::write_indent(f, indent_levels)?;
                write!(f, "}}")?;
            }
            AstStatementType::For(
                ast_for_init,
                ast_expr_condition_opt,
                ast_expr_final_opt,
                ast_statement_body,
                ast_body_end_label_opt,
                ast_loop_end_label_opt,
            ) => {
                write!(f, "for (")?;
                ast_for_init.fmt_node(f, 0)?;
                write!(f, "; ")?;

                if let Some(condition_expr) = ast_expr_condition_opt {
                    condition_expr.fmt_node(f, 0)?;
                }

                // Print out the end label for the body if it's been populated yet. It occurs just before the condition.
                // Helps with seeing where continue statements will jump to.
                if let Some(ast_body_end_label) = ast_body_end_label_opt {
                    write!(f, "; {}: ", ast_body_end_label.0)?;
                } else {
                    write!(f, ";")?;
                }

                if let Some(final_expr) = ast_expr_final_opt {
                    final_expr.fmt_node(f, 0)?;
                }

                writeln!(f, ")")?;
                ast_statement_body.fmt_node(f, indent_levels + 1)?;
                writeln!(f)?;
                Self::write_indent(f, indent_levels)?;

                // Print out the loop end label if it's been populated yet. Helps with seeing where break statements
                // will jump to.
                if let Some(ast_loop_end_label) = ast_loop_end_label_opt {
                    writeln!(f, "{}:", ast_loop_end_label.0)?;
                    Self::write_indent(f, indent_levels)?;
                }
            }
            AstStatementType::Break(None) => {
                write!(f, "break")?;
            }
            AstStatementType::Continue(None) => {
                write!(f, "continue")?;
            }
            AstStatementType::Break(Some(ref label)) => {
                write!(f, "break to {}", label.0)?;
            }
            AstStatementType::Continue(Some(ref label)) => {
                write!(f, "continue to {}", label.0)?;
            }
            AstStatementType::Null => {}
        }

        Ok(())
    }
}

impl AstForInit {
    fn validate_and_resolve_variables(
        &mut self,
        global_tracking: &mut GlobalTracking,
        block_tracking: &mut BlockTracking,
        errors: &mut Vec<String>,
    ) {
        match self {
            AstForInit::Declaration(decl) => {
                decl.validate_and_resolve_variables(global_tracking, block_tracking, errors);
            }
            AstForInit::Expression(Some(expr)) => {
                push_error(expr.validate_and_resolve_variables(block_tracking), errors);
            }
            AstForInit::Expression(None) => {}
        }
    }

    fn to_tac(
        &self,
        global_tracking: &mut GlobalTracking,
        instructions: &mut Vec<TacInstruction>,
    ) -> Result<(), String> {
        match self {
            AstForInit::Declaration(decl) => decl.to_tac(global_tracking, instructions),
            AstForInit::Expression(Some(expr)) => {
                expr.to_tac(global_tracking, instructions)?;
                Ok(())
            }
            AstForInit::Expression(None) => Ok(()),
        }
    }
}

impl FmtNode for AstForInit {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        match self {
            AstForInit::Declaration(decl) => decl.fmt_node(f, indent_levels),
            AstForInit::Expression(expr_opt) => {
                if let Some(expr) = expr_opt {
                    expr.fmt_node(f, indent_levels)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl AstUnaryOperator {
    fn is_postfix(&self) -> bool {
        match self {
            AstUnaryOperator::Negation | AstUnaryOperator::BitwiseNot | AstUnaryOperator::Not => {
                panic!("shouldn't be called except for prefix/postfix")
            }
            AstUnaryOperator::PrefixIncrement | AstUnaryOperator::PrefixDecrement => false,
            AstUnaryOperator::PostfixIncrement | AstUnaryOperator::PostfixDecrement => true,
        }
    }

    fn get_base_binary_op_from_affix_op(&self) -> AstBinaryOperator {
        match self {
            AstUnaryOperator::Negation | AstUnaryOperator::BitwiseNot | AstUnaryOperator::Not => {
                panic!("shouldn't be called except for prefix/postfix")
            }
            AstUnaryOperator::PrefixIncrement | AstUnaryOperator::PostfixIncrement => {
                AstBinaryOperator::Add
            }
            AstUnaryOperator::PrefixDecrement | AstUnaryOperator::PostfixDecrement => {
                AstBinaryOperator::Subtract
            }
        }
    }

    fn to_tac(&self) -> TacUnaryOperator {
        match self {
            AstUnaryOperator::Negation => TacUnaryOperator::Negation,
            AstUnaryOperator::BitwiseNot => TacUnaryOperator::BitwiseNot,
            AstUnaryOperator::Not => TacUnaryOperator::Not,
            AstUnaryOperator::PrefixIncrement
            | AstUnaryOperator::PrefixDecrement
            | AstUnaryOperator::PostfixIncrement
            | AstUnaryOperator::PostfixDecrement => panic!("prefix/postfix handled elsewhere"),
        }
    }
}

impl FmtNode for AstUnaryOperator {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        f.write_str(match self {
            AstUnaryOperator::Negation => "-",
            AstUnaryOperator::BitwiseNot => "~",
            AstUnaryOperator::Not => "!",
            AstUnaryOperator::PrefixIncrement => "++ (pre)",
            AstUnaryOperator::PrefixDecrement => "-- (pre)",
            AstUnaryOperator::PostfixIncrement => "++ (post)",
            AstUnaryOperator::PostfixDecrement => "-- (post)",
        })
    }
}

impl std::str::FromStr for AstUnaryOperator {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "-" => Ok(AstUnaryOperator::Negation),
            "~" => Ok(AstUnaryOperator::BitwiseNot),
            "!" => Ok(AstUnaryOperator::Not),
            "++" => Ok(AstUnaryOperator::PrefixIncrement),
            "--" => Ok(AstUnaryOperator::PrefixDecrement),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl AstBinaryOperator {
    // This specifies the precedence of operators when parentheses are not used. For example, 5 * 3 + 2 * 2 is 19, not
    // 50.
    fn precedence(&self) -> u8 {
        match self {
            AstBinaryOperator::Assign => 0,
            AstBinaryOperator::AddAssign => 0,
            AstBinaryOperator::SubtractAssign => 0,
            AstBinaryOperator::MultiplyAssign => 0,
            AstBinaryOperator::DivideAssign => 0,
            AstBinaryOperator::ModulusAssign => 0,
            AstBinaryOperator::BitwiseAndAssign => 0,
            AstBinaryOperator::BitwiseOrAssign => 0,
            AstBinaryOperator::BitwiseXorAssign => 0,
            AstBinaryOperator::ShiftLeftAssign => 0,
            AstBinaryOperator::ShiftRightAssign => 0,
            AstBinaryOperator::Conditional => 1,
            AstBinaryOperator::Or => 2,
            AstBinaryOperator::And => 3,
            AstBinaryOperator::BitwiseOr => 4,
            AstBinaryOperator::BitwiseXor => 5,
            AstBinaryOperator::BitwiseAnd => 6,
            AstBinaryOperator::Equal => 7,
            AstBinaryOperator::NotEqual => 7,
            AstBinaryOperator::LessThan => 8,
            AstBinaryOperator::LessOrEqual => 8,
            AstBinaryOperator::GreaterThan => 8,
            AstBinaryOperator::GreaterOrEqual => 8,
            AstBinaryOperator::ShiftLeft => 9,
            AstBinaryOperator::ShiftRight => 9,
            AstBinaryOperator::Add => 10,
            AstBinaryOperator::Subtract => 10,
            AstBinaryOperator::Multiply => 11,
            AstBinaryOperator::Divide => 11,
            AstBinaryOperator::Modulus => 11,
        }
    }

    fn to_tac(&self) -> TacBinaryOperator {
        match self {
            AstBinaryOperator::Add => TacBinaryOperator::Add,
            AstBinaryOperator::Subtract => TacBinaryOperator::Subtract,
            AstBinaryOperator::Multiply => TacBinaryOperator::Multiply,
            AstBinaryOperator::Divide => TacBinaryOperator::Divide,
            AstBinaryOperator::Modulus => TacBinaryOperator::Modulus,
            AstBinaryOperator::BitwiseAnd => TacBinaryOperator::BitwiseAnd,
            AstBinaryOperator::BitwiseOr => TacBinaryOperator::BitwiseOr,
            AstBinaryOperator::BitwiseXor => TacBinaryOperator::BitwiseXor,
            AstBinaryOperator::ShiftLeft => TacBinaryOperator::ShiftLeft,
            AstBinaryOperator::ShiftRight => TacBinaryOperator::ShiftRight,
            AstBinaryOperator::And
            | AstBinaryOperator::Or
            | AstBinaryOperator::Assign
            | AstBinaryOperator::AddAssign
            | AstBinaryOperator::SubtractAssign
            | AstBinaryOperator::MultiplyAssign
            | AstBinaryOperator::DivideAssign
            | AstBinaryOperator::ModulusAssign
            | AstBinaryOperator::BitwiseAndAssign
            | AstBinaryOperator::BitwiseOrAssign
            | AstBinaryOperator::BitwiseXorAssign
            | AstBinaryOperator::ShiftLeftAssign
            | AstBinaryOperator::ShiftRightAssign
            | AstBinaryOperator::Conditional => {
                panic!("should have been handled elsewhere")
            }
            AstBinaryOperator::Equal => TacBinaryOperator::Equal,
            AstBinaryOperator::NotEqual => TacBinaryOperator::NotEqual,
            AstBinaryOperator::LessThan => TacBinaryOperator::LessThan,
            AstBinaryOperator::LessOrEqual => TacBinaryOperator::LessOrEqual,
            AstBinaryOperator::GreaterThan => TacBinaryOperator::GreaterThan,
            AstBinaryOperator::GreaterOrEqual => TacBinaryOperator::GreaterOrEqual,
        }
    }

    fn get_base_binary_op_from_compound_op(&self) -> AstBinaryOperator {
        match self {
            AstBinaryOperator::AddAssign => AstBinaryOperator::Add,
            AstBinaryOperator::SubtractAssign => AstBinaryOperator::Subtract,
            AstBinaryOperator::MultiplyAssign => AstBinaryOperator::Multiply,
            AstBinaryOperator::DivideAssign => AstBinaryOperator::Divide,
            AstBinaryOperator::ModulusAssign => AstBinaryOperator::Modulus,
            AstBinaryOperator::BitwiseAndAssign => AstBinaryOperator::BitwiseAnd,
            AstBinaryOperator::BitwiseOrAssign => AstBinaryOperator::BitwiseOr,
            AstBinaryOperator::BitwiseXorAssign => AstBinaryOperator::BitwiseXor,
            AstBinaryOperator::ShiftLeftAssign => AstBinaryOperator::ShiftLeft,
            AstBinaryOperator::ShiftRightAssign => AstBinaryOperator::ShiftRight,
            AstBinaryOperator::Assign
            | AstBinaryOperator::Or
            | AstBinaryOperator::And
            | AstBinaryOperator::BitwiseOr
            | AstBinaryOperator::BitwiseXor
            | AstBinaryOperator::BitwiseAnd
            | AstBinaryOperator::Equal
            | AstBinaryOperator::NotEqual
            | AstBinaryOperator::LessThan
            | AstBinaryOperator::LessOrEqual
            | AstBinaryOperator::GreaterThan
            | AstBinaryOperator::GreaterOrEqual
            | AstBinaryOperator::ShiftLeft
            | AstBinaryOperator::ShiftRight
            | AstBinaryOperator::Add
            | AstBinaryOperator::Subtract
            | AstBinaryOperator::Multiply
            | AstBinaryOperator::Divide
            | AstBinaryOperator::Modulus
            | AstBinaryOperator::Conditional => {
                panic!("shouldn't be called for non-compound-assignments")
            }
        }
    }
}

impl FmtNode for AstBinaryOperator {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        f.write_str(match self {
            AstBinaryOperator::Add => "+",
            AstBinaryOperator::Subtract => "-",
            AstBinaryOperator::Multiply => "*",
            AstBinaryOperator::Divide => "/",
            AstBinaryOperator::Modulus => "%",
            AstBinaryOperator::BitwiseAnd => "&",
            AstBinaryOperator::BitwiseOr => "|",
            AstBinaryOperator::BitwiseXor => "^",
            AstBinaryOperator::ShiftLeft => "<<",
            AstBinaryOperator::ShiftRight => ">>",
            AstBinaryOperator::And => "&&",
            AstBinaryOperator::Or => "||",
            AstBinaryOperator::Equal => "==",
            AstBinaryOperator::NotEqual => "!=",
            AstBinaryOperator::LessThan => "<",
            AstBinaryOperator::LessOrEqual => "<=",
            AstBinaryOperator::GreaterThan => ">",
            AstBinaryOperator::GreaterOrEqual => ">=",
            AstBinaryOperator::Assign => "=",
            AstBinaryOperator::AddAssign => "+=",
            AstBinaryOperator::SubtractAssign => "-=",
            AstBinaryOperator::MultiplyAssign => "*=",
            AstBinaryOperator::DivideAssign => "/=",
            AstBinaryOperator::ModulusAssign => "%=",
            AstBinaryOperator::BitwiseAndAssign => "&=",
            AstBinaryOperator::BitwiseOrAssign => "|=",
            AstBinaryOperator::BitwiseXorAssign => "^=",
            AstBinaryOperator::ShiftLeftAssign => "<<=",
            AstBinaryOperator::ShiftRightAssign => ">>=",
            AstBinaryOperator::Conditional => panic!("shouldn't be called for conditional"),
        })
    }
}

impl std::str::FromStr for AstBinaryOperator {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(AstBinaryOperator::Add),
            "-" => Ok(AstBinaryOperator::Subtract),
            "*" => Ok(AstBinaryOperator::Multiply),
            "/" => Ok(AstBinaryOperator::Divide),
            "%" => Ok(AstBinaryOperator::Modulus),
            "&" => Ok(AstBinaryOperator::BitwiseAnd),
            "|" => Ok(AstBinaryOperator::BitwiseOr),
            "^" => Ok(AstBinaryOperator::BitwiseXor),
            "<<" => Ok(AstBinaryOperator::ShiftLeft),
            ">>" => Ok(AstBinaryOperator::ShiftRight),
            "&&" => Ok(AstBinaryOperator::And),
            "||" => Ok(AstBinaryOperator::Or),
            "==" => Ok(AstBinaryOperator::Equal),
            "!=" => Ok(AstBinaryOperator::NotEqual),
            "<" => Ok(AstBinaryOperator::LessThan),
            "<=" => Ok(AstBinaryOperator::LessOrEqual),
            ">" => Ok(AstBinaryOperator::GreaterThan),
            ">=" => Ok(AstBinaryOperator::GreaterOrEqual),
            "=" => Ok(AstBinaryOperator::Assign),
            "+=" => Ok(AstBinaryOperator::AddAssign),
            "-=" => Ok(AstBinaryOperator::SubtractAssign),
            "*=" => Ok(AstBinaryOperator::MultiplyAssign),
            "/=" => Ok(AstBinaryOperator::DivideAssign),
            "%=" => Ok(AstBinaryOperator::ModulusAssign),
            "&=" => Ok(AstBinaryOperator::BitwiseAndAssign),
            "|=" => Ok(AstBinaryOperator::BitwiseOrAssign),
            "^=" => Ok(AstBinaryOperator::BitwiseXorAssign),
            "<<=" => Ok(AstBinaryOperator::ShiftLeftAssign),
            ">>=" => Ok(AstBinaryOperator::ShiftRightAssign),
            "?" => Ok(AstBinaryOperator::Conditional), /* other half is parsed directly, elsewhere */
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl AstExpression {
    fn validate_and_resolve_variables(
        &mut self,
        block_tracking: &mut BlockTracking,
    ) -> Result<(), String> {
        match self {
            AstExpression::Constant(num) => {}
            AstExpression::UnaryOperator(ast_unary_op, ast_exp_inner) => {
                match ast_unary_op {
                    AstUnaryOperator::PrefixIncrement
                    | AstUnaryOperator::PrefixDecrement
                    | AstUnaryOperator::PostfixIncrement
                    | AstUnaryOperator::PostfixDecrement => {
                        let AstExpression::Var(_) = **ast_exp_inner else {
                            return Err(format!(
                                "{} is not an lvalue",
                                display_with(|f| { ast_exp_inner.fmt_node(f, 0) })
                            ));
                        };
                    }
                    _ => (),
                }

                ast_exp_inner.validate_and_resolve_variables(block_tracking)?;
            }
            AstExpression::BinaryOperator(ast_exp_left, _binary_op, ast_exp_right) => {
                ast_exp_left.validate_and_resolve_variables(block_tracking)?;
                ast_exp_right.validate_and_resolve_variables(block_tracking)?;
            }
            AstExpression::Var(ref mut ast_ident) => {
                *ast_ident = block_tracking.resolve_variable(ast_ident)?.clone();
            }
            AstExpression::Assignment(ast_exp_left, _operator, ast_exp_right) => {
                let AstExpression::Var(_) = **ast_exp_left else {
                    return Err(format!(
                        "{} is not an lvalue",
                        display_with(|f| { ast_exp_left.fmt_node(f, 0) })
                    ));
                };

                ast_exp_left.validate_and_resolve_variables(block_tracking)?;
                ast_exp_right.validate_and_resolve_variables(block_tracking)?;
            }
            AstExpression::Conditional(ast_exp_left, ast_exp_middle, ast_exp_right) => {
                ast_exp_left.validate_and_resolve_variables(block_tracking)?;
                ast_exp_middle.validate_and_resolve_variables(block_tracking)?;
                ast_exp_right.validate_and_resolve_variables(block_tracking)?;
            }
        }

        Ok(())
    }

    fn to_tac(
        &self,
        global_tracking: &mut GlobalTracking,
        instructions: &mut Vec<TacInstruction>,
    ) -> Result<TacVal, String> {
        Ok(match self {
            AstExpression::Constant(num) => TacVal::Constant(*num),
            AstExpression::UnaryOperator(
                ast_unary_op @ (AstUnaryOperator::PrefixIncrement
                | AstUnaryOperator::PrefixDecrement
                | AstUnaryOperator::PostfixIncrement
                | AstUnaryOperator::PostfixDecrement),
                ast_exp_inner,
            ) => {
                let tac_exp_val = ast_exp_inner.to_tac(global_tracking, instructions)?;

                let TacVal::Var(ref tac_exp_var) = tac_exp_val else {
                    panic!("lhs in an assignment wasn't an lvalue: {:?}", ast_exp_inner);
                };

                // Postfix operator needs to resolve to the original value, not the later value, so copy its original value first.
                let final_val = if ast_unary_op.is_postfix() {
                    let tempvar = global_tracking.allocate_temporary();
                    instructions.push(TacInstruction::CopyVal(
                        tac_exp_val.clone(),
                        tempvar.clone(),
                    ));
                    TacVal::Var(tempvar)
                } else {
                    tac_exp_val.clone()
                };

                // Prefix increment or decrement is a compound assignment with a constant value of 1 and storing in the
                // variable.
                instructions.push(TacInstruction::BinaryOp(
                    tac_exp_val.clone(),
                    ast_unary_op.get_base_binary_op_from_affix_op().to_tac(),
                    TacVal::Constant(1),
                    tac_exp_var.clone(),
                ));

                final_val
            }
            AstExpression::UnaryOperator(ast_unary_op, ast_exp_inner) => {
                let tac_exp_inner_var = ast_exp_inner.to_tac(global_tracking, instructions)?;
                let tempvar = global_tracking.allocate_temporary();
                instructions.push(TacInstruction::UnaryOp(
                    ast_unary_op.to_tac(),
                    tac_exp_inner_var,
                    tempvar.clone(),
                ));
                TacVal::Var(tempvar)
            }
            // Logical AND and OR are short-circuit and can't be treated as regular binary operators.
            AstExpression::BinaryOperator(
                ast_exp_left,
                binary_op @ (AstBinaryOperator::And | AstBinaryOperator::Or),
                ast_exp_right,
            ) => {
                let is_and = if let AstBinaryOperator::And = binary_op {
                    true
                } else {
                    false
                };

                let tempvar = global_tracking.allocate_temporary();

                let shortcircuit_label = TacLabel(global_tracking.allocate_label(if is_and {
                    "and_shortcircuit"
                } else {
                    "or_shortcircuit"
                }));

                let end_label = TacLabel(global_tracking.allocate_label(if is_and {
                    "and_end"
                } else {
                    "or_end"
                }));

                // First emit code to calculate the left hand side.
                let tac_exp_left_var = ast_exp_left.to_tac(global_tracking, instructions)?;

                // Test if the left hand side would already give a conclusive answer for the operator, i.e. 0 for AND
                // or 1 for OR. Then we can jump to the short circuit handling.
                instructions.push((if is_and {
                    TacInstruction::JumpIfZero
                } else {
                    TacInstruction::JumpIfNotZero
                })(
                    tac_exp_left_var, shortcircuit_label.clone()
                ));

                // If we made it this far, the right hand side has to be executed.
                let tac_exp_right_var = ast_exp_right.to_tac(global_tracking, instructions)?;

                // Test if the right hand side is conclusive, and if so, also jump to the same short circuit handler.
                instructions.push((if is_and {
                    TacInstruction::JumpIfZero
                } else {
                    TacInstruction::JumpIfNotZero
                })(
                    tac_exp_right_var, shortcircuit_label.clone()
                ));

                // If we made it this far, then the result must be 1 for AND or 0 for OR. Copy it to the result and jump
                // over the short circuit handling code, to the end.
                instructions.push(TacInstruction::CopyVal(
                    TacVal::Constant(if is_and { 1 } else { 0 }),
                    tempvar.clone(),
                ));

                instructions.push(TacInstruction::Jump(end_label.clone()));

                // And here is the short circuit handling code, to set the result to the value that would cause a short
                // circuit for the operator: 0 for AND, or 1 for OR. And then fall through to the end.
                instructions.push(TacInstruction::Label(shortcircuit_label));
                instructions.push(TacInstruction::CopyVal(
                    TacVal::Constant(if is_and { 0 } else { 1 }),
                    tempvar.clone(),
                ));

                instructions.push(TacInstruction::Label(end_label));

                TacVal::Var(tempvar)
            }
            AstExpression::BinaryOperator(ast_exp_left, ast_binary_op, ast_exp_right) => {
                let tac_exp_left_val = ast_exp_left.to_tac(global_tracking, instructions)?;
                let tac_exp_right_val = ast_exp_right.to_tac(global_tracking, instructions)?;
                let tempvar = global_tracking.allocate_temporary();
                instructions.push(TacInstruction::BinaryOp(
                    tac_exp_left_val,
                    ast_binary_op.to_tac(),
                    tac_exp_right_val,
                    tempvar.clone(),
                ));
                TacVal::Var(tempvar)
            }
            AstExpression::Var(ast_ident) => TacVal::Var(ast_ident.to_tac()),
            AstExpression::Assignment(ast_exp_left, ast_binary_op, ast_exp_right) => {
                let tac_exp_left_val = ast_exp_left.to_tac(global_tracking, instructions)?;

                let TacVal::Var(ref tac_exp_left_var) = tac_exp_left_val else {
                    panic!("lhs in an assignment wasn't an lvalue: {:?}", ast_exp_left);
                };

                let tac_exp_right_val = ast_exp_right.to_tac(global_tracking, instructions)?;

                // Straight assignment with just the = operator is a simple copy.
                if let AstBinaryOperator::Assign = ast_binary_op {
                    instructions.push(TacInstruction::CopyVal(
                        tac_exp_right_val,
                        tac_exp_left_var.clone(),
                    ));
                } else {
                    // This is a compound assignment like +=. So we have to do the binary operation, and the output can
                    // be the left hand side variable.
                    instructions.push(TacInstruction::BinaryOp(
                        tac_exp_left_val.clone(),
                        ast_binary_op.get_base_binary_op_from_compound_op().to_tac(),
                        tac_exp_right_val,
                        tac_exp_left_var.clone(),
                    ));
                }

                tac_exp_left_val
            }
            AstExpression::Conditional(ast_exp_left, ast_exp_middle, ast_exp_right) => {
                let result_var = global_tracking.allocate_temporary();

                let right_begin_label = TacLabel(global_tracking.allocate_label("cond_else_begin"));
                let end_label = TacLabel(global_tracking.allocate_label("cond_end"));

                let tac_exp_left_val = ast_exp_left.to_tac(global_tracking, instructions)?;

                // If the condition is false, jump over the middle--the "then" part--to the beginning of the right--the
                // "else" part.
                instructions.push(TacInstruction::JumpIfZero(
                    tac_exp_left_val,
                    right_begin_label.clone(),
                ));

                let tac_exp_middle_val = ast_exp_middle.to_tac(global_tracking, instructions)?;

                // After the then-clause is done, store the result in the overall expression's result temporary variable
                // and jump to the end.
                instructions.push(TacInstruction::CopyVal(
                    tac_exp_middle_val,
                    result_var.clone(),
                ));

                instructions.push(TacInstruction::Jump(end_label.clone()));

                // And then emit the label for the start of the else part.
                instructions.push(TacInstruction::Label(right_begin_label));

                // And then the else part itself.
                let tac_exp_right_val = ast_exp_right.to_tac(global_tracking, instructions)?;
                instructions.push(TacInstruction::CopyVal(
                    tac_exp_right_val,
                    result_var.clone(),
                ));

                instructions.push(TacInstruction::Label(end_label));

                TacVal::Var(result_var)
            }
        })
    }
}

impl FmtNode for AstExpression {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        match self {
            AstExpression::Constant(val) => write!(f, "{}", val)?,
            AstExpression::UnaryOperator(operator, expr) => {
                write!(f, "(")?;
                operator.fmt_node(f, 0)?;
                expr.fmt_node(f, 0)?;
                write!(f, ")")?;
            }
            AstExpression::BinaryOperator(left, operator, right) => {
                write!(f, "(")?;
                left.fmt_node(f, 0)?;
                write!(f, " ")?;
                operator.fmt_node(f, 0)?;
                write!(f, " ")?;
                right.fmt_node(f, 0)?;
                write!(f, ")")?;
            }
            AstExpression::Var(ident) => {
                f.write_str(&ident.0)?;
            }
            AstExpression::Assignment(left, operator, right) => {
                write!(f, "(")?;
                left.fmt_node(f, 0)?;
                write!(f, " ")?;
                operator.fmt_node(f, 0)?;
                write!(f, " ")?;
                right.fmt_node(f, 0)?;
                write!(f, ")")?;
            }
            AstExpression::Conditional(left, middle, right) => {
                write!(f, "(")?;
                left.fmt_node(f, 0)?;
                write!(f, " ? ")?;
                middle.fmt_node(f, 0)?;
                write!(f, " : ")?;
                right.fmt_node(f, 0)?;
                write!(f, ")")?;
            }
        }

        Ok(())
    }
}

impl TacProgram {
    fn to_asm(&self) -> Result<AsmProgram, String> {
        let mut functions = Vec::new();
        for func in self.functions.iter() {
            functions.push(func.to_asm()?);
        }

        Ok(AsmProgram { functions })
    }
}

impl FmtNode for TacProgram {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        Self::fmt_nodelist(f, self.functions.iter(), "\n\n", 0)
    }
}

impl TacFunction {
    fn to_asm(&self) -> Result<AsmFunction, String> {
        let mut body = Vec::new();
        for instruction in self.body.iter() {
            instruction.to_asm(&mut body)?;
        }

        Ok(AsmFunction {
            name: self.name.clone(),
            body,
        })
    }
}

impl FmtNode for TacFunction {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        writeln!(f, "FUNC {}", self.name)?;
        Self::fmt_nodelist(f, self.body.iter(), "\n", indent_levels + 1)
    }
}

impl TacLabel {
    fn to_asm(&self) -> Result<AsmLabel, String> {
        Ok(AsmLabel::new(&self.0))
    }
}

impl TacInstruction {
    fn to_asm(&self, func_body: &mut Vec<AsmInstruction>) -> Result<(), String> {
        match self {
            TacInstruction::Return(val) => {
                func_body.push(AsmInstruction::Mov(val.to_asm()?, AsmLocation::Reg("eax")));
                func_body.push(AsmInstruction::Ret(0));
            }

            // !a is the same as (a == 0), so do a comparison to zero and set if equal.
            TacInstruction::UnaryOp(TacUnaryOperator::Not, src_val, dest_var) => {
                // Compare the value to zero.
                func_body.push(AsmInstruction::Cmp(AsmVal::Imm(0), src_val.to_asm()?));

                // Clear the destination before the SetCc.
                let dest_asm_loc = dest_var.to_asm()?;
                func_body.push(AsmInstruction::Mov(AsmVal::Imm(0), dest_asm_loc.clone()));

                // Set if equal to zero.
                func_body.push(AsmInstruction::SetCc(AsmCondCode::E, dest_asm_loc));
            }
            TacInstruction::UnaryOp(unary_op, src_val, dest_var) => {
                let dest_asm_loc = dest_var.to_asm()?;
                func_body.push(AsmInstruction::Mov(src_val.to_asm()?, dest_asm_loc.clone()));
                func_body.push(AsmInstruction::UnaryOp(unary_op.to_asm()?, dest_asm_loc));
            }

            // Divide and modulus have weird assembly instructions for them, so we need to treat them specially.
            TacInstruction::BinaryOp(
                left_val,
                binary_op @ (TacBinaryOperator::Divide | TacBinaryOperator::Modulus),
                right_val,
                dest_var,
            ) => {
                // idiv takes the numerator in EDX:EAX (that's low 32 bits in EAX and high 32 bits in EDX). Since we're
                // operating only on 32 bits, move the value to EAX first.
                func_body.push(AsmInstruction::Mov(
                    left_val.to_asm()?,
                    AsmLocation::Reg("eax"),
                ));

                // cdq sign extends EAX into EDX. Wow, it's purpose-built for this.
                func_body.push(AsmInstruction::Cdq);

                // The idiv instruction takes only the denominator as an argument, implicitly operating on EDX:EAX as
                // the numerator.
                func_body.push(AsmInstruction::Idiv(right_val.to_asm()?));

                // The result of idiv is stored with quotient in EAX and remainer in EDX.
                let result_loc = AsmLocation::Reg(if let TacBinaryOperator::Divide = binary_op {
                    "eax"
                } else {
                    assert!(if let TacBinaryOperator::Modulus = binary_op {
                        true
                    } else {
                        false
                    });

                    "edx"
                });

                // Move from idiv's result register to the intended destination.
                func_body.push(AsmInstruction::Mov(
                    AsmVal::Loc(result_loc),
                    dest_var.to_asm()?,
                ));
            }

            // Relational operators are converted to a SetCc with a different condition code each.
            TacInstruction::BinaryOp(
                left_val,
                comparison_op @ (TacBinaryOperator::Equal
                | TacBinaryOperator::NotEqual
                | TacBinaryOperator::LessThan
                | TacBinaryOperator::LessOrEqual
                | TacBinaryOperator::GreaterThan
                | TacBinaryOperator::GreaterOrEqual),
                right_val,
                dest_var,
            ) => {
                func_body.push(AsmInstruction::Cmp(right_val.to_asm()?, left_val.to_asm()?));

                let dest_asm_loc = dest_var.to_asm()?;

                // Zero out the destination because the SetCc instruction only writes the low 1 byte.
                func_body.push(AsmInstruction::Mov(AsmVal::Imm(0), dest_asm_loc.clone()));

                func_body.push(AsmInstruction::SetCc(
                    comparison_op.get_condition_code(),
                    dest_asm_loc,
                ));
            }

            // Other binary operators are handled in a much more straightforward way. They take the left hand side of
            // the operator in the destination register.
            TacInstruction::BinaryOp(left_val, binary_op, right_val, dest_var) => {
                // Normal binary operators take the left hand side in the destination register, so move it there first.
                let dest_asm_loc = dest_var.to_asm()?;
                func_body.push(AsmInstruction::Mov(
                    left_val.to_asm()?,
                    dest_asm_loc.clone(),
                ));

                // The operator takes the left and right sides as arguments, with left also implicitly being treated as
                // the destination.
                func_body.push(AsmInstruction::BinaryOp(
                    binary_op.to_asm()?,
                    right_val.to_asm()?,
                    dest_asm_loc,
                ));
            }
            TacInstruction::CopyVal(src_val, dest_loc) => {
                func_body.push(AsmInstruction::Mov(src_val.to_asm()?, dest_loc.to_asm()?));
            }
            TacInstruction::Jump(label) => {
                func_body.push(AsmInstruction::Jmp(label.to_asm()?));
            }
            jump_type @ (TacInstruction::JumpIfZero(val, label)
            | TacInstruction::JumpIfNotZero(val, label)) => {
                // First compare to zero.
                func_body.push(AsmInstruction::Cmp(AsmVal::Imm(0), val.to_asm()?));

                // Then jump if they were equal or not equal, depending on the jump type.
                func_body.push(AsmInstruction::JmpCc(
                    if let TacInstruction::JumpIfZero(_, _) = jump_type {
                        AsmCondCode::E
                    } else {
                        AsmCondCode::NE
                    },
                    label.to_asm()?,
                ));
            }
            TacInstruction::Label(label) => {
                func_body.push(AsmInstruction::Label(label.to_asm()?));
            }
        }

        Ok(())
    }
}

impl FmtNode for TacInstruction {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;
        match self {
            TacInstruction::Return(val) => {
                write!(f, "return ")?;
                val.fmt_node(f, 0)?;
            }
            TacInstruction::UnaryOp(unary_op, src, dest) => {
                dest.fmt_node(f, 0)?;
                write!(f, " = ")?;
                unary_op.fmt(f)?;
                write!(f, " ")?;
                src.fmt_node(f, 0)?;
            }
            TacInstruction::BinaryOp(left_val, binary_op, right_val, dest_var) => {
                dest_var.fmt_node(f, 0)?;
                write!(f, " = ")?;
                left_val.fmt_node(f, 0)?;
                write!(f, " ")?;
                binary_op.fmt(f)?;
                write!(f, " ")?;
                right_val.fmt_node(f, 0)?;
            }
            TacInstruction::CopyVal(src_val, dest_loc) => {
                dest_loc.fmt_node(f, 0)?;
                write!(f, " = ")?;
                src_val.fmt_node(f, 0)?;
            }
            TacInstruction::Jump(label) => {
                write!(f, "jump :{}", label.0)?;
            }
            TacInstruction::JumpIfZero(val, label) => {
                write!(f, "jump :{} if ", label.0)?;
                val.fmt_node(f, 0)?;
                write!(f, " == 0")?;
            }
            TacInstruction::JumpIfNotZero(val, label) => {
                write!(f, "jump :{} if ", label.0)?;
                val.fmt_node(f, 0)?;
                write!(f, " != 0")?;
            }
            TacInstruction::Label(label) => {
                writeln!(f)?;
                write!(f, "{}:", label.0)?;
            }
        }

        Ok(())
    }
}

impl FmtNode for TacVar {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl TacVal {
    fn to_asm(&self) -> Result<AsmVal, String> {
        Ok(match self {
            TacVal::Constant(num) => AsmVal::Imm(*num),
            TacVal::Var(var) => AsmVal::Loc(var.to_asm()?),
        })
    }
}

impl FmtNode for TacVal {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        match self {
            TacVal::Constant(num) => {
                write!(f, "{}", num)?;
            }
            TacVal::Var(var) => {
                var.fmt_node(f, 0)?;
            }
        }

        Ok(())
    }
}

impl TacVar {
    fn to_asm(&self) -> Result<AsmLocation, String> {
        Ok(AsmLocation::PseudoReg(self.0.clone()))
    }
}

impl TacUnaryOperator {
    fn to_asm(&self) -> Result<AsmUnaryOperator, String> {
        Ok(match self {
            TacUnaryOperator::Negation => AsmUnaryOperator::Neg,
            TacUnaryOperator::BitwiseNot => AsmUnaryOperator::Not,
            TacUnaryOperator::Not => panic!("unary not handled differently"),
        })
    }
}

impl fmt::Display for TacUnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            TacUnaryOperator::Negation => "Negate",
            TacUnaryOperator::BitwiseNot => "BitwiseNot",
            TacUnaryOperator::Not => "Not",
        })
    }
}

impl TacBinaryOperator {
    fn to_asm(&self) -> Result<AsmBinaryOperator, String> {
        Ok(match self {
            TacBinaryOperator::Add => AsmBinaryOperator::Add,
            TacBinaryOperator::Subtract => AsmBinaryOperator::Subtract,
            TacBinaryOperator::Multiply => AsmBinaryOperator::Imul,
            TacBinaryOperator::BitwiseAnd => AsmBinaryOperator::And,
            TacBinaryOperator::BitwiseOr => AsmBinaryOperator::Or,
            TacBinaryOperator::BitwiseXor => AsmBinaryOperator::Xor,
            TacBinaryOperator::ShiftLeft => AsmBinaryOperator::Shl,
            TacBinaryOperator::ShiftRight => AsmBinaryOperator::Sar,
            TacBinaryOperator::Divide | TacBinaryOperator::Modulus => {
                panic!("divide/modulus should have been handled elsewhere")
            }
            TacBinaryOperator::Equal
            | TacBinaryOperator::NotEqual
            | TacBinaryOperator::LessThan
            | TacBinaryOperator::LessOrEqual
            | TacBinaryOperator::GreaterThan
            | TacBinaryOperator::GreaterOrEqual => panic!("relational operators handled elsewhere"),
        })
    }

    fn get_condition_code(&self) -> AsmCondCode {
        match self {
            TacBinaryOperator::Equal => AsmCondCode::E,
            TacBinaryOperator::NotEqual => AsmCondCode::NE,
            TacBinaryOperator::LessThan => AsmCondCode::L,
            TacBinaryOperator::LessOrEqual => AsmCondCode::LE,
            TacBinaryOperator::GreaterThan => AsmCondCode::G,
            TacBinaryOperator::GreaterOrEqual => AsmCondCode::GE,
            _ => panic!("shouldn't call this on other binary operator types"),
        }
    }
}

impl fmt::Display for TacBinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            TacBinaryOperator::Add => "+",
            TacBinaryOperator::Subtract => "-",
            TacBinaryOperator::Multiply => "*",
            TacBinaryOperator::Divide => "/",
            TacBinaryOperator::Modulus => "%",
            TacBinaryOperator::BitwiseAnd => "&",
            TacBinaryOperator::BitwiseOr => "|",
            TacBinaryOperator::BitwiseXor => "^",
            TacBinaryOperator::ShiftLeft => "<<",
            TacBinaryOperator::ShiftRight => ">>",
            TacBinaryOperator::Equal => "==",
            TacBinaryOperator::NotEqual => "!=",
            TacBinaryOperator::LessThan => "<",
            TacBinaryOperator::LessOrEqual => "<=",
            TacBinaryOperator::GreaterThan => ">",
            TacBinaryOperator::GreaterOrEqual => ">=",
        })
    }
}

impl AsmProgram {
    fn finalize(&mut self) -> Result<(), String> {
        for func in self.functions.iter_mut() {
            func.finalize()?;
        }

        Ok(())
    }

    fn emit_code(&self) -> Result<String, String> {
        Ok(format!(
            "{}",
            display_with(|f| {
                write!(
                    f,
                    "INCLUDELIB msvcrt.lib\n\
                       .DATA\n\
                       \n\
                       .CODE\n"
                )?;

                for func in self.functions.iter() {
                    func.emit_code(f)?;
                }

                write!(f, "\nEND")?;

                Ok(())
            })
        ))
    }
}

impl FmtNode for AsmProgram {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        Self::fmt_nodelist(f, self.functions.iter(), "\n\n", 0)
    }
}

impl AsmFunction {
    fn finalize(&mut self) -> Result<(), String> {
        let mut frame = FuncStackFrame::new();

        // TODO: consider an iterator over all locations
        for inst in self.body.iter_mut() {
            match inst {
                AsmInstruction::Mov(src_val, dest_loc) => {
                    src_val.resolve_pseudoregister(&mut frame)?;
                    dest_loc.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::UnaryOp(_, dest_loc) => {
                    dest_loc.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::BinaryOp(_, src_val, dest_loc) => {
                    src_val.resolve_pseudoregister(&mut frame)?;
                    dest_loc.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::Idiv(denom_val) => {
                    denom_val.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::Cdq => {}
                AsmInstruction::AllocateStack(_) => {}
                AsmInstruction::Ret(_) => {}
                AsmInstruction::Cmp(src1_val, src2_val) => {
                    src1_val.resolve_pseudoregister(&mut frame)?;
                    src2_val.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::SetCc(_cond_code, dest_loc) => {
                    dest_loc.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::Jmp(_label) => {}
                AsmInstruction::JmpCc(_cond_code, _label) => {}
                AsmInstruction::Label(_label) => {}
            }
        }

        // Allocate the stack frame's size at the beginning of the function body.
        self.body
            .insert(0, AsmInstruction::AllocateStack(frame.size()));

        // In any place that has a Ret instruction, fill it in with the stack frame size.
        for inst in self.body.iter_mut() {
            if let AsmInstruction::Ret(size) = inst {
                *size = frame.size();
            }
        }

        for inst in self.body.iter_mut() {
            inst.convert_to_rsp_offset(&frame);
        }

        let mut i;

        // Shift left and shift right only allow immedate or CL (that's 8-bit ecx) register as the right hand side. If
        // the rhs isn't in there already, move it first.
        i = 0;
        while i < self.body.len() {
            if let AsmInstruction::BinaryOp(
                AsmBinaryOperator::Shl | AsmBinaryOperator::Sar,
                ref mut src_val,
                _dest_loc,
            ) = &mut self.body[i]
            {
                if src_val.get_base_reg_name() != Some("cx") {
                    let real_src_val = src_val.clone();
                    *src_val = AsmVal::Loc(AsmLocation::Reg("cl"));

                    self.body.insert(
                        i,
                        AsmInstruction::Mov(real_src_val, AsmLocation::Reg("ecx")),
                    );

                    // We made a change, so rerun the loop on this index in case further fixups are needed.
                    continue;
                }
            }

            i += 1;
        }

        // For any Mov that uses a stack offset for both src and dest, x64 assembly requires that we first store it in a
        // temporary register.
        i = 0;
        while i < self.body.len() {
            if let AsmInstruction::Mov(
                ref _src_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
                ref mut dest_loc @ AsmLocation::RspOffset(_, _),
            ) = &mut self.body[i]
            {
                let real_dest = dest_loc.clone();
                *dest_loc = AsmLocation::Reg("r10d");

                self.body.insert(
                    i + 1,
                    AsmInstruction::Mov(AsmVal::Loc(AsmLocation::Reg("r10d")), real_dest),
                );

                // We made a change, so rerun the loop on this index in case further fixups are needed.
                continue;
            }

            i += 1;
        }

        // Multiply doesn't allow a memory address as the destination. Fix it up so the destination is a temporary
        // register and then written to the destination memory address.
        i = 0;
        while i < self.body.len() {
            if let AsmInstruction::BinaryOp(
                AsmBinaryOperator::Imul,
                _src_val,
                ref mut dest_loc @ AsmLocation::RspOffset(_, _),
            ) = &mut self.body[i]
            {
                let real_dest = dest_loc.clone();

                // Rewrite the multiply instruction itself to operate against a temporary register instead of a
                // memory address.
                *dest_loc = AsmLocation::Reg("r11d");

                // Insert a mov before the multiply, to put the destination value in the temporary register.
                self.body.insert(
                    i,
                    AsmInstruction::Mov(AsmVal::Loc(real_dest.clone()), AsmLocation::Reg("r11d")),
                );

                // Insert a mov instruction after the multiply, to put the destination value into the intended
                // memory address.
                self.body.insert(
                    i + 2,
                    AsmInstruction::Mov(AsmVal::Loc(AsmLocation::Reg("r11d")), real_dest),
                );

                // We made a change, so rerun the loop on this index in case further fixups are needed.
                continue;
            }

            i += 1;
        }

        // Cmp of course can't take immediates for both left and right sides, because otherwise the value is known at
        // compile time. So move the right hand side into a destination register first.
        i = 0;
        while i < self.body.len() {
            if let AsmInstruction::Cmp(_src_val, ref mut dest_val @ AsmVal::Imm(_)) =
                &mut self.body[i]
            {
                let real_dest_val = dest_val.clone();
                *dest_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                self.body.insert(
                    i,
                    AsmInstruction::Mov(real_dest_val, AsmLocation::Reg("r10d")),
                );

                // We made a change, so rerun the loop on this index in case further fixups are needed.
                continue;
            }

            i += 1;
        }

        // Cmp doesn't support memory addresses for both operands, so move the left hand side into a register first.
        i = 0;
        while i < self.body.len() {
            if let AsmInstruction::Cmp(
                ref mut src_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
                ref _dest_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
            ) = &mut self.body[i]
            {
                let real_src_val = src_val.clone();
                *src_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                self.body.insert(
                    i,
                    AsmInstruction::Mov(real_src_val, AsmLocation::Reg("r10d")),
                );

                // We made a change, so rerun the loop on this index in case further fixups are needed.
                continue;
            }

            i += 1;
        }

        // For any binary operator that uses a stack offset for both right hand side and dest, x64 assembly requires
        // that we first store the destination in a temporary register.
        i = 0;
        while i < self.body.len() {
            if let AsmInstruction::BinaryOp(
                _binary_op,
                ref mut src_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
                ref _dest_loc @ AsmLocation::RspOffset(_, _),
            ) = &mut self.body[i]
            {
                let real_src_val = src_val.clone();
                *src_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                self.body.insert(
                    i,
                    AsmInstruction::Mov(real_src_val, AsmLocation::Reg("r10d")),
                );

                // We made a change, so rerun the loop on this index in case further fixups are needed.
                continue;
            }

            i += 1;
        }

        // idiv doesn't accept an immediate value as the operand, so fixup to put the immediate in a register first.
        i = 0;
        while i < self.body.len() {
            if let AsmInstruction::Idiv(ref mut denom_val @ AsmVal::Imm(_)) = &mut self.body[i] {
                let real_denom_val = denom_val.clone();
                *denom_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                // Insert a mov before this idiv to put its immediate value in a register.
                self.body.insert(
                    i,
                    AsmInstruction::Mov(real_denom_val, AsmLocation::Reg("r10d")),
                );

                // We made a change, so rerun the loop on this index in case further fixups are needed.
                continue;
            }

            i += 1;
        }

        Ok(())
    }

    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{} PROC", self.name)?;

        for inst in self.body.iter() {
            inst.emit_code(f)?;
        }

        writeln!(f, "{} ENDP", self.name)?;
        Ok(())
    }
}

impl FmtNode for AsmFunction {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        writeln!(f, "FUNC {}", self.name)?;
        Self::fmt_nodelist(f, self.body.iter(), "\n", 1)
    }
}

impl AsmLabel {
    fn new(name: &str) -> Self {
        Self(format!("$L{}", name))
    }

    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FmtNode for AsmLabel {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl AsmInstruction {
    fn convert_to_rsp_offset(&mut self, frame: &FuncStackFrame) {
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                src_val.convert_to_rsp_offset(frame);
                dest_loc.convert_to_rsp_offset(frame);
            }
            AsmInstruction::UnaryOp(_, dest_loc) => {
                dest_loc.convert_to_rsp_offset(frame);
            }
            AsmInstruction::BinaryOp(_, src_val, dest_loc) => {
                src_val.convert_to_rsp_offset(frame);
                dest_loc.convert_to_rsp_offset(frame);
            }
            AsmInstruction::Idiv(denom_val) => {
                denom_val.convert_to_rsp_offset(frame);
            }
            AsmInstruction::Cdq => {}
            AsmInstruction::AllocateStack(_) => {}
            AsmInstruction::Ret(_) => {}
            AsmInstruction::Cmp(src1_val, src2_val) => {
                src1_val.convert_to_rsp_offset(frame);
                src2_val.convert_to_rsp_offset(frame);
            }
            AsmInstruction::SetCc(_cond_code, dest_loc) => {
                dest_loc.convert_to_rsp_offset(frame);
            }
            AsmInstruction::Jmp(_label) => {}
            AsmInstruction::JmpCc(_cond_code, _label) => {}
            AsmInstruction::Label(_label) => {}
        }
    }

    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                format_code_and_comment(
                    f,
                    |f| {
                        write!(f, "mov ")?;
                        dest_loc.emit_code(f, 4)?;
                        write!(f, ",")?;
                        src_val.emit_code(f, 4)
                    },
                    |f| {
                        dest_loc.fmt_asm_comment(f)?;
                        write!(f, " <- ")?;
                        src_val.fmt_asm_comment(f)
                    },
                )?;
            }
            AsmInstruction::UnaryOp(unary_op, dest_loc) => {
                format_code_and_comment(
                    f,
                    |f| {
                        unary_op.emit_code(f)?;
                        write!(f, " ")?;
                        dest_loc.emit_code(f, 4)
                    },
                    |f| {
                        unary_op.emit_code(f)?;
                        write!(f, " ")?;
                        dest_loc.fmt_asm_comment(f)
                    },
                )?;
            }
            AsmInstruction::BinaryOp(binary_op, src_val, dest_loc) => {
                // sar and shl require only 1-byte operand for the shift amount.
                let src_size = match binary_op {
                    AsmBinaryOperator::Shl | AsmBinaryOperator::Sar => 1,
                    _ => 4,
                };

                format_code_and_comment(
                    f,
                    |f| {
                        binary_op.emit_code(f)?;
                        write!(f, " ")?;
                        dest_loc.emit_code(f, 4)?;
                        write!(f, ",")?;
                        src_val.emit_code(f, src_size)
                    },
                    |f| {
                        dest_loc.fmt_asm_comment(f)?;
                        write!(f, " <- ")?;
                        dest_loc.fmt_asm_comment(f)?;
                        write!(f, " ")?;
                        binary_op.fmt_node(f, 0)?;
                        write!(f, " ")?;
                        src_val.fmt_asm_comment(f)
                    },
                )?;
            }
            AsmInstruction::Cdq => {
                writeln!(f, "    cdq")?;
            }
            AsmInstruction::Idiv(denom_val) => {
                format_code_and_comment(
                    f,
                    |f| {
                        write!(f, "idiv ")?;
                        denom_val.emit_code(f, 4)
                    },
                    |f| {
                        write!(f, "edx:eax <- idiv ")?;
                        denom_val.fmt_asm_comment(f)
                    },
                )?;
            }
            AsmInstruction::AllocateStack(size) => {
                format_code_and_comment(
                    f,
                    |f| write!(f, "sub rsp,{}", size),
                    |f| write!(f, "stack_alloc {} bytes", size),
                )?;
            }
            AsmInstruction::Ret(size) => {
                format_code_and_comment(
                    f,
                    |f| write!(f, "add rsp,{}", size),
                    |f| write!(f, "stack_alloc {} bytes", size),
                )?;

                writeln!(f, "    ret")?;
            }
            AsmInstruction::Cmp(src1_val, src2_val) => {
                format_code_and_comment(
                    f,
                    |f| {
                        write!(f, "cmp ")?;
                        src2_val.emit_code(f, 4)?;
                        write!(f, ",")?;
                        src1_val.emit_code(f, 4)
                    },
                    |f| {
                        src2_val.fmt_asm_comment(f)?;
                        write!(f, " cmp ")?;
                        src1_val.fmt_asm_comment(f)
                    },
                )?;
            }
            AsmInstruction::SetCc(cond_code, dest_loc) => {
                format_code_and_comment(
                    f,
                    |f| {
                        // Makes an instruction like setne
                        write!(f, "set")?;
                        cond_code.emit_code(f)?;

                        write!(f, " ")?;

                        // SetCc only allows 1-byte destination operand.
                        dest_loc.emit_code(f, 1)
                    },
                    |f| {
                        write!(f, "set")?;
                        cond_code.emit_code(f)?;
                        write!(f, " ")?;
                        dest_loc.fmt_asm_comment(f)
                    },
                )?;
            }
            AsmInstruction::Jmp(label) => {
                write!(f, "    jmp ")?;
                label.emit_code(f)?;
                writeln!(f)?;
            }
            AsmInstruction::JmpCc(cond_code, label) => {
                // Together this makes assembly instructions like jge, jne, etc..
                write!(f, "    j")?;
                cond_code.emit_code(f)?;

                write!(f, " ")?;
                label.emit_code(f)?;
                writeln!(f)?;
            }
            AsmInstruction::Label(label) => {
                writeln!(f)?;
                label.emit_code(f)?;
                writeln!(f, ":")?;
            }
        }

        Ok(())
    }
}

impl FmtNode for AsmInstruction {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                f.write_str("Mov ")?;
                src_val.fmt_node(f, 0)?;
                f.write_str(" -> ")?;
                dest_loc.fmt_node(f, 0)?;
            }
            AsmInstruction::UnaryOp(unary_op, dest_loc) => {
                unary_op.fmt_node(f, 0)?;
                f.write_str(" ")?;
                dest_loc.fmt_node(f, 0)?;
            }
            AsmInstruction::BinaryOp(binary_op, src_val, dest_loc) => {
                dest_loc.fmt_node(f, 0)?;
                f.write_str(" ")?;
                binary_op.fmt_node(f, 0)?;
                f.write_str(" ")?;
                src_val.fmt_node(f, 0)?;
                f.write_str(" -> ")?;
                dest_loc.fmt_node(f, 0)?;
            }
            AsmInstruction::Cdq => {
                f.write_str("Cdq")?;
            }
            AsmInstruction::Idiv(denom_val) => {
                f.write_str("Idiv ")?;
                denom_val.fmt_node(f, 0)?;
            }
            AsmInstruction::AllocateStack(size) => {
                write!(f, "AllocateStack {}", size)?;
            }
            AsmInstruction::Ret(size) => {
                write!(f, "Ret (dealloc {} stack)", size)?;
            }
            AsmInstruction::Cmp(src1_val, src2_val) => {
                f.write_str("Cmp ")?;
                src1_val.fmt_node(f, 0)?;
                f.write_str(", ")?;
                src2_val.fmt_node(f, 0)?;
            }
            AsmInstruction::SetCc(cond_code, dest_loc) => {
                write!(f, "Set{} ", cond_code)?;
                dest_loc.fmt_node(f, 0)?;
            }
            AsmInstruction::Jmp(label) => {
                write!(f, "Jmp ")?;
                label.fmt_node(f, 0)?;
            }
            AsmInstruction::JmpCc(cond_code, label) => {
                write!(f, "Jmp{} ", cond_code)?;
                label.fmt_node(f, 0)?;
            }
            AsmInstruction::Label(label) => {
                label.fmt_node(f, 0)?;
                write!(f, ":")?;
            }
        }

        Ok(())
    }
}

impl AsmVal {
    fn get_base_reg_name(&self) -> Option<&'static str> {
        match self {
            AsmVal::Loc(loc) => loc.get_base_reg_name(),
            _ => None,
        }
    }

    fn convert_to_rsp_offset(&mut self, frame: &FuncStackFrame) {
        if let AsmVal::Loc(loc) = self {
            loc.convert_to_rsp_offset(frame);
        }
    }

    fn resolve_pseudoregister(&mut self, frame: &mut FuncStackFrame) -> Result<(), String> {
        if let AsmVal::Loc(loc) = self {
            return loc.resolve_pseudoregister(frame);
        }

        Ok(())
    }

    fn emit_code(&self, f: &mut fmt::Formatter, size: u8) -> fmt::Result {
        match self {
            AsmVal::Imm(num) => {
                write!(f, "{}", num)?;
            }
            AsmVal::Loc(loc) => {
                loc.emit_code(f, size)?;
            }
        }

        Ok(())
    }

    fn fmt_asm_comment(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmVal::Imm(num) => {
                write!(f, "{}", num)?;
            }
            AsmVal::Loc(loc) => {
                loc.fmt_asm_comment(f)?;
            }
        }

        Ok(())
    }
}

impl FmtNode for AsmVal {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        match self {
            AsmVal::Imm(num) => {
                write!(f, "{}", num)?;
            }
            AsmVal::Loc(loc) => {
                loc.fmt_node(f, 0)?;
            }
        }

        Ok(())
    }
}

impl AsmLocation {
    fn get_base_reg_name(&self) -> Option<&'static str> {
        match self {
            AsmLocation::Reg("rax" | "eax" | "ax" | "al") => Some("ax"),
            AsmLocation::Reg("rbx" | "ebx" | "bx" | "bl") => Some("bx"),
            AsmLocation::Reg("rcx" | "ecx" | "cx" | "cl") => Some("cx"),
            AsmLocation::Reg("rdx" | "edx" | "dx" | "dl") => Some("dx"),
            AsmLocation::Reg("rsi" | "esi" | "si" | "sil") => Some("si"),
            AsmLocation::Reg("rdi" | "edi" | "di" | "dil") => Some("di"),
            AsmLocation::Reg("rbp" | "ebp" | "bp" | "bpl") => Some("bp"),
            AsmLocation::Reg("rsp" | "esp" | "sp" | "spl") => Some("sp"),
            AsmLocation::Reg("r8" | "r8d" | "r8w" | "r8b" | "r8l") => Some("r8"),
            AsmLocation::Reg("r9" | "r9d" | "r9w" | "r9b" | "r9l") => Some("r9"),
            AsmLocation::Reg("r10" | "r10d" | "r10w" | "r10b" | "r10l") => Some("r10"),
            AsmLocation::Reg("r11" | "r11d" | "r11w" | "r11b" | "r11l") => Some("r11"),
            AsmLocation::Reg("r12" | "r12d" | "r12w" | "r12b" | "r12l") => Some("r12"),
            AsmLocation::Reg("r13" | "r13d" | "r13w" | "r13b" | "r13l") => Some("r13"),
            AsmLocation::Reg("r14" | "r14d" | "r14w" | "r14b" | "r14l") => Some("r14"),
            AsmLocation::Reg("r15" | "r15d" | "r15w" | "r15b" | "r15l") => Some("r15"),
            _ => None,
        }
    }

    fn resolve_pseudoregister(&mut self, frame: &mut FuncStackFrame) -> Result<(), String> {
        if let AsmLocation::PseudoReg(psr) = self {
            *self = frame.create_or_get_location(&psr)?;
        }

        Ok(())
    }

    fn convert_to_rsp_offset(&mut self, frame: &FuncStackFrame) {
        if let AsmLocation::RbpOffset(rbp_offset, name) = self {
            let rsp_offset = frame.size() as i32 + *rbp_offset;
            assert!(rsp_offset >= 0);
            *self = AsmLocation::RspOffset(rsp_offset as u32, name.clone());
        }
    }

    fn emit_code(&self, f: &mut fmt::Formatter, size: u8) -> fmt::Result {
        match self {
            AsmLocation::Reg(name) => {
                f.write_str(match (self.get_base_reg_name().unwrap(), size) {
                    ("ax", 1) => "al",
                    ("ax", 4) => "eax",
                    ("bx", 1) => "bl",
                    ("bx", 4) => "ebx",
                    ("cx", 1) => "cl",
                    ("cx", 4) => "ecx",
                    ("dx", 1) => "dl",
                    ("dx", 4) => "edx",
                    ("si", 1) => "sil",
                    ("si", 4) => "esi",
                    ("di", 1) => "dil",
                    ("di", 4) => "edi",
                    ("bp", 1) => "bpl",
                    ("bp", 4) => "ebp",
                    ("sp", 1) => "spl",
                    ("sp", 4) => "esp",
                    ("r8", 1) => "r8b",
                    ("r8", 4) => "r8d",
                    ("r9", 1) => "r9b",
                    ("r9", 4) => "r9d",
                    ("r10", 1) => "r10b",
                    ("r10", 4) => "r10d",
                    ("r11", 1) => "r11b",
                    ("r11", 4) => "r11d",
                    ("r12", 1) => "r12b",
                    ("r12", 4) => "r12d",
                    ("r13", 1) => "r13b",
                    ("r13", 4) => "r13d",
                    ("r14", 1) => "r14b",
                    ("r14", 4) => "r14d",
                    ("r15", 1) => "r15b",
                    ("r15", 4) => "r15d",
                    _ => panic!("unknown reg name and size {}, {}", name, size),
                })?;
            }
            AsmLocation::RspOffset(rsp_offset, _name) => {
                write!(
                    f,
                    "{} PTR [rsp+{}]",
                    match size {
                        1 => "BYTE",
                        2 => "WORD",
                        4 => "DWORD",
                        8 => "QWORD",
                        _ => panic!("unhandled pointer size {}", size),
                    },
                    rsp_offset
                )?;
            }
            _ => panic!("cannot handle emitting {:?}", self),
        }

        Ok(())
    }

    fn fmt_asm_comment(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmLocation::Reg(name) => {
                f.write_str(name)?;
            }
            AsmLocation::RspOffset(_rsp_offset, name) => {
                f.write_str(name)?;
            }
            _ => panic!("{:?} should not be written into ASM", self),
        }

        Ok(())
    }
}

impl FmtNode for AsmLocation {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        match self {
            AsmLocation::Reg(name) => {
                write!(f, "Reg {}", name)?;
            }
            AsmLocation::PseudoReg(name) => {
                f.write_str(&name)?;
            }
            AsmLocation::RbpOffset(offset, name) => {
                write!(
                    f,
                    "rbp{}{} ({})",
                    if *offset >= 0 { "+" } else { "" },
                    offset,
                    name
                )?;
            }
            AsmLocation::RspOffset(offset, name) => {
                write!(f, "rsp+{} ({})", offset, name)?;
            }
        }

        Ok(())
    }
}

impl AsmUnaryOperator {
    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmUnaryOperator::Neg => write!(f, "neg"),
            AsmUnaryOperator::Not => write!(f, "not"),
        }
    }
}

impl FmtNode for AsmUnaryOperator {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl AsmBinaryOperator {
    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            AsmBinaryOperator::Add => "add",
            AsmBinaryOperator::Subtract => "sub",
            AsmBinaryOperator::Imul => "imul",
            AsmBinaryOperator::And => "and",
            AsmBinaryOperator::Or => "or",
            AsmBinaryOperator::Xor => "xor",
            AsmBinaryOperator::Shl => "shl",
            AsmBinaryOperator::Sar => "sar",
        })
    }
}

impl FmtNode for AsmBinaryOperator {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        f.write_str(match self {
            AsmBinaryOperator::Add => "+",
            AsmBinaryOperator::Subtract => "-",
            AsmBinaryOperator::Imul => "*",
            AsmBinaryOperator::And => "&",
            AsmBinaryOperator::Or => "|",
            AsmBinaryOperator::Xor => "^",
            AsmBinaryOperator::Shl => "<<",
            AsmBinaryOperator::Sar => ">>",
        })
    }
}

impl AsmCondCode {
    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            AsmCondCode::E => "e",
            AsmCondCode::NE => "ne",
            AsmCondCode::L => "l",
            AsmCondCode::LE => "le",
            AsmCondCode::G => "g",
            AsmCondCode::GE => "ge",
        })
    }
}

impl fmt::Display for AsmCondCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            AsmCondCode::E => "E",
            AsmCondCode::NE => "NE",
            AsmCondCode::L => "L",
            AsmCondCode::LE => "LE",
            AsmCondCode::G => "G",
            AsmCondCode::GE => "GE",
        })
    }
}

impl GlobalTracking {
    fn new() -> Self {
        Self {
            next_temporary_id: 0,
            next_label_id: 0,
        }
    }

    fn allocate_temporary(&mut self) -> TacVar {
        self.next_temporary_id += 1;
        TacVar(format!("{:03}_tmp", self.next_temporary_id - 1))
    }

    fn create_temporary_ast_ident(&mut self, identifier: &AstIdentifier) -> AstIdentifier {
        self.next_temporary_id += 1;
        AstIdentifier(format!(
            "{:03}_var_{}",
            self.next_temporary_id - 1,
            identifier.0
        ))
    }

    fn allocate_label(&mut self, prefix: &str) -> String {
        self.next_label_id += 1;
        format!("{}_{:03}", prefix, self.next_label_id - 1)
    }
}

impl FunctionTracking {
    fn new() -> Self {
        Self {
            labels: HashMap::new(),
        }
    }

    fn add_goto_label(
        &mut self,
        global_tracking: &mut GlobalTracking,
        label: &AstLabel,
    ) -> Result<AstLabel, String> {
        if self.labels.contains_key(&label) {
            return Err(format!("duplicate label declaration \"{}\"", &label.0));
        }

        let temp_label =
            AstLabel(global_tracking.allocate_label(&format!("0userlabel_{}", &label.0)));
        let old_value = self.labels.insert(label.clone(), temp_label.clone());
        assert!(old_value.is_none());

        Ok(temp_label)
    }

    fn resolve_label(&self, label: &AstLabel) -> Result<AstLabel, String> {
        // If the label was found in this scope, return its mangled version.
        if let Some(temp_label) = self.labels.get(label) {
            Ok(temp_label.clone())
        } else {
            Err(format!("label '{}' not found", &label.0))
        }
    }
}

impl<'p> BlockTracking<'p> {
    fn new(parent_opt: Option<&'p BlockTracking>) -> Self {
        Self {
            parent_opt,
            variables: HashMap::new(),
        }
    }

    /// Adds a variable to the map and returns a unique, mangled identifier to refer to the variable with.
    fn add_variable(
        &mut self,
        global_tracking: &mut GlobalTracking,
        identifier: &AstIdentifier,
    ) -> Result<AstIdentifier, String> {
        if self.variables.contains_key(&identifier) {
            return Err(format!("duplicate variable declaration {}", &identifier.0));
        }

        let temp_var = global_tracking.create_temporary_ast_ident(&identifier);
        let old_value = self.variables.insert(identifier.clone(), temp_var.clone());
        assert!(old_value.is_none());

        Ok(temp_var)
    }

    fn resolve_variable(&self, identifier: &AstIdentifier) -> Result<&AstIdentifier, String> {
        // If the variable was found in this scope, return its mangled version.
        if let Some(temp_ident) = self.variables.get(identifier) {
            Ok(temp_ident)
        } else if let Some(parent) = self.parent_opt {
            // If this block is contained within another block, search that one for a variable instead.
            parent.resolve_variable(identifier)
        } else {
            Err(format!("variable '{}' not found", &identifier.0))
        }
    }
}

impl BreakTracking {
    fn new(
        parent_opt: Option<Box<BreakTracking>>,
        break_label: AstLabel,
        continue_label_opt: Option<AstLabel>,
    ) -> Self {
        Self {
            parent_opt,
            break_label,
            continue_label_opt,
        }
    }
}

impl FuncStackFrame {
    fn new() -> Self {
        Self {
            names: HashMap::new(),
            max_base_offset: 0,
        }
    }

    fn create_or_get_location(&mut self, name: &str) -> Result<AsmLocation, String> {
        if let Some(offset) = self.names.get(name) {
            Ok(AsmLocation::RbpOffset(*offset, String::from(name)))
        } else {
            // For now we are only storing 4-byte values.
            self.max_base_offset += 4;
            let offset = -(self.max_base_offset as i32);

            assert!(self.names.insert(String::from(name), offset).is_none());

            Ok(AsmLocation::RbpOffset(offset, String::from(name)))
        }
    }

    fn size(&self) -> u32 {
        self.max_base_offset
    }
}

fn lex_next_token<'i>(input: &'i str) -> Result<(&'i str, &'i str), String> {
    lazy_static! {
        static ref SKIPPED_TOKEN_REGEXES: Vec<regex::Regex> = vec![
            Regex::new(r"^\#[^\n]*").expect("failed to compile regex"),
            Regex::new(r"^\/\/[^\n]*").expect("failed to compile regex"),
        ];
        static ref TOKEN_REGEXES: Vec<regex::Regex> = vec![
            Regex::new(r"^(?:>>=|<<=|&&|\|\||==|!=|<=|>=|--|\+\+|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\{|\}|\(|\)|;|-|~|!|\+|/|\*|%|&|\||\^|<|>|=|\?|:|,|[a-zA-Z_]\w*\b|[0-9]+\b)").expect("failed to compile regex"),
        ];
    }

    for r in SKIPPED_TOKEN_REGEXES.iter() {
        if let Some(mat) = r.find(input) {
            let range = mat.range();
            //println!("match skipped {:?}: {}, {}", r, range.start, range.end);
            return Ok(("", input.split_at(range.end).1));
        }
    }

    for r in TOKEN_REGEXES.iter() {
        if let Some(mat) = r.find(input) {
            let range = mat.range();
            //println!("match {:?}: {}, {}", r, range.start, range.end);
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

    let body = parse_block(&mut tokens)?;

    *original_tokens = tokens;
    Ok(AstFunction {
        name,
        parameters,
        body,
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

    let var_name = tokens.consume_identifier()?;
    *original_tokens = tokens;
    Ok(var_name)
}

fn parse_block<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstBlock, String> {
    let mut tokens = original_tokens.clone();

    tokens.consume_expected_next_token("{")?;

    let mut body = vec![];

    loop {
        if let Ok(operator) = tokens.consume_expected_next_token("}") {
            break;
        }

        body.push(parse_block_item(&mut tokens)?);
    }

    *original_tokens = tokens;
    Ok(AstBlock(body))
}

fn parse_block_item<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstBlockItem, String> {
    let mut tokens = original_tokens.clone();

    let ret = if let Ok(declaration) = parse_declaration(&mut tokens) {
        Ok(AstBlockItem::Declaration(declaration))
    } else if let Ok(statement) = parse_statement(&mut tokens) {
        Ok(AstBlockItem::Statement(statement))
    } else {
        Err(String::from("failed to parse block item"))
    };

    if ret.is_ok() {
        *original_tokens = tokens;
    }

    ret
}

fn parse_statement<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstStatement, String> {
    let mut tokens = original_tokens.clone();

    let labels = parse_statement_labels(&mut tokens)?;

    let statement_type = if tokens.consume_expected_next_token("return").is_ok() {
        let st = AstStatementType::Return(parse_expression(&mut tokens)?);
        tokens.consume_expected_next_token(";")?;
        st
    } else if tokens.consume_expected_next_token(";").is_ok() {
        AstStatementType::Null
    } else if tokens.consume_expected_next_token("if").is_ok() {
        tokens.consume_expected_next_token("(")?;
        let condition_expr = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;
        let then_statement = parse_statement(&mut tokens)?;

        let else_statement_opt = if tokens.consume_expected_next_token("else").is_ok() {
            Some(Box::new(parse_statement(&mut tokens)?))
        } else {
            None
        };

        AstStatementType::If(condition_expr, Box::new(then_statement), else_statement_opt)
    } else if tokens.consume_expected_next_token("goto").is_ok() {
        let label_name = tokens.consume_identifier()?;
        tokens.consume_expected_next_token(";")?;
        AstStatementType::Goto(AstLabel(String::from(label_name)))
    } else if let Ok(for_statement) = parse_for_statement(&mut tokens) {
        for_statement
    } else if let Ok(block) = parse_block(&mut tokens) {
        AstStatementType::Compound(Box::new(block))
    } else if tokens.consume_expected_next_token("break").is_ok() {
        let st = AstStatementType::Break(None);
        tokens.consume_expected_next_token(";")?;
        st
    } else if tokens.consume_expected_next_token("continue").is_ok() {
        let st = AstStatementType::Continue(None);
        tokens.consume_expected_next_token(";")?;
        st
    } else {
        let st = AstStatementType::Expr(parse_expression(&mut tokens)?);
        tokens.consume_expected_next_token(";")?;
        st
    };

    *original_tokens = tokens;
    Ok(AstStatement::new(statement_type, labels))
}

fn parse_for_statement<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<AstStatementType, String> {
    let mut tokens = original_tokens.clone();

    tokens.consume_expected_next_token("for")?;
    tokens.consume_expected_next_token("(")?;

    let initializer = if let Ok(decl) = parse_declaration(&mut tokens) {
        AstForInit::Declaration(decl)
    } else if let Ok(expr_opt) = parse_optional_expression(&mut tokens, ";") {
        AstForInit::Expression(expr_opt)
    } else {
        return Err(format!("failed to parse for loop initializer"));
    };

    let condition_opt = parse_optional_expression(&mut tokens, ";")?;

    let final_expr_opt = parse_optional_expression(&mut tokens, ")")?;

    let body = parse_statement(&mut tokens)?;

    *original_tokens = tokens;
    Ok(AstStatementType::For(
        initializer,
        condition_opt,
        final_expr_opt,
        Box::new(body),
        None,
        None,
    ))
}

fn parse_statement_labels<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<Vec<AstLabel>, String> {
    fn parse_label<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstLabel, String> {
        let mut tokens = original_tokens.clone();

        // Label is a name followed by :
        let label_name = tokens.consume_identifier()?;
        tokens.consume_expected_next_token(":")?;

        *original_tokens = tokens;
        Ok(AstLabel(String::from(label_name)))
    }

    let mut tokens = original_tokens.clone();

    let mut labels = vec![];

    loop {
        if let Ok(label) = parse_label(&mut tokens) {
            labels.push(label);
        } else {
            *original_tokens = tokens;
            return Ok(labels);
        }
    }
}

fn parse_declaration<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<AstDeclaration, String> {
    let mut tokens = original_tokens.clone();

    tokens.consume_expected_next_token("int")?;

    let var_name = tokens.consume_identifier()?;

    // Optional initializer is present.
    let initializer_opt = if let Ok(_) = tokens.consume_expected_next_token("=") {
        Some(parse_expression(&mut tokens)?)
    } else {
        None
    };

    tokens.consume_expected_next_token(";")?;

    *original_tokens = tokens;

    Ok(AstDeclaration {
        identifier: AstIdentifier(String::from(var_name)),
        initializer_opt,
    })
}

fn parse_expression<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    fn parse_expression_with_precedence<'i, 't>(
        original_tokens: &mut Tokens<'i, 't>,
        min_precedence_allowed: u8,
    ) -> Result<AstExpression, String> {
        // Always try for at least one factor in an expression.
        let mut left = parse_factor(original_tokens)?;

        // The tokens we clone here might be committed, or we might not find a valid expression within.
        let mut tokens = original_tokens.clone();

        loop {
            // Attempt to parse a binary operator expression.
            if let Ok(operator) = tokens.consume_and_parse_next_token::<AstBinaryOperator>() {
                // Only allow parsing a binary operator with same or higher precedence, or else it messes up the
                // precedence ordering. This is called precedence climbing.
                if operator.precedence() >= min_precedence_allowed {
                    match operator {
                        AstBinaryOperator::Assign
                        | AstBinaryOperator::AddAssign
                        | AstBinaryOperator::SubtractAssign
                        | AstBinaryOperator::MultiplyAssign
                        | AstBinaryOperator::DivideAssign
                        | AstBinaryOperator::ModulusAssign
                        | AstBinaryOperator::BitwiseAndAssign
                        | AstBinaryOperator::BitwiseOrAssign
                        | AstBinaryOperator::BitwiseXorAssign
                        | AstBinaryOperator::ShiftLeftAssign
                        | AstBinaryOperator::ShiftRightAssign => {
                            // Assignment is right-associative, not left-associative, so parse with the precedence of the
                            // operator so that further tokens of the same precedence would also go to the right side, not
                            // left side.
                            let right = parse_expression_with_precedence(
                                &mut tokens,
                                operator.precedence(),
                            )?;

                            // This assignment now becomes the left hand side of the expression.
                            left = AstExpression::Assignment(
                                Box::new(left),
                                operator,
                                Box::new(right),
                            );
                        }
                        AstBinaryOperator::Conditional => {
                            // The middle expression is restarted with 0 precedence because it's cleanly delineated by
                            // the : that follows.
                            let middle = parse_expression_with_precedence(&mut tokens, 0)?;
                            tokens.consume_expected_next_token(":")?;

                            // Conditional operator is right-associative, so parse with current precedence.
                            let right = parse_expression_with_precedence(
                                &mut tokens,
                                operator.precedence(),
                            )?;

                            left = AstExpression::Conditional(
                                Box::new(left),
                                Box::new(middle),
                                Box::new(right),
                            );
                        }
                        _ => {
                            // If the right hand side is itself going to encounter a binary expression, it can only be a
                            // strictly higher precedence, or else it shouldn't be part of the right-hand-side expression.
                            let right = parse_expression_with_precedence(
                                &mut tokens,
                                operator.precedence() + 1,
                            )?;

                            // This binary operation now becomes the left hand side of the expression.
                            left = AstExpression::BinaryOperator(
                                Box::new(left),
                                operator,
                                Box::new(right),
                            );
                        }
                    }

                    // Since we successfully consumed an expression, commit the tokens we consumed for this.
                    *original_tokens = tokens.clone();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(left)
    }

    parse_expression_with_precedence(original_tokens, 0)
}

fn parse_optional_expression<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
    terminator: &str,
) -> Result<Option<AstExpression>, String> {
    let mut tokens = original_tokens.clone();

    // Try to parse an expression and then the desired terminator.
    let ret = if let Ok(expr) = parse_expression(&mut tokens) {
        tokens.consume_expected_next_token(terminator)?;
        Ok(Some(expr))
    } else if tokens.consume_expected_next_token(terminator).is_ok() {
        // Well, the expression wasn't found. See if just the terminator is, in which case the optional expression is
        // valid, just with no expression found.
        Ok(None)
    } else {
        // If the terminator isn't found, then it's not valid.
        return Err(format!(
            "failed to parse optional expression starting at {:?}",
            tokens
        ));
    };

    if ret.is_ok() {
        *original_tokens = tokens;
    }

    ret
}

fn parse_factor<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    let mut ret = if let Ok(integer_literal) = tokens.consume_and_parse_next_token::<u32>() {
        Ok(AstExpression::Constant(integer_literal))
    } else if let Ok(_) = tokens.consume_expected_next_token("(") {
        let inner = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;
        Ok(inner)
    } else if let Ok(operator) = tokens.consume_and_parse_next_token::<AstUnaryOperator>() {
        let inner = parse_factor(&mut tokens)?;
        Ok(AstExpression::UnaryOperator(operator, Box::new(inner)))
    } else if let Ok(var_name) = tokens.consume_identifier() {
        Ok(AstExpression::Var(AstIdentifier(String::from(var_name))))
    } else {
        Err(String::from("unknown factor"))
    };

    // Immediately after parsing a factor, attempt to parse a postfix operator, because it's higher precedence than
    // anything else.
    if let Ok(factor) = ret {
        if let Some(suffix_operator) = parse_postfix_operator(&mut tokens) {
            ret = Ok(AstExpression::UnaryOperator(
                suffix_operator,
                Box::new(factor),
            ));
        } else {
            ret = Ok(factor);
        }
    }

    if ret.is_ok() {
        *original_tokens = tokens;
    }

    ret
}

fn parse_postfix_operator<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Option<AstUnaryOperator> {
    let mut tokens = original_tokens.clone();

    let ret = if let Ok(operator) = tokens.consume_and_parse_next_token::<AstUnaryOperator>() {
        match operator {
            AstUnaryOperator::PrefixIncrement => Some(AstUnaryOperator::PostfixIncrement),
            AstUnaryOperator::PrefixDecrement => Some(AstUnaryOperator::PostfixDecrement),
            _ => None,
        }
    } else {
        None
    };

    if ret.is_some() {
        *original_tokens = tokens;
    }

    ret
}

fn generate_program_code(mode: Mode, ast_program: &mut AstProgram) -> Result<String, String> {
    let tac_program = ast_program.to_tac()?;

    println!(
        "tac:\n{}\n",
        display_with(|f| { tac_program.fmt_node(f, 0) })
    );

    match mode {
        Mode::All | Mode::CodegenOnly => {
            let mut asm_program = tac_program.to_asm()?;

            println!("asm:\n{}", display_with(|f| asm_program.fmt_node(f, 0)));

            asm_program.finalize()?;

            println!(
                "\nasm after fixup:\n{}",
                display_with(|f| asm_program.fmt_node(f, 0))
            );

            asm_program.emit_code()
        }
        _ => Ok(String::new()),
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

// TODO: should have proper error handling in here
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
    let token_strings = lex_all_tokens(&input)?;
    let mut tokens = Tokens(&token_strings);
    /*
    for token in tokens.iter() {
        println!("{}", token);
    }

    println!();
    */

    if let Mode::LexOnly = mode {
        return Ok(AstProgram::new(vec![]));
    }

    // TODO all parsing should return a list of errors, not just one. for now, wrap it in a single error
    let mut ast = {
        let mut functions = vec![];
        while let Ok(function) = parse_function(&mut tokens) {
            functions.push(function);
        }

        AstProgram::new(functions)
    };

    println!("AST:\n{}\n", display_with(|f| ast.fmt_node(f, 0)));

    if tokens.0.len() != 0 {
        return Err(vec![format!(
            "extra tokens after main function end: {:?}",
            tokens
        )]);
    }

    if let Mode::ParseOnly = mode {
        return Ok(ast);
    }

    ast.validate_and_resolve()?;

    println!(
        "AST after resolve:\n{}\n",
        display_with(|f| ast.fmt_node(f, 0))
    );

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
            Ok(ref mut ast) => match args.mode {
                Mode::All | Mode::TacOnly | Mode::CodegenOnly => {
                    let asm = generate_program_code(args.mode, ast)?;

                    if let Mode::All = args.mode {
                        println!("\nassembly:\n{}", asm);

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

    /// Lex, parse, and validate only. Exit code is zero if successful.
    #[value(name = "validate", alias = "v")]
    ValidateOnly,

    /// Lex, parse, validate, and generate TAC only. Exit code is zero if successful.
    #[value(name = "tac", alias = "t")]
    TacOnly,

    /// Lex, parse, validate, generate TAC, and codegen only. Exit code is zero if successful.
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
    let args = LcArgs::parse();

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

    mod lex {
        use super::*;

        #[test]
        fn simple() {
            let input = r"int main() {
        return 2;
    }";
            assert_eq!(
                lex_all_tokens(&input),
                Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"])
            );
        }

        #[test]
        fn no_whitespace() {
            let input = r"int main(){return 2;}";
            assert_eq!(
                lex_all_tokens(&input),
                Ok(vec!["int", "main", "(", ")", "{", "return", "2", ";", "}"])
            );
        }

        #[test]
        fn negative() {
            assert_eq!(
                lex_all_tokens("int main() { return -1; }"),
                Ok(vec![
                    "int", "main", "(", ")", "{", "return", "-", "1", ";", "}"
                ])
            );
        }

        #[test]
        fn bitwise_not() {
            assert_eq!(
                lex_all_tokens("int main() { return ~1; }"),
                Ok(vec![
                    "int", "main", "(", ")", "{", "return", "~", "1", ";", "}"
                ])
            );
        }

        #[test]
        fn logical_not() {
            assert_eq!(
                lex_all_tokens("int main() { return !1; }"),
                Ok(vec![
                    "int", "main", "(", ")", "{", "return", "!", "1", ";", "}"
                ])
            );
        }

        #[test]
        fn no_at() {
            assert!(lex_all_tokens("int main() { return 0@1; }").is_err());
        }

        #[test]
        fn no_backslash() {
            assert!(lex_all_tokens("\\").is_err());
        }

        #[test]
        fn no_backtick() {
            assert!(lex_all_tokens("`").is_err());
        }

        #[test]
        fn bad_identifier() {
            assert!(lex_all_tokens("int main() { return 1foo; }").is_err());
        }

        #[test]
        fn no_at_identifier() {
            assert!(lex_all_tokens("int main() { return @b; }").is_err());
        }
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

    fn validate_error_count(input: &str, expected_error_count: usize) {
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
        codegen_run_and_check_exit_code(
            &format!("int main() {{\n{}\n}}", body),
            expected_exit_code,
        );
    }

    fn test_codegen_mainfunc_failure(body: &str) {
        codegen_run_and_expect_compile_failure(&format!("int main() {{ {} }}", body))
    }

    mod fail {
        use super::*;

        #[test]
        fn parse_extra_paren() {
            test_codegen_mainfunc_failure("return (3));");
        }

        #[test]
        fn parse_unclosed_paren() {
            test_codegen_mainfunc_failure("return (3;");
        }

        #[test]
        fn parse_missing_immediate() {
            test_codegen_mainfunc_failure("return ~;");
        }

        #[test]
        fn parse_missing_immediate_2() {
            test_codegen_mainfunc_failure("return -~;");
        }

        #[test]
        fn parse_missing_semicolon() {
            test_codegen_mainfunc_failure("return 5");
        }

        #[test]
        fn parse_missing_semicolon_binary_op() {
            test_codegen_mainfunc_failure("return 5 + 6");
        }

        #[test]
        fn parse_parens_around_operator() {
            test_codegen_mainfunc_failure("return (-)5;");
        }

        #[test]
        fn parse_operator_wrong_order() {
            test_codegen_mainfunc_failure("return 5-;");
        }

        #[test]
        fn parse_double_operator() {
            test_codegen_mainfunc_failure("return 1 * / 2;");
        }

        #[test]
        fn parse_unbalanced_paren() {
            test_codegen_mainfunc_failure("return 1 + (2;");
        }

        #[test]
        fn parse_missing_opening_paren() {
            test_codegen_mainfunc_failure("return 1 + 2);");
        }

        #[test]
        fn parse_unexpected_paren() {
            test_codegen_mainfunc_failure("return 1 (- 2);");
        }

        #[test]
        fn parse_misplaced_semicolon_paren() {
            test_codegen_mainfunc_failure("return 1 + (2;)");
        }

        #[test]
        fn parse_missing_first_binary_operand() {
            test_codegen_mainfunc_failure("return / 2;");
        }

        #[test]
        fn parse_missing_second_binary_operand() {
            test_codegen_mainfunc_failure("return 2 / ;");
        }

        #[test]
        fn parse_relational_missing_first_operand() {
            test_codegen_mainfunc_failure("return <= 2;");
        }

        #[test]
        fn parse_relational_missing_second_operand() {
            test_codegen_mainfunc_failure("return 1 < > 3;");
        }

        #[test]
        fn parse_and_missing_second_operand() {
            test_codegen_mainfunc_failure("return 2 && ~;");
        }

        #[test]
        fn parse_or_missing_semicolon() {
            test_codegen_mainfunc_failure("return 2 || 4");
        }

        #[test]
        fn parse_unary_not_missing_semicolon() {
            test_codegen_mainfunc_failure("return !10");
        }

        #[test]
        fn parse_double_bitwise_or() {
            test_codegen_mainfunc_failure("return 1 | | 2;");
        }

        #[test]
        fn parse_unary_not_missing_operand() {
            test_codegen_mainfunc_failure("return 10 <= !;");
        }

        #[test]
        fn duplicate_variable() {
            test_codegen_mainfunc_failure("int x = 5; int x = 4; return x;");
        }

        #[test]
        fn duplicate_variable_after_use() {
            test_codegen_mainfunc_failure("int x = 5; return x; int x = 4; return x;");
        }

        #[test]
        fn unknown_variable() {
            test_codegen_mainfunc_failure("return x;");
        }

        #[test]
        fn unknown_variable_after_shortcircuit() {
            test_codegen_mainfunc_failure("return 0 && x;");
        }

        #[test]
        fn unknown_variable_in_binary_op() {
            test_codegen_mainfunc_failure("return x < 5;");
        }

        #[test]
        fn unknown_variable_in_bitwise_op() {
            test_codegen_mainfunc_failure("return a >> 2;");
        }

        #[test]
        fn unknown_variable_in_unary_op() {
            test_codegen_mainfunc_failure("return -x;");
        }

        #[test]
        fn unknown_variable_lhs_compound_assignment() {
            test_codegen_mainfunc_failure("a += 1; return 0;");
        }

        #[test]
        fn unknown_variable_rhs_compound_assignment() {
            test_codegen_mainfunc_failure("int b = 10; b *= a; return 0;");
        }

        #[test]
        fn malformed_plusequals() {
            test_codegen_mainfunc_failure("int a = 0; a + = 1; return a;");
        }

        #[test]
        fn malformed_decrement() {
            test_codegen_mainfunc_failure("int a = 5; a - -; return a;");
        }

        #[test]
        fn malformed_increment() {
            test_codegen_mainfunc_failure("int a = 5; a + +; return a;");
        }

        #[test]
        fn malformed_less_equals() {
            test_codegen_mainfunc_failure("return 1 < = 2;");
        }

        #[test]
        fn malformed_not_equals() {
            test_codegen_mainfunc_failure("return 1 ! = 2;");
        }

        #[test]
        fn malformed_divide_equals() {
            test_codegen_mainfunc_failure("int a = 10; a =/ 5; return a;");
        }

        #[test]
        fn missing_semicolon() {
            test_codegen_mainfunc_failure("int a = 5 a = a + 5; return a;");
        }

        #[test]
        fn return_in_assignment() {
            test_codegen_mainfunc_failure("int a = return 5;");
        }

        #[test]
        fn declare_keyword_as_var() {
            test_codegen_mainfunc_failure("int return = 6; return return + 1;");
        }

        #[test]
        fn declare_after_use() {
            test_codegen_mainfunc_failure("a = 5; int a; return a;");
        }

        #[test]
        fn invalid_lvalue_binary_op() {
            test_codegen_mainfunc_failure("int a = 5; a + 3 = 4; return a;");
        }

        #[test]
        fn invalid_lvalue_unary_op() {
            test_codegen_mainfunc_failure("int a = 5; !a = 4; return a;");
        }

        #[test]
        fn declare_invalid_var_name_with_space() {
            test_codegen_mainfunc_failure("int x y = 3; return y;");
        }

        #[test]
        fn declare_invalid_var_name_starting_number() {
            test_codegen_mainfunc_failure("int 10 = 3; return 10;");
            test_codegen_mainfunc_failure("int 10a = 3; return 10a;");
        }

        #[test]
        fn declare_invalid_type_name() {
            test_codegen_mainfunc_failure("ints x = 3; return x;");
        }

        #[test]
        fn invalid_mixed_precedence_assignment() {
            test_codegen_mainfunc_failure("int a = 1; int b = 2; a = 3 * b = a; return a;");
        }

        #[test]
        fn compound_initializer() {
            test_codegen_mainfunc_failure("int a += 0; return a;");
        }

        #[test]
        fn invalid_unary_lvalue() {
            test_codegen_mainfunc_failure("int a = 0; -a += 1; return a;");
        }

        #[test]
        fn invalid_compound_lvalue() {
            test_codegen_mainfunc_failure("int a = 10; (a += 1) -= 2;");
        }

        #[test]
        fn decrement_binary_op() {
            test_codegen_mainfunc_failure("int a = 0; return a -- 1;");
        }

        #[test]
        fn increment_binary_op() {
            test_codegen_mainfunc_failure("int a = 0; return a ++ 1;");
        }

        #[test]
        fn increment_declaration() {
            test_codegen_mainfunc_failure("int a++; return 0;");
        }

        #[test]
        fn double_postfix() {
            test_codegen_mainfunc_failure("int a = 10; return a++--;");
        }

        #[test]
        fn postfix_incr_non_lvalue() {
            test_codegen_mainfunc_failure("int a = 0; (a = 4)++;");
        }

        #[test]
        fn prefix_incr_non_lvalue() {
            test_codegen_mainfunc_failure("int a = 1; ++(a+1); return 0;");
        }

        #[test]
        fn prefix_decr_constant() {
            test_codegen_mainfunc_failure("return --3;");
        }

        #[test]
        fn postfix_undeclared_var() {
            test_codegen_mainfunc_failure("a--; return 0;");
        }

        #[test]
        fn prefix_undeclared_var() {
            test_codegen_mainfunc_failure("++a; return 0;");
        }

        #[test]
        fn declaration_as_statement() {
            test_codegen_mainfunc_failure("if (5) int i = 0;");
        }

        #[test]
        fn empty_if_body() {
            test_codegen_mainfunc_failure("if (0) else return 0;");
        }

        #[test]
        fn if_as_assignment() {
            test_codegen_mainfunc_failure("int flag = 0; int a = if (flag) 2; else 3; return a;");
        }

        #[test]
        fn if_no_parentheses() {
            test_codegen_mainfunc_failure("if 0 return 1;");
        }

        #[test]
        fn extra_else() {
            test_codegen_mainfunc_failure("if (1) return 1; else return 2; else return 3;");
        }

        #[test]
        fn undeclared_var_in_if() {
            test_codegen_mainfunc_failure("if (1) return c; int c = 0;");
        }

        #[test]
        fn incomplete_ternary() {
            test_codegen_mainfunc_failure("return 1 ? 2;");
        }

        #[test]
        fn ternary_extra_left() {
            test_codegen_mainfunc_failure("return 1 ? 2 ? 3 : 4;");
        }

        #[test]
        fn ternary_extra_right() {
            test_codegen_mainfunc_failure("return 1 ? 2 : 3 : 4;");
        }

        #[test]
        fn ternary_wrong_delimiter() {
            test_codegen_mainfunc_failure("int x = 10; return x ? 1 = 2;");
        }

        #[test]
        fn ternary_undeclared_var() {
            test_codegen_mainfunc_failure("return a > 0 ? 1 : 2; int a = 5;");
        }

        #[test]
        fn ternary_invalid_assign() {
            test_codegen_mainfunc_failure(
                r"
            int a = 2;
            int b = 1;
            a > b ? a = 1 : a = 0;
            return a;
            ",
            );
        }

        #[test]
        fn keywords_as_var_identifier() {
            test_codegen_mainfunc_failure("int if = 0; return if;");
            test_codegen_mainfunc_failure("int int = 0; return int;");
            test_codegen_mainfunc_failure("int void = 0; return void;");
            test_codegen_mainfunc_failure("int return = 0; return return;");
        }

        #[test]
        fn extra_closing_brace() {
            test_codegen_mainfunc_failure("if(0){ return 1; }} return 2;");
        }

        #[test]
        fn missing_closing_brace() {
            test_codegen_mainfunc_failure("if(0){ return 1; return 2;");
        }

        #[test]
        fn missing_semicolon_in_block() {
            test_codegen_mainfunc_failure("int a = 4; { a = 5; return a }");
        }

        #[test]
        fn block_in_ternary() {
            test_codegen_mainfunc_failure("int a; return 1 ? { a = 2 } : a = 4;");
        }

        #[test]
        fn duplicate_var_declaration() {
            test_codegen_mainfunc_failure("{ int a; int a; }");
        }

        #[test]
        fn use_var_after_scope() {
            test_codegen_mainfunc_failure("{ int a = 2; } return a;");
        }

        #[test]
        fn use_var_before_declare() {
            test_codegen_mainfunc_failure("int a; { b = 10; } int b; return b;");
        }

        #[test]
        fn duplicate_var_declaration_after_block() {
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
        }

        #[test]
        fn break_outside_loop() {
            codegen_run_and_expect_compile_failure(r"if (1) break;");
        }

        #[test]
        fn continue_outside_loop() {
            codegen_run_and_expect_compile_failure(
                r"
    {
        int a;
        continue;
    }
    return 0;
            ",
            );
        }
    }

    #[test]
    fn unary_neg() {
        test_codegen_expression("-5", -5);
    }

    #[test]
    fn unary_bitnot() {
        test_codegen_expression("~12", -13);
    }

    #[test]
    fn unary_not() {
        test_codegen_expression("!5", 0);
        test_codegen_expression("!0", 1);
    }

    #[test]
    fn unary_neg_zero() {
        test_codegen_expression("-0", 0);
    }

    #[test]
    fn unary_bitnot_zero() {
        test_codegen_expression("~0", -1);
    }

    #[test]
    fn unary_neg_min_val() {
        test_codegen_expression("-2147483647", -2147483647);
    }

    #[test]
    fn unary_bitnot_and_neg() {
        test_codegen_expression("~-3", 2);
    }

    #[test]
    fn unary_bitnot_and_neg_zero() {
        test_codegen_expression("-~0", 1);
    }

    #[test]
    fn unary_bitnot_and_neg_min_val() {
        test_codegen_expression("~-2147483647", 2147483646);
    }

    #[test]
    fn unary_grouping_outside() {
        test_codegen_expression("(-2)", -2);
    }

    #[test]
    fn unary_grouping_inside() {
        test_codegen_expression("~(2)", -3);
    }

    #[test]
    fn unary_grouping_inside_and_outside() {
        test_codegen_expression("-(-4)", 4);
    }

    #[test]
    fn unary_grouping_several() {
        test_codegen_expression("-((((((10))))))", -10);
    }

    #[test]
    fn unary_not_and_neg() {
        test_codegen_expression("!-3", 0);
    }

    #[test]
    fn unary_not_and_arithmetic() {
        test_codegen_expression("!(4-4)", 1);
        test_codegen_expression("!(4 - 5)", 0);
    }

    #[test]
    fn expression_binary_operation() {
        test_codegen_expression("5 + 6", 11);
    }

    #[test]
    fn expression_negative_divide() {
        test_codegen_expression("-110 / 10", -11);
    }

    #[test]
    fn expression_negative_multiply() {
        test_codegen_expression("10 * -11", -110);
    }

    #[test]
    fn expression_factors_and_terms() {
        test_codegen_expression("(1 + 2 + 3 + 4) * (10 - 21)", -110);
    }

    #[test]
    fn and_false() {
        test_codegen_expression("(10 && 0) + (0 && 4) + (0 && 0)", 0);
    }

    #[test]
    fn and_true() {
        test_codegen_expression("1 && -1", 1);
    }

    #[test]
    fn and_shortcircuit() {
        test_codegen_expression("0 && (1 / 0)", 0);
    }

    #[test]
    fn or_shortcircuit() {
        test_codegen_expression("1 || (1 / 0)", 1);
    }

    #[test]
    fn multi_shortcircuit() {
        test_codegen_expression("0 || 0 && (1 / 0)", 0);
    }

    #[test]
    fn and_or_precedence() {
        test_codegen_expression("1 || 0 && 2", 1);
    }

    #[test]
    fn and_or_precedence_2() {
        test_codegen_expression("(1 || 0) && 0", 0);
    }

    #[test]
    fn relational_lt() {
        test_codegen_expression("1234 < 1234", 0);
        test_codegen_expression("1234 < 1235", 1);
    }

    #[test]
    fn relational_gt() {
        test_codegen_expression("1234 > 1234", 0);
        test_codegen_expression("1234 > 1233", 1);
        test_codegen_expression("(1 > 2) + (1 > 1)", 0);
    }

    #[test]
    fn relational_le() {
        test_codegen_expression("1234 <= 1234", 1);
        test_codegen_expression("1234 <= 1233", 0);
        test_codegen_expression("1 <= -1", 0);
        test_codegen_expression("(0 <= 2) + (0 <= 0)", 2);
    }

    #[test]
    fn relational_ge() {
        test_codegen_expression("1234 >= 1234", 1);
        test_codegen_expression("1234 >= 1235", 0);
        test_codegen_expression("(1 >= 1) + (1 >= -4)", 2);
    }

    #[test]
    fn equality_eq() {
        test_codegen_expression("1234 == 1234", 1);
        test_codegen_expression("1234 == 1235", 0);
    }

    #[test]
    fn equality_ne() {
        test_codegen_expression("1234 != 1234", 0);
        test_codegen_expression("1234 != 1235", 1);
    }

    #[test]
    fn logical_and() {
        test_codegen_expression("0 && 1 && 2", 0);
        test_codegen_expression("5 && 6 && 7", 1);
        test_codegen_expression("5 && 6 && 0", 0);
    }

    #[test]
    fn logical_or() {
        test_codegen_expression("0 || 0 || 1", 1);
        test_codegen_expression("1 || 0 || 0", 1);
        test_codegen_expression("0 || 0 || 0", 0);
    }

    #[test]
    fn equals_precedence() {
        test_codegen_expression("0 == 0 != 0", 1);
    }

    #[test]
    fn equals_relational_precedence() {
        test_codegen_expression("2 == 2 >= 0", 0);
    }

    #[test]
    fn equals_or_precedence() {
        test_codegen_expression("2 == 2 || 0", 1);
    }

    #[test]
    fn relational_associativity() {
        test_codegen_expression("5 >= 0 > 1 <= 0", 1);
    }

    #[test]
    fn compare_arithmetic_results() {
        test_codegen_expression("~2 * -2 == 1 + 5", 1);
    }

    #[test]
    fn all_operator_precedence() {
        test_codegen_expression("-1 * -2 + 3 >= 5 == 1 && (6 - 6) || 7", 1);
    }

    #[test]
    fn all_operator_precedence_2() {
        test_codegen_expression("(0 == 0 && 3 == 2 + 1 > 1) + 1", 1);
    }

    #[test]
    fn arithmetic_operator_precedence() {
        test_codegen_expression("1 * 2 + 3 * -4", -10);
    }

    #[test]
    fn arithmetic_operator_associativity_minus() {
        test_codegen_expression("5 - 2 - 1", 2);
    }

    #[test]
    fn arithmetic_operator_associativity_div() {
        test_codegen_expression("12 / 3 / 2", 2);
    }

    #[test]
    fn arithmetic_operator_associativity_grouping() {
        test_codegen_expression("(3 / 2 * 4) + (5 - 4 + 3)", 8);
    }

    #[test]
    fn arithmetic_operator_associativity_grouping_2() {
        test_codegen_expression("5 * 4 / 2 - 3 % (2 + 1)", 10);
    }

    #[test]
    fn sub_neg() {
        test_codegen_expression("2- -1", 3);
    }

    #[test]
    fn unop_add() {
        test_codegen_expression("~2 + 3", 0);
    }

    #[test]
    fn unop_parens() {
        test_codegen_expression("~(1 + 2)", -4);
    }

    #[test]
    fn modulus() {
        test_codegen_expression("10 % 3", 1);
    }

    #[test]
    fn bitand_associativity() {
        test_codegen_expression("7 * 1 & 3 * 1", 3);
    }

    #[test]
    fn or_xor_associativity() {
        test_codegen_expression("7 ^ 3 | 3 ^ 1", 6);
    }

    #[test]
    fn and_xor_associativity() {
        test_codegen_expression("7 ^ 3 & 6 ^ 2", 7);
    }

    #[test]
    fn shl_immediate() {
        test_codegen_expression("5 << 2", 20);
    }

    #[test]
    fn shl_tempvar() {
        test_codegen_expression("5 << (2 * 1)", 20);
    }

    #[test]
    fn sar_immediate() {
        test_codegen_expression("20 >> 2", 5);
    }

    #[test]
    fn sar_tempvar() {
        test_codegen_expression("20 >> (2 * 1)", 5);
    }

    #[test]
    fn shift_associativity() {
        test_codegen_expression("33 << 4 >> 2", 132);
    }

    #[test]
    fn shift_associativity_2() {
        test_codegen_expression("33 >> 2 << 1", 16);
    }

    #[test]
    fn shift_precedence() {
        test_codegen_expression("40 << 4 + 12 >> 1", 0x00140000);
    }

    #[test]
    fn sar_negative() {
        test_codegen_expression("-5 >> 1", -3);
    }

    #[test]
    fn bitwise_precedence() {
        test_codegen_expression("80 >> 2 | 1 ^ 5 & 7 << 1", 21);
    }

    #[test]
    fn arithmetic_and_booleans() {
        test_codegen_expression("~(0 && 1) - -(4 || 3)", 0);
    }

    #[test]
    fn bitand_equals_precedence() {
        test_codegen_expression("4 & 7 == 4", 0);
    }

    #[test]
    fn bitor_notequals_precedence() {
        test_codegen_expression("4 | 7 != 4", 5);
    }

    #[test]
    fn shift_relational_precedence() {
        test_codegen_expression("20 >> 4 <= 3 << 1", 1);
    }

    #[test]
    fn xor_relational_precedence() {
        test_codegen_expression("5 ^ 7 < 5", 5);
    }

    #[test]
    fn var_use() {
        test_codegen_mainfunc(
            "int _x = 5; int y = 6; int z; _x = 1; z = 3; return _x + y + z;",
            10,
        );
    }

    #[test]
    fn assign_expr() {
        test_codegen_mainfunc("int x = 5; int y = x = 3 + 1; return x + y;", 8);
    }

    #[test]
    fn declaration_after_expression() {
        test_codegen_mainfunc("int x; x = 5; int y = -x; return y;", -5);
    }

    #[test]
    fn mixed_precedence_assignment() {
        test_codegen_mainfunc("int x = 5; int y = 4; x = 3 * (y = x); return x + y;", 20);
    }

    #[test]
    fn assign_after_not_short_circuit_or() {
        test_codegen_mainfunc("int x = 0; 0 || (x = 1); return x;", 1);
    }

    #[test]
    fn assign_after_short_circuit_and() {
        test_codegen_mainfunc("int x = 0; 0 && (x = 1); return x;", 0);
    }

    #[test]
    fn assign_after_short_circuit_or() {
        test_codegen_mainfunc("int x = 0; 1 || (x = 1); return x;", 0);
    }

    #[test]
    fn assign_low_precedence() {
        test_codegen_mainfunc("int x; x = 0 || 5; return x;", 1);
    }

    #[test]
    fn assign_var_in_initializer() {
        test_codegen_mainfunc("int x = x + 5; return x;", 5);
    }

    #[test]
    fn empty_main_body() {
        test_codegen_mainfunc("", 0);
    }

    #[test]
    fn null_statement() {
        test_codegen_mainfunc(";", 0);
    }

    #[test]
    fn null_then_return() {
        test_codegen_mainfunc("; return 1;", 1);
    }

    #[test]
    fn empty_expression() {
        test_codegen_mainfunc("return 0;;;", 0);
    }

    #[test]
    fn unused_expression() {
        test_codegen_mainfunc("2 + 2; return 0;", 0);
    }

    #[test]
    fn bitwise_in_initializer() {
        test_codegen_mainfunc(
            r"
    int a = 15;
    int b = a ^ 5;  // 10
    return 1 | b;   // 11",
            11,
        );
    }

    #[test]
    fn bitwise_ops_vars() {
        test_codegen_mainfunc("int a = 3; int b = 5; int c = 8; return a & b | c;", 9);
    }

    #[test]
    fn bitwise_shl_var() {
        test_codegen_mainfunc("int x = 3; return x << 3;", 24);
    }

    #[test]
    fn bitwise_sar_assign() {
        test_codegen_mainfunc(
            "int var_to_shift = 1234; int x = 0; x = var_to_shift >> 4; return x;",
            77,
        );
    }

    #[test]
    fn compound_bitwise_and() {
        test_codegen_mainfunc("int to_and = 3; to_and &= 6; return to_and;", 2);
    }

    #[test]
    fn compound_bitwise_or() {
        test_codegen_mainfunc("int to_or = 1; to_or |= 30; return to_or;", 31);
    }

    #[test]
    fn compound_bitwise_shl() {
        test_codegen_mainfunc("int to_shiftl = 3; to_shiftl <<= 4; return to_shiftl;", 48);
    }

    #[test]
    fn compound_bitwise_sar() {
        test_codegen_mainfunc(
            "int to_shiftr = 382574; to_shiftr >>= 4; return to_shiftr;",
            23910,
        );
    }

    #[test]
    fn compound_bitwise_xor() {
        test_codegen_mainfunc("int to_xor = 7; to_xor ^= 5; return to_xor;", 2);
    }

    #[test]
    fn compound_div() {
        test_codegen_mainfunc("int to_divide = 8; to_divide /= 4; return to_divide;", 2);
    }

    #[test]
    fn compound_subtract() {
        test_codegen_mainfunc(
            "int to_subtract = 10; to_subtract -= 8; return to_subtract;",
            2,
        );
    }

    #[test]
    fn compound_mod() {
        test_codegen_mainfunc("int to_mod = 5; to_mod %= 3; return to_mod;", 2);
    }

    #[test]
    fn compound_mult() {
        test_codegen_mainfunc(
            "int to_multiply = 4; to_multiply *= 3; return to_multiply;",
            12,
        );
    }

    #[test]
    fn compound_add() {
        test_codegen_mainfunc("int to_add = 0; to_add += 4; return to_add;", 4);
    }

    #[test]
    fn compound_assignment_chained() {
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
    }

    #[test]
    fn compound_bitwise_assignment_chained() {
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
    }

    #[test]
    fn compound_assignment_lowest_precedence() {
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
    }

    #[test]
    fn compound_bitwise_assignment_lowest_precedence() {
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
    }

    #[test]
    fn increment_decrement_expressions() {
        test_codegen_mainfunc(
            "int a = 0; int b = 0; a++; ++a; b--; --b; return (a == 2 && b == -2);",
            1,
        );
    }

    #[test]
    fn incr_decr_in_binary_expressions() {
        test_codegen_mainfunc(
            r"
    int a = 2;
    int b = 3 + a++;
    int c = 4 + ++b;
    return (a == 3 && b == 6 && c == 10);
    ",
            1,
        );
    }

    #[test]
    fn incr_decr_parentheses() {
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
    }

    #[test]
    fn if_assign() {
        test_codegen_mainfunc("int x = 5; if (x == 5) x = 4; return x;", 4);
    }

    #[test]
    fn if_else_assign() {
        test_codegen_mainfunc(
            "int x = 5; if (x == 5) x = 4; else x == 6; if (x == 6) x = 7; else x = 8; return x;",
            8,
        );
    }

    #[test]
    fn if_binary_op_in_condition_true() {
        test_codegen_mainfunc("if (1 + 2 == 3) return 5;", 5);
    }

    #[test]
    fn if_binary_op_in_condition_false() {
        test_codegen_mainfunc("if (1 + 2 == 4) return 5;", 0);
    }

    #[test]
    fn if_else_if() {
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
    }

    #[test]
    fn if_else_if_nested_execute_else() {
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
    }

    #[test]
    fn if_nested_twice() {
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
    }

    #[test]
    fn if_nested_twice_execute_else() {
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
    }

    #[test]
    fn nested_else_execute_outer_else() {
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
    }

    #[test]
    fn if_null_body() {
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
    }

    #[test]
    fn multiple_if_else() {
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
    }

    #[test]
    fn if_compound_assignment_in_condition() {
        test_codegen_mainfunc("int a = 0; if (a += 1) return a; return 10;", 1);
    }

    #[test]
    fn if_postfix_in_condition() {
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
    }

    #[test]
    fn if_prefix_in_condition() {
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
    }

    #[test]
    fn assign_ternary() {
        test_codegen_mainfunc("int a = 0; a = 1 ? 2 : 3; return a;", 2);
    }

    #[test]
    fn ternary_binary_op_in_middle() {
        test_codegen_mainfunc("int a = 1 ? 3 % 2 : 4; return a;", 1);
    }

    #[test]
    fn ternary_logical_or_precedence() {
        test_codegen_mainfunc("int a = 10; return a || 0 ? 20 : 0;", 20);
    }

    #[test]
    fn ternary_logical_or_precedence_right() {
        test_codegen_mainfunc("return 0 ? 1 : 0 || 2;", 1);
    }

    #[test]
    fn ternary_in_assignment() {
        test_codegen_mainfunc(
            r"
    int x = 0;
    int y = 0;
    y = (x = 5) ? x : 2;
    return (x == 5 && y == 5);
    ",
            1,
        );
    }

    #[test]
    fn nested_ternary() {
        test_codegen_mainfunc(
            r"
    int a = 1;
    int b = 2;
    int flag = 0;
    return a > b ? 5 : flag ? 6 : 7;
    ",
            7,
        );
    }

    #[test]
    fn nested_ternary_literals() {
        test_codegen_mainfunc(
            r"
    int a = 1 ? 2 ? 3 : 4 : 5;
    int b = 0 ? 2 ? 3 : 4 : 5;
    return a * b;
    ",
            15,
        );
    }

    #[test]
    fn ternary_assignment_rhs() {
        test_codegen_mainfunc(
            r"
    int flag = 1;
    int a = 0;
    flag ? a = 1 : (a = 0);
    return a;
    ",
            1,
        );
    }

    mod goto {
        use super::*;

        #[test]
        fn skip_declaration() {
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
        }

        #[test]
        fn same_as_var_name() {
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
        }

        #[test]
        fn same_as_func_name() {
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
        }

        #[test]
        fn nested_label() {
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
        }

        #[test]
        fn label_all_statements() {
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
        }

        #[test]
        fn label_name() {
            test_codegen_mainfunc(
                r"
    goto _foo_1_;  // a label may include numbers and underscores
    return 0;
_foo_1_:
    return 1;
            ",
                1,
            );
        }

        #[test]
        fn unused_label() {
            test_codegen_mainfunc(
                r"
unused:
    return 0;
            ",
                0,
            );
        }

        #[test]
        fn whitespace_after_label() {
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
        }

        #[test]
        fn goto_after_declaration() {
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
        }

        #[test]
        fn goto_inner_scope() {
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
        }

        #[test]
        fn goto_outer_scope() {
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
        }

        #[test]
        fn jump_between_sibling_scopes() {
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
        }

        mod fail {
            use super::*;

            #[test]
            fn label_name() {
                test_codegen_mainfunc_failure(r"0invalid_label: return 0;");
            }

            #[test]
            fn label_keyword_name() {
                test_codegen_mainfunc_failure(r"return: return 0;");
            }

            #[test]
            fn whitespace_after_label() {
                test_codegen_mainfunc_failure(
                    r"
    goto;
lbl:
    return 0;
                ",
                );
            }

            #[test]
            fn label_declaration() {
                test_codegen_mainfunc_failure(
                    r"
// NOTE: this is a syntax error in C17 but valid in C23
label:
    int a = 0;
                ",
                );
            }

            #[test]
            fn label_without_statement() {
                test_codegen_mainfunc_failure(
                    r"
    // NOTE: this is invalid in C17, but valid in C23
    foo:
                ",
                );
            }

            #[test]
            fn parenthesized_label() {
                test_codegen_mainfunc_failure(
                    r"
    goto(a);
a:
    return 0;
                ",
                );
            }

            #[test]
            fn duplicate_label() {
                test_codegen_mainfunc_failure(
                    r"
    int x = 0;
label:
    x = 1;
label:
    return 2;
                ",
                );
            }

            #[test]
            fn unknown_label() {
                test_codegen_mainfunc_failure(
                    r"
    goto label;
    return 0;
                ",
                );
            }

            #[test]
            fn variable_as_label() {
                test_codegen_mainfunc_failure(
                    r"
    int a;
    goto a;
    return 0;
                ",
                );
            }

            #[test]
            fn undeclared_var_in_labeled_statement() {
                test_codegen_mainfunc_failure(
                    r"
lbl:
    return a;
    return 0;
                ",
                );
            }

            #[test]
            fn label_as_variable() {
                test_codegen_mainfunc_failure(
                    r"
    int x = 0;
    a:
    x = a;
    return 0;
                ",
                );
            }

            #[test]
            fn label_in_expression() {
                test_codegen_mainfunc_failure(r"1 && label: 2;");
            }

            #[test]
            fn label_outside_function() {
                codegen_run_and_expect_compile_failure(
                    r"
label:
int main(void) {
    return 0;
}
                ",
                );
            }

            #[test]
            fn different_label_same_scope() {
                codegen_run_and_expect_compile_failure(
                    r"
    // different labels do not define different scopes
label1:;
    int a = 10;
label2:;
    int a = 11;
    return 1;
                ",
                );
            }

            #[test]
            fn duplicate_labels_different_scopes() {
                codegen_run_and_expect_compile_failure(
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
            }

            #[test]
            fn goto_use_before_declare() {
                codegen_run_and_expect_compile_failure(
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
            }

            #[test]
            fn labeled_break_outside_loop() {
                codegen_run_and_expect_compile_failure(
                    r"
    // make sure our usual analysis of break/continue labels also traverses labeled statements
    label: break;
    return 0;
                ",
                );
            }
        }
    }

    #[test]
    fn var_assign_to_self() {
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
    }

    #[test]
    fn var_assign_to_self_inner_block() {
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
    }

    #[test]
    fn var_assign_to_self_from_other_var_declaration_in_inner_block() {
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
    }

    #[test]
    fn empty_blocks() {
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
    }

    #[test]
    fn var_hidden_then_visible() {
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
    }

    #[test]
    fn var_shadowed() {
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
    }

    #[test]
    fn var_inner_uninitialized() {
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
    }

    #[test]
    fn var_same_name_different_blocks() {
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
    }

    #[test]
    fn nested_if_compound_statements() {
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
    }

    #[test]
    fn nested_var_declarations() {
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
    }

    mod for_loops {
        use super::*;

        #[test]
        fn empty_header() {
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
        }

        #[test]
        fn break_statement() {
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
        }

        #[test]
        fn continue_statement() {
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
        }

        #[test]
        fn declaration() {
            test_codegen_mainfunc(
                r"
    int a = 0;

    for (int i = -100; i <= 0; i = i + 1)
        a = a + 1;
    return a;
            ",
                101,
            );
        }

        #[test]
        fn shadow_declaration() {
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
        }

        #[test]
        fn nested_shadowed_declarations() {
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
        }

        #[test]
        fn no_post_expression() {
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
        }

        #[test]
        fn continue_with_no_post_expression() {
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
        }

        #[test]
        fn no_condition() {
            test_codegen_mainfunc(
                r"
    for (int i = 400; ; i = i - 100)
        if (i == 100)
            return 0;
            ",
                0,
            );
        }

        #[test]
        fn nested_break() {
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
        }

        #[test]
        fn compound_assignent_in_post_expression() {
            test_codegen_mainfunc(
                r"
    int i = 1;
    for (i *= -1; i >= -100; i -=3)
        ;
    return (i == -103);
            ",
                1,
            );
        }

        #[test]
        fn jump_past_initializer() {
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
        }

        #[test]
        fn jump_within_body() {
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
        }

        mod fail {
            use super::*;

            #[test]
            fn extra_header_clause() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (int i = 0; i < 10; i = i + 1; )
        ;
    return 0;
                ",
                );
            }

            #[test]
            fn missing_header_clause() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (int i = 0;)
        ;
    return 0;
                ",
                );
            }

            #[test]
            fn extra_parens() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (int i = 2; ))
        int a = 0;
                ",
                );
            }

            #[test]
            fn invalid_declaration_compound_assignment() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (int i += 1; i < 10; i += 1) {
        return 0;
    }
                ",
                );
            }

            #[test]
            fn declaration_in_condition() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (; int i = 0; i = i + 1)
        ;
    return 0;
                ",
                );
            }

            #[test]
            fn label_in_header() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (int i = 0; label: i < 10; i = i + 1) {
        ;
    }
    return 0;
                ",
                );
            }

            #[test]
            fn undeclared_variable() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (i = 0; i < 1; i = i + 1)
    {
        return 0;
    }
                ",
                );
            }

            #[test]
            fn reference_body_variable_in_condition() {
                codegen_run_and_expect_compile_failure(
                    r"
    for (;; i++) {
        int i = 0;
    }
                ",
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn while_loop() {
        test_codegen_mainfunc("int x = 1; while (x < 10) x = x + 1; return x;", 10);
    }

    #[test]
    #[ignore]
    fn while_loop_with_break() {
        test_codegen_mainfunc(
            "int x = 1; while (x < 10) { x = x + 1; break; } return x;",
            2,
        );
    }

    #[test]
    #[ignore]
    fn while_loop_with_continue() {
        test_codegen_mainfunc(
            "int x = 1; while (x < 10) { x = x + 1; continue; x = 50; } return x;",
            10,
        );
    }

    #[test]
    #[ignore]
    fn do_while_loop() {
        test_codegen_mainfunc("do { return 1; } while (0); return 2;", 1);
    }

    #[test]
    #[ignore]
    fn do_while_loop_with_break() {
        test_codegen_mainfunc("do { break; return 1; } while (0); return 2;", 2);
    }

    #[test]
    #[ignore]
    fn do_while_loop_with_continue() {
        test_codegen_mainfunc(
            "int x = 20; do { continue; } while ((x = 50) < 10); return x;",
            50,
        );
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
