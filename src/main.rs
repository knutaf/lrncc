#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate regex;

use {
    clap::Parser,
    rand::distr::Alphanumeric,
    rand::Rng,
    regex::Regex,
    std::{
        collections::{HashMap, HashSet},
        fmt,
        fmt::Display,
        ops::Deref,
        path,
        path::*,
        process::*,
    },
    tracing::{debug, error, info, instrument},
    tracing_subscriber,
    tracing_subscriber::{prelude::*, util::SubscriberInitExt, EnvFilter, Registry},
};

const VARIABLE_SIZE: u32 = 4;
// x64 ABI says that stack parameters are 8-bytes in size, regardless of what data type is passed in them.
const STACK_PARAMETER_SIZE: u32 = 8;
const POINTER_SIZE: u32 = 8;
const FUNCTION_CALL_ALIGNMENT: u32 = 16;

const KEYWORDS: [&'static str; 13] = [
    "int", "void", "return", "if", "goto", "for", "while", "do", "break", "continue", "switch",
    "case", "default",
];

const PARAMETER_REG_MAPPING: [&'static str; 4] = ["cx", "dx", "r8", "r9"];

fn generate_random_string(len: usize) -> String {
    rand::rng()
        .sample_iter(&Alphanumeric)
        .map(char::from)
        .take(len)
        .collect()
}

// Convenience method for formatting an io::Error to String.
fn format_io_err(err: std::io::Error) -> String {
    format!("{}: {}", err.kind(), err)
}

fn pad_to_alignment(num: u32, alignment: u32) -> u32 {
    if num <= alignment {
        return alignment;
    }

    if num % alignment == 0 {
        return num;
    }

    ((num / alignment) + 1) * alignment
}

fn format_command_args(command: &Command) -> String {
    format!(
        "{} {}",
        command.get_program().to_str().unwrap(),
        command
            .get_args()
            .map(|s| s.to_str().unwrap())
            .collect::<Vec<&str>>()
            .join(" ")
    )
}

fn push_error<TError>(result: Result<TError, String>, errors: &mut Vec<String>) -> bool {
    if let Err(error) = result {
        errors.push(error);
        true
    } else {
        false
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

// This implements FmtNode for any reference to a type that implements FmtNode. It's mostly useful along with
// DisplayFmtNode, below.
impl<T: FmtNode + ?Sized> FmtNode for &T {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        (*self).fmt_node(f, indent_levels)
    }
}

impl<T: FmtNode + ?Sized> FmtNode for &mut T {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        (*(self as &T)).fmt_node(f, indent_levels)
    }
}

// Because Rust doesn't allow a blanket implementation of fmt::Display for FmtNode, we can instead make a bare wrapper
// to do it for us.
struct DisplayFmtNode<T>(T);
impl<T: FmtNode> fmt::Display for DisplayFmtNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt_node(f, 0)
    }
}

/// Produces a mangled name that won't clash with a C identifier but is valid to use in ASM listings.
fn mangle_function_name(name: &str) -> String {
    // Do not mangle "main" or else your program crashes. Not totally sure why.
    if name == "main" {
        String::from("main")
    } else {
        format!("@@@{}", name)
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

    fn try_consume_expected_next_token(&mut self, expected_token: &str) -> bool {
        self.consume_expected_next_token(expected_token).is_ok()
    }

    fn consume_expected_next_token(&mut self, expected_token: &str) -> Result<(), String> {
        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        if tokens[0] == expected_token {
            *self = remaining_tokens;
            debug!(name: "consume_expected_token", target: "parse", token = expected_token, "consumed");
            Ok(())
        } else {
            debug!(
                "expected next token \"{}\" but found \"{}\"",
                expected_token, tokens[0]
            );
            Err(format!(
                "expected next token \"{}\" but found \"{}\"",
                expected_token, tokens[0]
            ))
        }
    }

    fn try_consume_identifier(&mut self) -> Option<&'i str> {
        self.consume_identifier().ok()
    }

    fn consume_identifier(&mut self) -> Result<&'i str, String> {
        fn is_token_identifier(token: &str) -> bool {
            lazy_static! {
                static ref IDENT_REGEX: Regex =
                    Regex::new(r"^[a-zA-Z_]\w*$").expect("failed to compile regex");
                static ref KEYWORDS_REGEX: Regex =
                    Regex::new(&format!(r"^(?:{})$", KEYWORDS.join("|")))
                        .expect("failed to compile regex");
            }

            IDENT_REGEX.is_match(token) && !KEYWORDS_REGEX.is_match(token)
        }

        let (tokens, remaining_tokens) = self.consume_tokens(1)?;

        if is_token_identifier(tokens[0]) {
            *self = remaining_tokens;
            debug!("consumed identifier {}", tokens[0]);
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
struct AstProgram {
    functions: Vec<AstFunction>,
}

#[derive(Debug, Clone)]
struct AstBlock(Vec<AstBlockItem>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct AstIdentifier(String);

#[derive(Debug, Clone)]
struct AstFunction {
    name: AstIdentifier,
    parameters: Vec<AstIdentifier>,
    body_opt: Option<Vec<AstBlockItem>>,
}

#[derive(Debug, Clone)]
enum AstBlockItem {
    Statement(AstStatement),
    VarDeclaration(AstVarDeclaration),
    FuncDeclaration(AstFunction),
}

#[derive(Clone, Debug)]
struct AstVarDeclaration {
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
    While(
        AstExpression,
        Box<AstStatement>,
        Option<AstLabel>,
        Option<AstLabel>,
    ),
    DoWhile(
        AstExpression,
        Box<AstStatement>,
        Option<AstLabel>,
        Option<AstLabel>,
    ),
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
    Switch(
        AstExpression,
        Box<AstStatement>,
        Option<AstLabel>,
        Vec<SwitchCase>,
    ),
    SwitchCase(SwitchCase, Box<AstStatement>),
    Null,
}

#[derive(Clone, Debug)]
enum AstForInit {
    Declaration(AstVarDeclaration),
    Expression(Option<AstExpression>),
}

#[derive(Clone, Debug)]
struct SwitchCase {
    expr_opt: Option<AstExpression>,
    label_opt: Option<AstLabel>,
}

#[derive(Clone, Debug)]
enum AstExpression {
    Constant(u32),
    UnaryOperator(AstUnaryOperator, Box<AstExpression>),
    BinaryOperator(Box<AstExpression>, AstBinaryOperator, Box<AstExpression>),
    Var(AstIdentifier),
    Assignment(Box<AstExpression>, AstBinaryOperator, Box<AstExpression>),
    Conditional(Box<AstExpression>, Box<AstExpression>, Box<AstExpression>),
    FuncCall(AstIdentifier, Vec<AstExpression>),
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
    parameters: Vec<TacVar>,
    body_opt: Option<Vec<TacInstruction>>,
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
    JumpIfEqual(TacVal, TacVal, TacLabel),
    Label(TacLabel),
    FuncCall(TacLabel, Vec<TacVal>, TacVar),
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
    mangled_name: String,
    body_opt: Option<Vec<AsmInstruction>>,
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
    Call(String),
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
    PseudoArgReg(u32),
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
struct SymbolTable {
    symbols: HashMap<AstIdentifier, Symbol>,
}

#[derive(Debug, Clone)]
enum Symbol {
    Var(AstIdentifier),
    Function(usize, bool),
}

#[derive(Debug)]
struct GlobalTracking {
    next_temporary_id: u32,
    next_label_id: u32,
    symbols: SymbolTable,
}

#[derive(Debug)]
struct FunctionTracking {
    labels: HashMap<AstLabel, AstLabel>,
}

#[derive(Debug)]
struct FuncStackFrame {
    names: HashMap<String, i32>,
    locals_size: u32,
    arguments_size: u32,
}

enum AstNode<'n> {
    Function(&'n mut AstFunction),
    Block(&'n mut AstBlock),
    Statement(&'n mut AstStatement),
    VarDeclaration(&'n mut AstVarDeclaration),
    ForInit(&'n mut AstForInit),
    Expression(&'n mut AstExpression),
}

enum AsmNode<'n> {
    Loc(&'n mut AsmLocation),
}

/// A common pattern is traversing a tree and keeping track of some potentially nested context. This object allows for
/// easy nesting and unnesting as the traversal enters or exits nodes in the tree.
#[derive(Debug)]
struct TraversalContext<'e, T> {
    inner: Option<Box<T>>,
    errors_opt: Option<&'e mut Vec<String>>,
    errors_len: usize,
}

/// To participate in TraversalContext, the context object needs to have a parent reference somewhere in it.
trait Nestable {
    fn get_parent_opt_mut(&mut self) -> &mut Option<Box<Self>>;
}

impl AstProgram {
    fn new(functions: Vec<AstFunction>) -> Self {
        Self { functions }
    }

    fn validate_and_resolve(&mut self) -> Result<(), Vec<String>> {
        let mut global_tracking = GlobalTracking::new();

        let mut errors = vec![];
        for func in self.functions.iter_mut() {
            let mut function_tracking = FunctionTracking::new();

            func.validate_and_resolve_symbols(&mut global_tracking, &mut errors);
            debug!(target: "resolve", func = %DisplayFmtNode(&func), "func after validate_and_resolve");

            func.validate_and_allocate_goto_labels(
                &mut global_tracking,
                &mut function_tracking,
                &mut errors,
            );
            debug!(target: "resolve", func = %DisplayFmtNode(&func), "func after label checking");

            func.rewrite_goto_labels(&function_tracking, &mut errors);
            debug!(target: "resolve", func = %DisplayFmtNode(&func), "func after goto resolution");

            func.label_loops_and_switches(&mut global_tracking, &mut errors);
            debug!(target: "resolve", func = %DisplayFmtNode(&func), "func after loop labeling");

            func.resolve_switch_cases(&mut global_tracking, &mut errors);
            debug!(target: "resolve", func = %DisplayFmtNode(&func), "func after switch resolving");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn to_tac(&self) -> Result<TacProgram, String> {
        let mut global_tracking = GlobalTracking::new();

        let mut functions = vec![];
        for func in self.functions.iter() {
            functions.push(func.to_tac(&mut global_tracking)?);
        }

        Ok(TacProgram { functions })
    }
}

impl FmtNode for AstProgram {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        Self::fmt_nodelist(f, self.functions.iter(), "\n\n", 0)
    }
}

impl AstBlock {
    fn traverse_ast<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AstNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        let res = (func)(true, AstNode::Block(self), context);
        if context.push_error(res) {
            return;
        }

        for block_item in self.0.iter_mut() {
            block_item.traverse_ast(context, func);
        }

        let res = (func)(false, AstNode::Block(self), context);
        context.push_error(res);
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

impl AstFunction {
    fn is_definition(&self) -> bool {
        self.body_opt.is_some()
    }

    fn validate_and_resolve_symbols(
        &mut self,
        global_tracking: &mut GlobalTracking,
        errors: &mut Vec<String>,
    ) {
        struct ResolveTracking<'g, 'e> {
            global_tracking: &'g mut GlobalTracking,
            block_tracking_opt: TraversalContext<'e, BlockTracking>,
        }

        impl<'g, 'e> ResolveTracking<'g, 'e> {
            fn lookup_symbol(&self, identifier: &AstIdentifier) -> Result<&Symbol, String> {
                // First look up the symbol in the current block, which is the narrowest scope. This will also search
                // parent scopes in turn.
                if let Some(block_tracking) = self.block_tracking_opt.get() {
                    if let Ok(symbol) = block_tracking.lookup_symbol(identifier) {
                        return Ok(symbol);
                    }
                }

                // If it wasn't found in any block scope then search the global scope last.
                match self.global_tracking.symbols.lookup_symbol(identifier) {
                    Ok(Some(symbol)) => Ok(symbol),
                    Ok(None) => Err(format!("identifier {} not found", identifier)),
                    Err(err @ _) => Err(err),
                }
            }

            fn lookup_function_declaration(
                &self,
                identifier: &AstIdentifier,
            ) -> Result<&Symbol, String> {
                let symbol @ Symbol::Function(_, _) = self.lookup_symbol(identifier)? else {
                    return Err(format!("{} is not a function", identifier));
                };

                Ok(symbol)
            }
        }

        struct BlockTracking {
            parent_opt: Option<Box<BlockTracking>>,
            symbols: SymbolTable,
        }

        impl BlockTracking {
            fn new() -> Self {
                Self {
                    parent_opt: None,
                    symbols: SymbolTable::new(),
                }
            }

            fn lookup_symbol(&self, identifier: &AstIdentifier) -> Result<&Symbol, String> {
                if let Some(symbol) = self.symbols.symbols.get(identifier) {
                    Ok(symbol)
                } else {
                    return if let Some(ref parent) = self.parent_opt {
                        parent.lookup_symbol(identifier)
                    } else {
                        Err(format!("identifier '{}' not found", identifier))
                    };
                }
            }

            fn resolve_variable(
                &self,
                identifier: &AstIdentifier,
            ) -> Result<&AstIdentifier, String> {
                let Symbol::Var(ref temp_identifier) = self.lookup_symbol(identifier)? else {
                    return Err(format!("{} is not a variable", identifier));
                };

                Ok(temp_identifier)
            }
        }

        impl Nestable for BlockTracking {
            fn get_parent_opt_mut(&mut self) -> &mut Option<Box<Self>> {
                &mut self.parent_opt
            }
        }

        let mut resolve_tracking = TraversalContext::new(
            Some(ResolveTracking {
                global_tracking,
                block_tracking_opt: TraversalContext::new(None, None),
            }),
            Some(errors),
        );

        self.traverse_ast(&mut resolve_tracking, &mut |is_enter, node, context| {
            if is_enter {
                match node {
                    AstNode::VarDeclaration(decl) => {
                        let resolve_tracking = context.get_mut().unwrap();

                        let Some(block_tracking) = resolve_tracking.block_tracking_opt.get_mut()
                        else {
                            unreachable!("variable declaration outside of a block");
                        };

                        decl.identifier = block_tracking.symbols
                            .add_variable(&mut resolve_tracking.global_tracking, &decl.identifier)?;
                    }
                    AstNode::Function(function) => {
                        let resolve_tracking = context.get_mut().unwrap();

                        // First lookup this new function declaration in the global symbol table. If the symbol is found
                        // but is not a function, e.g.. is a variable, then no checks need to be done specifically
                        // against it. We'll next try to add it to the appropriate scope, and that will fail if there's
                        // a type mismatch.
                        if let Some(Symbol::Function(param_count, is_definition)) = resolve_tracking.global_tracking.symbols.lookup_symbol(&function.name)? {
                            // Already had a definition stored in the symbol table, and now trying to store another.
                            if *is_definition && function.is_definition() {
                                return Err(format!("duplicate function definition for {}", function.name));
                            }

                            if *param_count != function.parameters.len() {
                                return Err(format!("cannot declare function {} with {} parameters. previously declared with {} parameters", function.name, function.parameters.len(), param_count));
                            }
                        }

                        // We are in a block, so add this function declaration to the current block.
                        if let Some(block_tracking) = resolve_tracking.block_tracking_opt.get_mut() {
                            if function.is_definition() {
                                return Err(format!("not permitted to have function definition within another function definition"));
                            }

                            block_tracking.symbols.add_function(&function)?;
                        }

                        // Function declarations have external linkage, and so all declarations of a given function need
                        // to have the same signature, so also add it to the global table.
                        resolve_tracking.global_tracking.symbols.add_function(&function)?;

                        // Entering the function body creates a new scope. Note that this scope includes the parameters
                        // and the function body together, so that declaring a variable in the function body that
                        // tries to shadow a parameter is an error.
                        if function.is_definition() {
                            let mut new_block_tracking = BlockTracking::new();
                            for param in function.parameters.iter_mut() {
                                let temp_ident = new_block_tracking.symbols.add_variable(resolve_tracking.global_tracking, param)?;
                                *param = temp_ident;
                            }

                            context
                                .get_mut()
                                .unwrap()
                                .block_tracking_opt
                                .nest(new_block_tracking);
                        }
                    }
                    AstNode::Block(_)
                    | AstNode::Statement(AstStatement {
                        typ: AstStatementType::For(_, _, _, _, _, _),
                        ..
                    }) => {
                        context
                            .get_mut()
                            .unwrap()
                            .block_tracking_opt
                            .nest(BlockTracking::new());
                    }
                    AstNode::Expression(
                        AstExpression::Assignment(ast_exp_destination, _, _)
                        | AstExpression::UnaryOperator(
                            AstUnaryOperator::PrefixIncrement
                            | AstUnaryOperator::PrefixDecrement
                            | AstUnaryOperator::PostfixIncrement
                            | AstUnaryOperator::PostfixDecrement,
                            ast_exp_destination,
                        ),
                    ) => {
                        let AstExpression::Var(_) = **ast_exp_destination else {
                            return Err(format!(
                                "{} is not an lvalue",
                                display_with(|f| { ast_exp_destination.fmt_node(f, 0) })
                            ));
                        };
                    }
                    AstNode::Expression(AstExpression::Var(ast_ident)) => {
                        let Some(block_tracking) =
                            context.get_mut().unwrap().block_tracking_opt.get_mut()
                        else {
                            unreachable!("variable use {} outside of a block", ast_ident);
                        };

                        *ast_ident = block_tracking.resolve_variable(ast_ident)?.clone();
                    }
                    AstNode::Expression(AstExpression::FuncCall(ast_ident, arguments)) => {
                        let Symbol::Function(param_count, _) = context.get().unwrap().lookup_function_declaration(ast_ident)? else {
                            unreachable!("lookup_function_declaration returned something other than a function");
                        };

                        if arguments.len() != *param_count {
                            return Err(format!("function {} expected {} arguments but was passed {}", ast_ident, param_count, arguments.len()));
                        }
                    }
                    _ => {}
                }
            } else {
                match node {
                    AstNode::Function(function) => {
                        if function.is_definition() {
                            let _ = context.get_mut().unwrap().block_tracking_opt.unnest();
                        }
                    }
                    AstNode::Block(_)
                    | AstNode::Statement(AstStatement {
                        typ: AstStatementType::For(_, _, _, _, _, _),
                        ..
                    }) => {
                        let _ = context.get_mut().unwrap().block_tracking_opt.unnest();
                    }
                    _ => {}
                }
            }

            Ok(())
        });

        // Either the inner, nested part of the context had a clean traversal or there were errors.
        assert!(
            resolve_tracking
                .get()
                .unwrap()
                .block_tracking_opt
                .is_clean_traversal()
                || resolve_tracking.had_errors()
        );
    }

    fn traverse_ast<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AstNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        let res = (func)(true, AstNode::Function(self), context);
        if context.push_error(res) {
            return;
        }

        if let Some(ref mut body) = self.body_opt {
            for block_item in body.iter_mut() {
                block_item.traverse_ast(context, func);
            }
        }

        let res = (func)(false, AstNode::Function(self), context);
        context.push_error(res);
    }

    fn to_tac(&self, global_tracking: &mut GlobalTracking) -> Result<TacFunction, String> {
        let body_opt = if let Some(body) = self.body_opt.as_ref() {
            let mut body_instructions = vec![];
            for block_item in body.iter() {
                block_item.to_tac(global_tracking, &mut body_instructions)?;
            }

            // In C, functions lacking a return value either automatically return 0 (in the case of main), have undefined
            // behavior (if the function's return value is actually used), or the return value doesn't matter (if the return
            // value isn't examined). To handle all three cases, just add an extra "return 0" to the end of every function
            // body.
            body_instructions.push(TacInstruction::Return(TacVal::Constant(0)));
            Some(body_instructions)
        } else {
            None
        };

        Ok(TacFunction {
            name: self.name.0.clone(),
            parameters: self.parameters.iter().map(AstIdentifier::to_tac).collect(),
            body_opt,
        })
    }

    fn validate_and_allocate_goto_labels(
        &mut self,
        global_tracking: &mut GlobalTracking,
        function_tracking: &mut FunctionTracking,
        errors: &mut Vec<String>,
    ) {
        // Examine all the goto labels in the function and rewrite them with an internal name while also checking
        // that there are no duplicates.
        self.traverse_ast(
            &mut TraversalContext::new_blank(Some(errors)),
            &mut |is_enter, mut node, context| {
                if !is_enter {
                    return Ok(());
                }

                let AstNode::Statement(ref mut ast_statement) = node else {
                    return Ok(());
                };

                for label in ast_statement.labels.iter_mut() {
                    let res = function_tracking.add_goto_label(global_tracking, label);
                    if let Ok(new_label) = res {
                        *label = new_label;
                    } else {
                        context.push_error(res);
                    }
                }

                Ok(())
            },
        );
    }

    fn rewrite_goto_labels(
        &mut self,
        function_tracking: &FunctionTracking,
        errors: &mut Vec<String>,
    ) {
        // Examine each goto statement and convert its target label to the internal label name.
        self.traverse_ast(
            &mut TraversalContext::new_blank(Some(errors)),
            &mut |is_enter, node, _context| {
                if !is_enter {
                    return Ok(());
                }

                let AstNode::Statement(AstStatement {
                    typ: AstStatementType::Goto(ref mut label),
                    ..
                }) = node
                else {
                    return Ok(());
                };

                *label = function_tracking.resolve_label(label)?;
                Ok(())
            },
        );
    }

    // Loop labeling (and also switch statements): each loop has two labels associated with it: one that a break
    // statement should jump to (after the end of the loop) and one that a continue statement should jump to (after the
    // end of the body).
    //
    // Switch statements are similar but do not have a label for continue statements.
    fn label_loops_and_switches(
        &mut self,
        global_tracking: &mut GlobalTracking,
        errors: &mut Vec<String>,
    ) {
        #[derive(Debug)]
        struct BreakTracking {
            parent_opt: Option<Box<BreakTracking>>,
            break_label: AstLabel,
            continue_label_opt: Option<AstLabel>,
        }

        impl BreakTracking {
            fn new(break_label: AstLabel, continue_label_opt: Option<AstLabel>) -> Self {
                Self {
                    parent_opt: None,
                    break_label,
                    continue_label_opt,
                }
            }

            fn find_continue_label(&self) -> Option<&AstLabel> {
                // Walk the parent chain until we find the closest continue label.
                if self.continue_label_opt.is_some() {
                    self.continue_label_opt.as_ref()
                } else if let Some(parent) = self.parent_opt.as_ref() {
                    parent.find_continue_label()
                } else {
                    None
                }
            }
        }

        impl Nestable for BreakTracking {
            fn get_parent_opt_mut(&mut self) -> &mut Option<Box<Self>> {
                &mut self.parent_opt
            }
        }

        let mut break_tracking_opt = TraversalContext::new(None, Some(errors));

        self.traverse_ast(
            &mut break_tracking_opt,
            &mut |is_enter, mut node, context| {
                let AstNode::Statement(ref mut ast_statement) = node else {
                    return Ok(());
                };

                if is_enter {
                    match ast_statement.typ {
                        AstStatementType::For(
                            _,
                            _,
                            _,
                            _,
                            ref mut ast_body_end_label_opt,
                            ref mut ast_loop_end_label_opt,
                        )
                        | AstStatementType::While(
                            _,
                            _,
                            ref mut ast_body_end_label_opt,
                            ref mut ast_loop_end_label_opt,
                        )
                        | AstStatementType::DoWhile(
                            _,
                            _,
                            ref mut ast_body_end_label_opt,
                            ref mut ast_loop_end_label_opt,
                        ) => {
                            assert!(ast_body_end_label_opt.is_none());
                            assert!(ast_loop_end_label_opt.is_none());

                            // Create new labels for this loop's body end and loop end, to be used in break and continue
                            // statements.
                            *ast_body_end_label_opt =
                                Some(AstLabel(global_tracking.allocate_label("loop_body_end")));
                            *ast_loop_end_label_opt =
                                Some(AstLabel(global_tracking.allocate_label("loop_end")));

                            context.nest(BreakTracking::new(
                                ast_loop_end_label_opt.as_ref().unwrap().clone(),
                                ast_body_end_label_opt.clone(),
                            ));
                        }
                        AstStatementType::Break(ref mut label_opt) => {
                            assert!(label_opt.is_none());

                            let Some(ref break_tracking) = context.get() else {
                                return Err(format!(
                                    "break statement with no containing loop or switch"
                                ));
                            };

                            // Write in the correct label to use for breaking in this scope.
                            *label_opt = Some(break_tracking.break_label.clone());
                        }
                        AstStatementType::Continue(ref mut label_opt) => {
                            assert!(label_opt.is_none());

                            // This could happen if the continue statement is found outside of a loop.
                            let Some(ref break_tracking) = context.get() else {
                                return Err(format!("continue statement with no containing loop"));
                            };

                            // This could happen if the continue statement is found outside of a loop or anything else
                            // that uses BreakTracking, such as a switch statement.
                            let Some(continue_label) = break_tracking.find_continue_label() else {
                                return Err(format!("continue statement with no containing loop"));
                            };

                            // Write in the correct label to use for continuing in this scope.
                            *label_opt = Some(continue_label.clone());
                        }
                        AstStatementType::Switch(_, _, ref mut ast_body_end_label_opt, _) => {
                            assert!(ast_body_end_label_opt.is_none());

                            // Create new labels for this switch statement's body end, to be used in break
                            // statements.
                            *ast_body_end_label_opt =
                                Some(AstLabel(global_tracking.allocate_label("switch_body_end")));

                            // Store the pre-existing tracking info into the new tracking info, so we can revert to
                            // it after this loop is done.
                            context.nest(BreakTracking::new(
                                ast_body_end_label_opt.as_ref().unwrap().clone(),
                                None,
                            ));
                        }
                        _ => {}
                    }
                } else {
                    match ast_statement.typ {
                        AstStatementType::For(_, _, _, _, _, _)
                        | AstStatementType::While(_, _, _, _)
                        | AstStatementType::DoWhile(_, _, _, _)
                        | AstStatementType::Switch(_, _, _, _) => {
                            assert!(context.get().is_some());

                            // This for loop is done, so revert the break tracking to its containing one.
                            let _ = context.unnest();
                        }
                        _ => {}
                    }
                }

                Ok(())
            },
        );

        assert!(break_tracking_opt.is_clean_traversal());
    }

    fn resolve_switch_cases(
        &mut self,
        global_tracking: &mut GlobalTracking,
        errors: &mut Vec<String>,
    ) {
        #[derive(Debug)]
        struct SwitchTracking {
            parent_opt: Option<Box<SwitchTracking>>,
            is_default_case_found: bool,
            case_values: HashSet<i32>,
            cases: Vec<SwitchCase>,
        }

        impl SwitchTracking {
            fn new() -> Self {
                Self {
                    parent_opt: None,
                    is_default_case_found: false,
                    case_values: HashSet::new(),
                    cases: Vec::new(),
                }
            }
        }

        impl Nestable for SwitchTracking {
            fn get_parent_opt_mut(&mut self) -> &mut Option<Box<Self>> {
                &mut self.parent_opt
            }
        }

        let mut switch_tracking_opt = TraversalContext::new(None, Some(errors));

        self.traverse_ast(
            &mut switch_tracking_opt,
            &mut |is_enter, mut node, context| {
                let AstNode::Statement(ref mut ast_statement) = node else {
                    return Ok(());
                };

                if is_enter {
                    match ast_statement.typ {
                        AstStatementType::Switch(_, _, _, _) => {
                            // Encountered a switch statement, so enter a new switch tracking.
                            context.nest(SwitchTracking::new());
                        }
                        AstStatementType::SwitchCase(ref mut case, _) => {
                            let Some(ref mut switch_tracking) = context.get_mut() else {
                                return Err(format!(
                                    "case statement outside of a switch statement"
                                ));
                            };

                            if let Some(expr) = case.expr_opt.as_ref() {
                                let val = expr.eval_i32_constant()?;
                                if !switch_tracking.case_values.insert(val) {
                                    return Err(format!("duplicate case {}", val));
                                }

                                case.label_opt =
                                    Some(AstLabel(global_tracking.allocate_label(&format!(
                                        "case_{}{}",
                                        if val < 0 { "neg" } else { "" },
                                        val.abs()
                                    ))));
                            } else {
                                if switch_tracking.is_default_case_found {
                                    return Err(format!("multiple default cases found"));
                                }

                                case.label_opt =
                                    Some(AstLabel(global_tracking.allocate_label("case_def")));
                                switch_tracking.is_default_case_found = true;
                            }

                            assert!(case.label_opt.is_some());
                            switch_tracking.cases.push(case.clone());
                        }
                        _ => {}
                    }
                } else {
                    if let AstStatementType::Switch(_, _, _, ref mut cases) = ast_statement.typ {
                        let switch_tracking = context.unnest();
                        *cases = switch_tracking.cases;
                    }
                }

                Ok(())
            },
        );

        // On success, we should have unwrapped all the switch tracking.
        assert!(switch_tracking_opt.is_clean_traversal());
    }
}

impl FmtNode for AstFunction {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;
        write!(f, "FUNC {}(", self.name)?;
        fmt_list(f, self.parameters.iter(), ", ")?;
        write!(f, ")")?;

        if let Some(ref body) = self.body_opt {
            writeln!(f, ":")?;
            for block_item in body.iter() {
                block_item.fmt_node(f, indent_levels + 1)?;
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

impl AstIdentifier {
    fn to_tac(&self) -> TacVar {
        TacVar(self.0.clone())
    }

    fn to_tac_label(&self) -> TacLabel {
        TacLabel(self.0.clone())
    }
}

impl fmt::Display for AstIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl AstBlockItem {
    fn traverse_ast<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AstNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        match self {
            AstBlockItem::Statement(statement) => {
                statement.traverse_ast(context, func);
            }
            AstBlockItem::VarDeclaration(declaration) => {
                declaration.traverse_ast(context, func);
            }
            AstBlockItem::FuncDeclaration(declaration) => {
                declaration.traverse_ast(context, func);
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
            AstBlockItem::VarDeclaration(declaration) => {
                declaration.to_tac(global_tracking, instructions)
            }
            AstBlockItem::FuncDeclaration(declaration) => {
                // Function definitions as block items aren't supported and should have been rejected in a precious
                // analysis pass. For declarations, no code is emitted.
                assert!(!declaration.is_definition());
                Ok(())
            }
        }
    }
}

impl FmtNode for AstBlockItem {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        match self {
            AstBlockItem::Statement(statement) => statement.fmt_node(f, indent_levels),
            AstBlockItem::VarDeclaration(declaration) => declaration.fmt_node(f, indent_levels),
            AstBlockItem::FuncDeclaration(declaration) => declaration.fmt_node(f, indent_levels),
        }
    }
}

impl AstVarDeclaration {
    fn traverse_ast<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AstNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        let res = (func)(true, AstNode::VarDeclaration(self), context);
        if context.push_error(res) {
            return;
        }

        if let Some(initializer) = &mut self.initializer_opt {
            initializer.traverse_ast(context, func);
        }

        let res = (func)(false, AstNode::VarDeclaration(self), context);
        context.push_error(res);
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

impl FmtNode for AstVarDeclaration {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;

        write!(f, "int {}", self.identifier)?;

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

impl fmt::Display for AstLabel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl AstStatement {
    fn new(typ: AstStatementType, labels: Vec<AstLabel>) -> Self {
        Self { typ, labels }
    }

    fn traverse_ast<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AstNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        let res = (func)(true, AstNode::Statement(self), context);
        if context.push_error(res) {
            return;
        }

        match &mut self.typ {
            AstStatementType::Return(expr) | AstStatementType::Expr(expr) => {
                expr.traverse_ast(context, func);
            }
            AstStatementType::If(condition_expr, then_statement, else_statement_opt) => {
                condition_expr.traverse_ast(context, func);
                then_statement.traverse_ast(context, func);

                if let Some(else_statement) = else_statement_opt {
                    else_statement.traverse_ast(context, func);
                }
            }
            AstStatementType::While(condition_expr, body_statement, _, _)
            | AstStatementType::DoWhile(condition_expr, body_statement, _, _)
            | AstStatementType::Switch(condition_expr, body_statement, _, _) => {
                condition_expr.traverse_ast(context, func);
                body_statement.traverse_ast(context, func);
            }
            AstStatementType::Goto(_label) => {}
            AstStatementType::Compound(block) => {
                block.traverse_ast(context, func);
            }
            AstStatementType::For(
                ast_for_init,
                ast_expr_condition_opt,
                ast_expr_final_opt,
                ast_statement_body,
                _ast_body_end_label_opt,
                _ast_loop_end_label_opt,
            ) => {
                ast_for_init.traverse_ast(context, func);

                if let Some(expr) = ast_expr_condition_opt {
                    expr.traverse_ast(context, func);
                }

                if let Some(expr) = ast_expr_final_opt {
                    expr.traverse_ast(context, func);
                }

                ast_statement_body.traverse_ast(context, func);
            }
            AstStatementType::Break(_label) => {}
            AstStatementType::Continue(_label) => {}
            AstStatementType::SwitchCase(SwitchCase { expr_opt, .. }, statement) => {
                if let Some(expr) = expr_opt {
                    expr.traverse_ast(context, func);
                }

                statement.traverse_ast(context, func);
            }
            AstStatementType::Null => {}
        }

        let res = (func)(false, AstNode::Statement(self), context);
        context.push_error(res);
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
            AstStatementType::While(
                condition_expr,
                body_statement,
                ast_body_end_label_opt,
                ast_loop_end_label_opt,
            ) => {
                // The body end label can actually go just before the condition, since it's only used as a target for
                // continue statements.
                instructions.push(TacInstruction::Label(
                    ast_body_end_label_opt.as_ref().unwrap().to_tac(),
                ));

                let tac_condition_val = condition_expr.to_tac(global_tracking, instructions)?;

                instructions.push(TacInstruction::JumpIfZero(
                    tac_condition_val,
                    ast_loop_end_label_opt.as_ref().unwrap().to_tac(),
                ));

                body_statement.to_tac(global_tracking, instructions)?;

                // It's a loop, so of course we have to jump back to the top after the end of the body. The body end
                // label is actually at the top of the loop for this type of loop.
                instructions.push(TacInstruction::Jump(
                    ast_body_end_label_opt.as_ref().unwrap().to_tac(),
                ));

                // The loop end label is used for two purposes: jump to it if the loop condition fails; and jump to it
                // from break statements.
                instructions.push(TacInstruction::Label(
                    ast_loop_end_label_opt.as_ref().unwrap().to_tac(),
                ));
            }
            AstStatementType::DoWhile(
                condition_expr,
                body_statement,
                ast_body_end_label_opt,
                ast_loop_end_label_opt,
            ) => {
                // We need a label at the beginning of the body to jump back to after the condition evaluates to true.
                let body_begin_label =
                    TacLabel(global_tracking.allocate_label("do_while_body_begin"));
                instructions.push(TacInstruction::Label(body_begin_label.clone()));

                // In a do-while loop, the body executes before the condition is evaluated.
                body_statement.to_tac(global_tracking, instructions)?;

                // The body end label goes just before the condition. It's only used as a target for continue
                // statements.
                instructions.push(TacInstruction::Label(
                    ast_body_end_label_opt.as_ref().unwrap().to_tac(),
                ));

                let tac_condition_val = condition_expr.to_tac(global_tracking, instructions)?;

                // If the condition is true, jump back to do the body again. If it's false, it'll fall through to exit
                // the loop.
                instructions.push(TacInstruction::JumpIfNotZero(
                    tac_condition_val,
                    body_begin_label,
                ));

                // For this kind of loop, the end label is only used for break statements.
                instructions.push(TacInstruction::Label(
                    ast_loop_end_label_opt.as_ref().unwrap().to_tac(),
                ));
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
            AstStatementType::Switch(
                condition_expr,
                body_statement,
                ast_body_end_label_opt,
                cases,
            ) => {
                // Evaluate the target condition of the switch statement.
                let tac_condition_val = condition_expr.to_tac(global_tracking, instructions)?;

                // Previously we have gathered all the cases within the switch statement. Emit code that checks each one
                // to see which should be jumped to. First emit all the explicit cases (not the default case, if there
                // is one).
                for case in cases.iter() {
                    let Some(ref expr) = case.expr_opt else {
                        continue;
                    };

                    let tac_case_val = expr.to_tac(global_tracking, instructions)?;

                    instructions.push(TacInstruction::JumpIfEqual(
                        tac_condition_val.clone(),
                        tac_case_val,
                        case.label_opt.as_ref().unwrap().to_tac(),
                    ));
                }

                // The default case, if one is present, must be last, so that it's not matched before any spexific case.
                let mut is_default_case_present = false;
                for case in cases.iter() {
                    if case.expr_opt.is_none() {
                        instructions.push(TacInstruction::Jump(
                            case.label_opt.as_ref().unwrap().to_tac(),
                        ));

                        is_default_case_present = true;

                        break;
                    }
                }

                // If no default case was present, then unconditionally jump instead to the end of the switch statement.
                // That means if nothing matches, then nothing in the switch should be executed.
                if !is_default_case_present {
                    instructions.push(TacInstruction::Jump(
                        ast_body_end_label_opt.as_ref().unwrap().to_tac(),
                    ));
                }

                // The body--the actual contents of the switch statement is emitted after the jump table. If nothing
                // above matched, then nothing in the body will execute. If something matched, then it'll jump to
                // somewhere inside the body.
                body_statement.to_tac(global_tracking, instructions)?;

                instructions.push(TacInstruction::Label(
                    ast_body_end_label_opt.as_ref().unwrap().to_tac(),
                ));
            }
            AstStatementType::SwitchCase(SwitchCase { label_opt, .. }, statement) => {
                instructions.push(TacInstruction::Label(label_opt.as_ref().unwrap().to_tac()));

                statement.to_tac(global_tracking, instructions)?;
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
            writeln!(f, "{}:", label)?;
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
            AstStatementType::While(
                condition_expr,
                body_statement,
                ast_body_end_label_opt,
                ast_loop_end_label_opt,
            ) => {
                write!(f, "while (")?;

                if let Some(ast_body_end_label) = ast_body_end_label_opt {
                    write!(f, "{}: ", ast_body_end_label)?;
                }

                condition_expr.fmt_node(f, indent_levels)?;
                writeln!(f, ")")?;
                body_statement.fmt_node(f, indent_levels + 1)?;

                writeln!(f)?;
                Self::write_indent(f, indent_levels)?;

                // Print out the loop end label if it's been populated yet. Helps with seeing where break statements
                // will jump to.
                if let Some(ast_loop_end_label) = ast_loop_end_label_opt {
                    writeln!(f, "{}:", ast_loop_end_label)?;
                    Self::write_indent(f, indent_levels)?;
                }
            }
            AstStatementType::DoWhile(
                condition_expr,
                body_statement,
                ast_body_end_label_opt,
                ast_loop_end_label_opt,
            ) => {
                writeln!(f, "do")?;

                body_statement.fmt_node(f, indent_levels + 1)?;

                writeln!(f)?;
                Self::write_indent(f, indent_levels)?;

                if let Some(ast_body_end_label) = ast_body_end_label_opt {
                    write!(f, "{}: ", ast_body_end_label)?;
                }

                write!(f, "while (")?;
                condition_expr.fmt_node(f, indent_levels)?;
                write!(f, ")")?;

                // Print out the loop end label if it's been populated yet. Helps with seeing where break statements
                // will jump to.
                if let Some(ast_loop_end_label) = ast_loop_end_label_opt {
                    writeln!(f)?;
                    Self::write_indent(f, indent_levels)?;

                    writeln!(f, "{}:", ast_loop_end_label)?;
                    Self::write_indent(f, indent_levels)?;
                }
            }
            AstStatementType::Switch(
                condition_expr,
                body_statement,
                ast_body_end_label_opt,
                _cases,
            ) => {
                write!(f, "switch (")?;

                condition_expr.fmt_node(f, indent_levels)?;
                writeln!(f, ")")?;

                body_statement.fmt_node(f, indent_levels + 1)?;

                writeln!(f)?;
                Self::write_indent(f, indent_levels)?;

                // Print out the loop end label if it's been populated yet. Helps with seeing where break statements
                // will jump to.
                if let Some(ast_body_end_label) = ast_body_end_label_opt {
                    writeln!(f, "{}:", ast_body_end_label)?;
                    Self::write_indent(f, indent_levels)?;
                }
            }
            AstStatementType::SwitchCase(
                SwitchCase {
                    expr_opt,
                    label_opt,
                },
                statement,
            ) => {
                if let Some(expr) = expr_opt {
                    write!(f, "case ")?;
                    expr.fmt_node(f, indent_levels)?;
                    write!(f, ":")?;

                    if let Some(label) = label_opt {
                        write!(f, " {}:", label)?;
                    }

                    writeln!(f)?;
                } else {
                    writeln!(f, "default:")?;
                }

                statement.fmt_node(f, indent_levels)?;
            }
            AstStatementType::Goto(label) => {
                write!(f, "goto {}", label)?;
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
                    write!(f, "; {}: ", ast_body_end_label)?;
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
                    writeln!(f, "{}:", ast_loop_end_label)?;
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
                write!(f, "break to {}", label)?;
            }
            AstStatementType::Continue(Some(ref label)) => {
                write!(f, "continue to {}", label)?;
            }
            AstStatementType::Null => {}
        }

        Ok(())
    }
}

impl AstForInit {
    fn traverse_ast<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AstNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        let res = (func)(true, AstNode::ForInit(self), context);
        if context.push_error(res) {
            return;
        }

        match self {
            AstForInit::Declaration(decl) => {
                decl.traverse_ast(context, func);
            }
            AstForInit::Expression(Some(expr)) => {
                expr.traverse_ast(context, func);
            }
            AstForInit::Expression(None) => {}
        }

        let res = (func)(false, AstNode::ForInit(self), context);
        context.push_error(res);
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
    #[instrument(target = "consteval", level = "debug")]
    fn eval_i32_constant(&self) -> Result<i32, String> {
        match self {
            AstExpression::Constant(num) => Ok(*num as i32),
            AstExpression::UnaryOperator(ast_unary_op, ast_expr) => {
                let val = ast_expr.eval_i32_constant()?;
                // TODO: decent chance this isn't a robust and full-fidelity emulation of how it should act in C.
                match ast_unary_op {
                    AstUnaryOperator::Negation => Ok(val * -1),
                    AstUnaryOperator::BitwiseNot => Ok(!val),
                    AstUnaryOperator::Not => Ok(if val == 0 { 1 } else { 0 }),
                    AstUnaryOperator::PrefixIncrement
                    | AstUnaryOperator::PrefixDecrement
                    | AstUnaryOperator::PostfixIncrement
                    | AstUnaryOperator::PostfixDecrement => Err(format!(
                        "cannot evaluate constant expression with {} operator",
                        display_with(|f| ast_unary_op.fmt_node(f, 0))
                    )),
                }
            }
            AstExpression::BinaryOperator(ast_exp_left, ast_binary_op, ast_exp_right) => {
                let left_val = ast_exp_left.eval_i32_constant()?;
                let right_val = ast_exp_right.eval_i32_constant()?;

                // TODO: decent chance this isn't a robust and full-fidelity emulation of how it should act in C,
                // especially at overflow/underflow/boundaries.
                match ast_binary_op {
                    AstBinaryOperator::Add => Ok(left_val + right_val),
                    AstBinaryOperator::Subtract => Ok(left_val - right_val),
                    AstBinaryOperator::Multiply => Ok(left_val * right_val),
                    AstBinaryOperator::Divide => Ok(left_val / right_val),
                    AstBinaryOperator::Modulus => Ok(left_val % right_val),
                    AstBinaryOperator::BitwiseAnd => Ok(left_val & right_val),
                    AstBinaryOperator::BitwiseOr => Ok(left_val | right_val),
                    AstBinaryOperator::BitwiseXor => Ok(left_val ^ right_val),
                    AstBinaryOperator::ShiftLeft => Ok(left_val << right_val),
                    AstBinaryOperator::ShiftRight => Ok(left_val >> right_val),
                    AstBinaryOperator::And => Ok((left_val != 0 && right_val != 0) as i32),
                    AstBinaryOperator::Or => Ok((left_val != 0 || right_val != 0) as i32),
                    AstBinaryOperator::Equal => Ok((left_val == right_val) as i32),
                    AstBinaryOperator::NotEqual => Ok((left_val != right_val) as i32),
                    AstBinaryOperator::LessThan => Ok((left_val < right_val) as i32),
                    AstBinaryOperator::LessOrEqual => Ok((left_val <= right_val) as i32),
                    AstBinaryOperator::GreaterThan => Ok((left_val > right_val) as i32),
                    AstBinaryOperator::GreaterOrEqual => Ok((left_val >= right_val) as i32),
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
                    | AstBinaryOperator::ShiftRightAssign
                    | AstBinaryOperator::Conditional => {
                        panic!("should have been handled elsewhere")
                    }
                }
            }
            AstExpression::Var(_) => {
                Err(format!("cannot evaluate constant expression with variable"))
            }
            AstExpression::Assignment(_, _, _) => {
                Err(format!("cannot evaluate constant expression with variable"))
            }
            AstExpression::Conditional(ast_exp_left, ast_exp_middle, ast_exp_right) => {
                let left_val = ast_exp_left.eval_i32_constant()?;
                let middle_val = ast_exp_middle.eval_i32_constant()?;
                let right_val = ast_exp_right.eval_i32_constant()?;
                Ok(if left_val != 0 { middle_val } else { right_val })
            }
            AstExpression::FuncCall(_, _) => Err(format!(
                "cannot evaluate constant expression with function call"
            )),
        }
    }

    fn traverse_ast<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AstNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        let res = (func)(true, AstNode::Expression(self), context);
        if context.push_error(res) {
            return;
        }

        match self {
            AstExpression::Constant(_num) => {}
            AstExpression::UnaryOperator(_ast_unary_op, ast_exp_inner) => {
                ast_exp_inner.traverse_ast(context, func);
            }
            AstExpression::BinaryOperator(ast_exp_left, _binary_op, ast_exp_right) => {
                ast_exp_left.traverse_ast(context, func);
                ast_exp_right.traverse_ast(context, func);
            }
            AstExpression::Var(_ast_ident) => {}
            AstExpression::Assignment(ast_exp_left, _operator, ast_exp_right) => {
                ast_exp_left.traverse_ast(context, func);
                ast_exp_right.traverse_ast(context, func);
            }
            AstExpression::Conditional(ast_exp_left, ast_exp_middle, ast_exp_right) => {
                ast_exp_left.traverse_ast(context, func);
                ast_exp_middle.traverse_ast(context, func);
                ast_exp_right.traverse_ast(context, func);
            }
            AstExpression::FuncCall(_identifier, ref mut arguments) => {
                for arg in arguments.iter_mut() {
                    arg.traverse_ast(context, func);
                }
            }
        }

        let res = (func)(false, AstNode::Expression(self), context);
        context.push_error(res);
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
            AstExpression::FuncCall(identifier, arguments) => {
                let mut tac_args = vec![];
                for ast_arg in arguments.iter() {
                    tac_args.push(ast_arg.to_tac(global_tracking, instructions)?);
                }

                let tempvar = global_tracking.allocate_temporary();
                instructions.push(TacInstruction::FuncCall(
                    identifier.to_tac_label(),
                    tac_args,
                    tempvar.clone(),
                ));

                TacVal::Var(tempvar)
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
            AstExpression::FuncCall(identifier, arguments) => {
                write!(f, "CALL {}(", identifier)?;
                for (i, arg) in arguments.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    arg.fmt_node(f, 0)?;
                }
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
        let Some(ref tac_body) = self.body_opt else {
            return Ok(AsmFunction::new(&self.name, None));
        };

        let mut frame = FuncStackFrame::new();

        let mut body = Vec::new();

        // In our generated code, move any arguments passed in registers into their home area in memory. This makes it
        // easy to reference everything uniformly by stack offsets instead of having to track when a value's storage
        // location moves into or out of memory.
        for (i, param) in self.parameters.iter().enumerate() {
            let location = frame.create_parameter_location(i, param);

            if let Some(reg) = PARAMETER_REG_MAPPING.get(i) {
                body.push(AsmInstruction::Mov(
                    AsmVal::Loc(AsmLocation::Reg(reg)),
                    location.clone(),
                ));
            }
        }

        // In order to allocate sufficient stack space in this frame, iterate over all function calls and find out the
        // max number of arguments passed to a child function. We'll need to reserve enough space to store all those
        // arguments.
        let max_arg_count = tac_body
            .iter()
            .map(|instruction| {
                if let TacInstruction::FuncCall(_, arguments, _) = instruction {
                    arguments.len() as u32
                } else {
                    0
                }
            })
            .max()
            .unwrap_or(0);

        // We haven't yet stored this arguments information in the frame tracking.
        assert!(frame.arguments_size == 0);

        if max_arg_count > 0 {
            // If any arguments were passed, then the entire set of "homed" parameters are allocated on the stack, even
            // if fewer were used. This is x64 ABI. I don't make the rules.
            let args_count_to_reserve =
                std::cmp::max(max_arg_count, PARAMETER_REG_MAPPING.len() as u32);
            frame.arguments_size = args_count_to_reserve * STACK_PARAMETER_SIZE;
        }

        for instruction in tac_body.iter() {
            instruction.to_asm(&mut body)?;
        }

        let mut errors = Vec::new();
        for inst in body.iter_mut() {
            inst.traverse_asm(
                &mut TraversalContext::new_blank(Some(&mut errors)),
                &mut |is_enter, node, _context| {
                    if !is_enter {
                        return Ok(());
                    }

                    match node {
                        AsmNode::Loc(loc) => loc.resolve_pseudoregister(&mut frame),
                    }
                },
            );

            if !errors.is_empty() {
                return Err(errors.swap_remove(0));
            }
        }

        debug!(target: "asm", func = self.name, locals_size = frame.locals_size, arguments_size = frame.arguments_size, "size before padding");

        // The x64 ABI requires 16-byte alignment of the stack prior to any "call" instruction. If padding is needed,
        // add it to the arguments size, i.e. after local variables. When setting up the call, argument locations are
        // addressed relative to RSP rather than RBP, so this extra padding will effectively be between local variables
        // and arguments.
        //
        // Important to remember that even though RSP is 16-byte aligned when issuing the call instruction, RSP is off
        // by one pointer size when entering the new function, because call pushes the return address on the stack, so
        // include that in the frame size to figure out the right padding to add.
        if frame.arguments_size > 0 {
            // This is the size of the locals size and arguments size and the return address.
            let total_frame_size = frame.size() + POINTER_SIZE;

            // Pad the total frame size to the required alignment and add to the arguments size whatever padding is
            // needed to achieve that correct total frame size.
            frame.arguments_size +=
                pad_to_alignment(total_frame_size, FUNCTION_CALL_ALIGNMENT) - total_frame_size;
        }

        debug!(target: "asm", func = self.name, locals_size = frame.locals_size, arguments_size = frame.arguments_size, "size after padding");

        // Allocate the stack frame's size at the beginning of the function body.
        body.insert(0, AsmInstruction::AllocateStack(frame.size()));

        // Now that the stack frame size is known, in any place that has a Ret instruction, fill it in.
        for inst in body.iter_mut() {
            if let AsmInstruction::Ret(size) = inst {
                *size = frame.size();
            }
        }

        let mut asm_func = AsmFunction::new(&self.name, Some(body));

        debug!(target: "asm", asm = %DisplayFmtNode(&asm_func), "before convert rbp to rsp offset");

        // Also now that the frame size is known, all RBP-relative offsets can be converted to RSP-relative.
        for inst in asm_func.body_opt.as_mut().unwrap().iter_mut() {
            assert!(errors.is_empty());
            inst.traverse_asm(
                &mut TraversalContext::new_blank(Some(&mut errors)),
                &mut |is_enter, node, _context| {
                    if !is_enter {
                        return Ok(());
                    }

                    match node {
                        AsmNode::Loc(loc) => {
                            loc.convert_to_rsp_offset(&mut frame);
                        }
                    }

                    Ok(())
                },
            );

            if !errors.is_empty() {
                return Err(errors.swap_remove(0));
            }
        }

        Ok(asm_func)
    }
}

impl FmtNode for TacFunction {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        write!(f, "FUNC {} (", self.name)?;
        Self::fmt_nodelist(f, self.parameters.iter(), ", ", indent_levels)?;
        writeln!(f, ")")?;

        if let Some(ref body) = self.body_opt {
            Self::fmt_nodelist(f, body.iter(), "\n", indent_levels + 1)?;
        }

        Ok(())
    }
}

impl TacLabel {
    fn to_asm(&self) -> Result<AsmLabel, String> {
        Ok(AsmLabel::new(&self.0))
    }
}

impl fmt::Display for TacLabel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
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
            TacInstruction::CopyVal(src_val, dest_var) => {
                func_body.push(AsmInstruction::Mov(src_val.to_asm()?, dest_var.to_asm()?));
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
            TacInstruction::JumpIfEqual(val1, val2, label) => {
                func_body.push(AsmInstruction::Cmp(val1.to_asm()?, val2.to_asm()?));
                func_body.push(AsmInstruction::JmpCc(AsmCondCode::E, label.to_asm()?));
            }
            TacInstruction::Label(label) => {
                func_body.push(AsmInstruction::Label(label.to_asm()?));
            }
            TacInstruction::FuncCall(label, arguments, dest_var) => {
                // Each argument is moved into a pseudoregister tagged as belonging to an argument and numbered so that
                // it can eventually be resolved into the correct slot near the end of the frame. Because arguments are
                // pushed onto the stack in reverse order, arg 0 will be closest to RSP.
                for (i, arg) in arguments.iter().enumerate() {
                    func_body.push(AsmInstruction::Mov(
                        arg.to_asm()?,
                        AsmLocation::PseudoArgReg(i as u32),
                    ));
                }

                func_body.push(AsmInstruction::Call(mangle_function_name(&label.0)));

                // Function return value is stored in eax. Move it to the expected destination location.
                func_body.push(AsmInstruction::Mov(
                    AsmVal::Loc(AsmLocation::Reg("eax")),
                    dest_var.to_asm()?,
                ));
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
                write!(f, "jump :{}", label)?;
            }
            TacInstruction::JumpIfZero(val, label) => {
                write!(f, "jump :{} if ", label)?;
                val.fmt_node(f, 0)?;
                write!(f, " == 0")?;
            }
            TacInstruction::JumpIfNotZero(val, label) => {
                write!(f, "jump :{} if ", label)?;
                val.fmt_node(f, 0)?;
                write!(f, " != 0")?;
            }
            TacInstruction::JumpIfEqual(val1, val2, label) => {
                write!(f, "jump :{} if ", label)?;
                val1.fmt_node(f, 0)?;
                write!(f, " == ")?;
                val2.fmt_node(f, 0)?;
            }
            TacInstruction::Label(label) => {
                writeln!(f)?;
                write!(f, "{}:", label)?;
            }
            TacInstruction::FuncCall(label, arguments, dest_var) => {
                dest_var.fmt_node(f, 0)?;
                write!(f, " = call :{} (", label)?;
                let mut first = true;
                for arg in arguments.iter() {
                    if !first {
                        write!(f, ", ")?;
                    }

                    arg.fmt_node(f, 0)?;
                    first = false;
                }
                write!(f, ")")?;
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
                writeln!(
                    f,
                    r"
INCLUDELIB msvcrt.lib
                    "
                )?;

                writeln!(f, "; ALIAS declarations in this section allow for function names with keywords to be used as identifiers")?;

                // Only emit each function once. Let definitions take precedence over declarations.
                let mut func_names = HashSet::new();
                for func in self
                    .functions
                    .iter()
                    .filter(|f| f.is_definition())
                    .chain(self.functions.iter().filter(|f| !f.is_definition()))
                {
                    if func_names.insert(&func.name) {
                        func.emit_declaration(f)?;
                    }
                }

                writeln!(
                    f,
                    r"
.DATA
.CODE
                         "
                )?;

                for func in self.functions.iter().filter(|f| f.is_definition()) {
                    func.emit_code(f)?;
                }

                writeln!(f)?;
                writeln!(f, "; NOKEYWORD is used here to allow us to import or export the original function name, when that name might be a MASM keyword.")?;

                // Only emit each function once. Let definitions take precedence over declarations.
                func_names.clear();
                for func in self
                    .functions
                    .iter()
                    .filter(|f| f.is_definition())
                    .chain(self.functions.iter().filter(|f| !f.is_definition()))
                {
                    if func_names.insert(&func.name) {
                        func.emit_footer(f)?;
                    }
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
    fn new(name: &str, body_opt: Option<Vec<AsmInstruction>>) -> Self {
        Self {
            name: String::from(name),
            mangled_name: mangle_function_name(name),
            body_opt,
        }
    }

    fn is_definition(&self) -> bool {
        self.body_opt.is_some()
    }

    fn has_mangled_name(&self) -> bool {
        self.name != self.mangled_name
    }

    fn finalize(&mut self) -> Result<(), String> {
        let Some(ref mut body) = self.body_opt else {
            return Ok(());
        };

        let mut i;

        // Shift left and shift right only allow immedate or CL (that's 8-bit ecx) register as the right hand side. If
        // the rhs isn't in there already, move it first.
        i = 0;
        while i < body.len() {
            if let AsmInstruction::BinaryOp(
                AsmBinaryOperator::Shl | AsmBinaryOperator::Sar,
                ref mut src_val,
                _dest_loc,
            ) = &mut body[i]
            {
                if src_val.get_base_reg_name() != Some("cx") {
                    let real_src_val = src_val.clone();
                    *src_val = AsmVal::Loc(AsmLocation::Reg("cl"));

                    body.insert(
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
        while i < body.len() {
            if let AsmInstruction::Mov(
                ref _src_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
                ref mut dest_loc @ AsmLocation::RspOffset(_, _),
            ) = &mut body[i]
            {
                let real_dest = dest_loc.clone();
                *dest_loc = AsmLocation::Reg("r10d");

                body.insert(
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
        while i < body.len() {
            if let AsmInstruction::BinaryOp(
                AsmBinaryOperator::Imul,
                _src_val,
                ref mut dest_loc @ AsmLocation::RspOffset(_, _),
            ) = &mut body[i]
            {
                let real_dest = dest_loc.clone();

                // Rewrite the multiply instruction itself to operate against a temporary register instead of a
                // memory address.
                *dest_loc = AsmLocation::Reg("r11d");

                // Insert a mov before the multiply, to put the destination value in the temporary register.
                body.insert(
                    i,
                    AsmInstruction::Mov(AsmVal::Loc(real_dest.clone()), AsmLocation::Reg("r11d")),
                );

                // Insert a mov instruction after the multiply, to put the destination value into the intended
                // memory address.
                body.insert(
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
        while i < body.len() {
            if let AsmInstruction::Cmp(_src_val, ref mut dest_val @ AsmVal::Imm(_)) = &mut body[i] {
                let real_dest_val = dest_val.clone();
                *dest_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                body.insert(
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
        while i < body.len() {
            if let AsmInstruction::Cmp(
                ref mut src_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
                ref _dest_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
            ) = &mut body[i]
            {
                let real_src_val = src_val.clone();
                *src_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                body.insert(
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
        while i < body.len() {
            if let AsmInstruction::BinaryOp(
                _binary_op,
                ref mut src_val @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
                ref _dest_loc @ AsmLocation::RspOffset(_, _),
            ) = &mut body[i]
            {
                let real_src_val = src_val.clone();
                *src_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                body.insert(
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
        while i < body.len() {
            if let AsmInstruction::Idiv(ref mut denom_val @ AsmVal::Imm(_)) = &mut body[i] {
                let real_denom_val = denom_val.clone();
                *denom_val = AsmVal::Loc(AsmLocation::Reg("r10d"));

                // Insert a mov before this idiv to put its immediate value in a register.
                body.insert(
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

    fn emit_declaration(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The ALIAS <new_name> = <existing_name> directive creates a new entry with "new_name", and requires that you
        // have a definition or EXTERN entry for "existing_name" elsewhere in the ASM listing. For functions that don't
        // have any name mangling, of course this doesn't have any effect and in fact causes a multiple declarations
        // error.
        if self.has_mangled_name() {
            // If we have a definition, it is emitted later in the listing using its mangled name so that it can be
            // easily referenced within this listing. This is necessary especially if the function name is a built-in,
            // reserved word like "add", "sub", "mov", etc..
            //
            // If we have only a declaration, then there will be an EXTERN directive later in the ASM listing that
            // brings in the function by its original name from some other compilation unit. But we might not be able
            // to refer to that original name, if it's one of those reserved words. So set up an alias with a mangled
            // name that we can use to refer to it in this listing.
            if !self.is_definition() {
                writeln!(f, "ALIAS <{}> = <{}>", self.mangled_name, self.name)?;
            }
        }

        Ok(())
    }

    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Some(ref body) = self.body_opt else {
            unreachable!("should not call emit_code on a declaration");
        };

        write!(f, "{} PROC", self.mangled_name)?;

        // For any functions that have a mangled name, we don't want to expose the mangled name to allow linking against
        // it from other compilation units. We'll expose the correct name a different way. See `emit_footer` for that.
        if self.has_mangled_name() {
            write!(f, " PRIVATE")?;
        }

        writeln!(f)?;

        for inst in body.iter() {
            inst.emit_code(f)?;
        }

        writeln!(f, "{} ENDP", self.mangled_name)
    }

    fn emit_footer(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // If we are defining a new function in this listing, we did so using a mangled name to avoid clashing with
        // reserved MASM keywords. Because we mangled the name, we also declared it as PRIVATE (a.k.a. static) so that
        // other compilation units can't link against it. But we do want the original name available to be linked
        // against, so make it not a keyword and assign a new label to it, then publicize that label. That makes it show
        // up properly in the object file.
        if self.is_definition() {
            if self.has_mangled_name() {
                writeln!(f, "OPTION NOKEYWORD: <{}>", self.name)?;
                writeln!(f, "{} = {}", self.name, self.mangled_name)?;
                writeln!(f, "PUBLIC {}", self.name)?;
            }

        // If we were bringing in a function, it needs an EXTERN declaration. Because it might be a reserved MASM
        // keyword, remove the name from the keywords list.
        } else {
            writeln!(f, "OPTION NOKEYWORD: <{}>", self.name)?;
            writeln!(f, "EXTERN {}:PROC", self.name)?;
        }

        Ok(())
    }
}

impl FmtNode for AsmFunction {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        writeln!(f, "FUNC {}", self.name)?;

        if let Some(ref body) = self.body_opt {
            Self::fmt_nodelist(f, body.iter(), "\n", 1)?;
        }

        Ok(())
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
    fn traverse_asm<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AsmNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                src_val.traverse_asm(context, func);
                dest_loc.traverse_asm(context, func);
            }
            AsmInstruction::UnaryOp(_, dest_loc) => {
                dest_loc.traverse_asm(context, func);
            }
            AsmInstruction::BinaryOp(_, src_val, dest_loc) => {
                src_val.traverse_asm(context, func);
                dest_loc.traverse_asm(context, func);
            }
            AsmInstruction::Idiv(denom_val) => {
                denom_val.traverse_asm(context, func);
            }
            AsmInstruction::Cdq => {}
            AsmInstruction::AllocateStack(_) => {}
            AsmInstruction::Ret(_) => {}
            AsmInstruction::Cmp(src1_val, src2_val) => {
                src1_val.traverse_asm(context, func);
                src2_val.traverse_asm(context, func);
            }
            AsmInstruction::SetCc(_cond_code, dest_loc) => {
                dest_loc.traverse_asm(context, func);
            }
            AsmInstruction::Jmp(_label) => {}
            AsmInstruction::JmpCc(_cond_code, _label) => {}
            AsmInstruction::Label(_label) => {}
            AsmInstruction::Call(_func_name) => {}
        }
    }

    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                format_code_and_comment(
                    f,
                    |f| {
                        write!(f, "mov ")?;
                        dest_loc.emit_code(f, VARIABLE_SIZE as u8)?;
                        write!(f, ",")?;
                        src_val.emit_code(f, VARIABLE_SIZE as u8)
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
                        dest_loc.emit_code(f, VARIABLE_SIZE as u8)
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
                    _ => VARIABLE_SIZE as u8,
                };

                format_code_and_comment(
                    f,
                    |f| {
                        binary_op.emit_code(f)?;
                        write!(f, " ")?;
                        dest_loc.emit_code(f, VARIABLE_SIZE as u8)?;
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
                        denom_val.emit_code(f, VARIABLE_SIZE as u8)
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
                    |f| write!(f, "stack_dealloc {} bytes", size),
                )?;

                writeln!(f, "    ret")?;
            }
            AsmInstruction::Cmp(src1_val, src2_val) => {
                format_code_and_comment(
                    f,
                    |f| {
                        write!(f, "cmp ")?;
                        src2_val.emit_code(f, VARIABLE_SIZE as u8)?;
                        write!(f, ",")?;
                        src1_val.emit_code(f, VARIABLE_SIZE as u8)
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
            AsmInstruction::Call(func_name) => {
                writeln!(f, "    call {}", func_name)?;
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
            AsmInstruction::Call(func_name) => {
                write!(f, "Call {}", func_name)?;
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

    fn traverse_asm<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AsmNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        if let AsmVal::Loc(loc) = self {
            loc.traverse_asm(context, func);
        }
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
    fn traverse_asm<TContext, FuncVisit>(
        &mut self,
        context: &mut TraversalContext<TContext>,
        func: &mut FuncVisit,
    ) where
        FuncVisit: FnMut(bool, AsmNode, &mut TraversalContext<TContext>) -> Result<(), String>,
    {
        let res = (func)(true, AsmNode::Loc(self), context);
        if context.push_error(res) {
            return;
        }

        let res = (func)(false, AsmNode::Loc(self), context);
        context.push_error(res);
    }

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
        match self {
            AsmLocation::PseudoReg(psr) => {
                *self = frame.create_or_get_location(&psr)?;
            }

            // This is a pseudoregister that represents an argument rather than a local variable. arg_num 0 is the first
            // in the parameter list and therefore closest to RSP.
            AsmLocation::PseudoArgReg(arg_num) => {
                // Some arguments are passed in registers, so first check if this is one of those.
                *self = if let Some(reg) = PARAMETER_REG_MAPPING.get(*arg_num as usize) {
                    AsmLocation::Reg(reg)
                } else {
                    // Beyond the register arguments, the rest are stored on the stack. The stack grows
                    // downwards, meaning that allocating space on the stack means making RSP have a numerically
                    // smaller value. So the first argument (0th) would be written at bytes [RSP + 0, RSP + 8), i.e.
                    // RSP+0.
                    AsmLocation::RspOffset(
                        *arg_num * STACK_PARAMETER_SIZE,
                        format!("arg_{}", *arg_num),
                    )
                };
            }
            _ => {}
        }

        Ok(())
    }

    fn convert_to_rsp_offset(&mut self, frame: &FuncStackFrame) {
        if let AsmLocation::RbpOffset(rbp_offset, name) = self {
            *self =
                AsmLocation::RspOffset(frame.rsp_offset_from_rbp_offset(*rbp_offset), name.clone());
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
            AsmLocation::PseudoArgReg(num) => {
                write!(f, "arg_{}", num)?;
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

impl SymbolTable {
    fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }

    fn add_function(&mut self, function: &AstFunction) -> Result<(), String> {
        if let Some(symbol) = self.symbols.get(&function.name) {
            return match symbol {
                Symbol::Var(_) => Err(format!(
                    "function {} previously defined as a variable in this scope",
                    function.name
                )),
                Symbol::Function(param_count, _) => {
                    if *param_count != function.parameters.len() {
                        Err(format!("function {} declared with {} parameters, but previously declared with {} parameters", function.name, function.parameters.len(), param_count))
                    } else {
                        Ok(())
                    }
                }
            };
        }

        let old_value = self.symbols.insert(
            function.name.clone(),
            Symbol::Function(function.parameters.len(), function.is_definition()),
        );
        assert!(old_value.is_none());

        Ok(())
    }

    fn lookup_symbol(&self, identifier: &AstIdentifier) -> Result<Option<&Symbol>, String> {
        if let Some(symbol) = self.symbols.get(identifier) {
            if let Symbol::Function(_, _) = symbol {
                Ok(Some(symbol))
            } else {
                Err(format!(
                    "identifier {} declared as not a function",
                    identifier
                ))
            }
        } else {
            Ok(None)
        }
    }

    /// Adds a variable to the map and returns a unique, mangled identifier to refer to the variable with.
    fn add_variable(
        &mut self,
        global_tracking: &mut GlobalTracking,
        ast_identifier: &AstIdentifier,
    ) -> Result<AstIdentifier, String> {
        if let Some(symbol) = self.symbols.get(&ast_identifier) {
            match symbol {
                Symbol::Var(_) => {
                    return Err(format!("duplicate variable declaration {}", ast_identifier));
                }
                Symbol::Function(_, _) => {
                    return Err(format!(
                        "variable {} previously defined as a function in this scope",
                        ast_identifier
                    ));
                }
            }
        }

        let temp_var = global_tracking.create_temporary_ast_ident(&ast_identifier);
        let old_value = self
            .symbols
            .insert(ast_identifier.clone(), Symbol::Var(temp_var.clone()));
        assert!(old_value.is_none());

        Ok(temp_var)
    }
}

impl GlobalTracking {
    fn new() -> Self {
        Self {
            next_temporary_id: 0,
            next_label_id: 0,
            symbols: SymbolTable::new(),
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
            identifier
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
            return Err(format!("duplicate label declaration \"{}\"", label));
        }

        let temp_label = AstLabel(global_tracking.allocate_label(&format!("0userlabel_{}", label)));
        let old_value = self.labels.insert(label.clone(), temp_label.clone());
        assert!(old_value.is_none());

        Ok(temp_label)
    }

    fn resolve_label(&self, label: &AstLabel) -> Result<AstLabel, String> {
        // If the label was found in this scope, return its mangled version.
        if let Some(temp_label) = self.labels.get(label) {
            Ok(temp_label.clone())
        } else {
            Err(format!("label '{}' not found", label))
        }
    }
}

// Example stack. Caller passed us 6 parameters. This function uses 3 local variables and itself calls another function
// that takes 5 parameters.
//
// Name            |size | RBP offset| RSP offset
// ----------------------------------------------
// param 5         | 8   | RBP + 48  | RSP + 104
// param 4         | 8   | RBP + 40  | RSP + 96
// param 3         | 8   | RBP + 32  | RSP + 88
// param 2         | 8   | RBP + 24  | RSP + 80
// param 1         | 8   | RBP + 16  | RSP + 72
// param 0         | 8   | RBP + 8   | RSP + 64
// return addr     | 8   | RBP + 0   | RSP + 56
// local 1         | 4   | RBP - 4   | RSP + 52
// local 2         | 4   | RBP - 8   | RSP + 48
// local 3         | 4   | RBP - 12  | RSP + 44
// padding         | 4   |           | RSP + 40
// arg 4           | 8   |           | RSP + 32
// arg 3           | 8   |           | RSP + 24
// arg 2           | 8   |           | RSP + 16
// arg 1           | 8   |           | RSP + 8
// arg 0           | 8   |           | RSP + 0
//
// Locals size: 3 * 4 = 12
// Arguments size: 5 * 8 = 40
// Frame size including return value: 12 + 40 + 8 = 60
// Not 16-byte aligned, so need to add 4 bytes of padding
// Total frame size: 64 bytes
//
// The stack grows "downwards", meaning each new function call's data is at a numerically smaller address than the
// previous one. So the "top" of the stack has the numerically smallest address.
//
// RBP is the "base pointer", and means the bottom of the current function's stack area. This is the location where the
// call instruction that led to this function's execution stored the return value.
//
// Parameters passed to this function live in memory owned by the caller, so they are at positive offsets from RBP,
// right after the return address, so starting at RBP+8.
//
// The first local variable for the current function is the first thing within this stack frame. Since the stack grows
// downwards, that means negative offsets from RBP. Each variable right now uses 4 bytes, putting the first local
// variable in RBP-4. In the example, the last local variable referenced by the current stack frame is at RBP-12,
// because that's the "top" of this frame's local variables.
//
// RSP is the "stack pointer" and points to the top of the stack. If this function doesn't call any other functions,
// then RSP would point to the last local variable. If this function does call another function, then RSP points to the
// first argument. Subsequent arguments live at positive RSP offsets near the top of the stack, with the first
// argument being at RSP+0 and each subsequent one at an 8-byte slot farther from the top, e.g. RSP+8, RSP+16, etc..
//
// When issuing a call instruction, RSP must be 16-byte aligned, so if any padding is needed, it takes the next space
// after the last stack argument.
//
// The x64 ABI says that the first 4 parameters are passed in registers RCX, RDX, R8, and R9, respectively, but space
// must also be allocated on the stack in case the callee wants to take the address of one of those parameters. That
// reserved space is called the "home area".
impl FuncStackFrame {
    fn new() -> Self {
        Self {
            names: HashMap::new(),
            locals_size: 0,
            arguments_size: 0,
        }
    }

    // Store a mapping from a parameter name to its location in this stack frame.
    fn create_parameter_location(&mut self, param_num: usize, param: &TacVar) -> AsmLocation {
        // RBP points to the return address, so to get to the parameters, we have to skip over that. Then the very next
        // slot is the first parameter.
        let rbp_offset = (POINTER_SIZE + (param_num as u32 * STACK_PARAMETER_SIZE)) as i32;

        // Parameters are allocated by the caller (i.e. live in the caller's stack frame), so all parameters must be at
        // positive RBP offsets, or else they'd actually be in this stack frame. Moreover, they need to be passt the
        // return address.
        assert!(rbp_offset >= POINTER_SIZE as i32);

        let res = self.names.insert(param.0.clone(), rbp_offset);
        assert!(res.is_none());

        AsmLocation::RbpOffset(rbp_offset, param.0.clone())
    }

    fn create_or_get_location(&mut self, name: &str) -> Result<AsmLocation, String> {
        if let Some(offset) = self.names.get(name) {
            Ok(AsmLocation::RbpOffset(*offset, String::from(name)))
        } else {
            // For now we are only storing VARIABLE_SIZE values. The maximum offset from the base pointer that is used
            // in this stack frame grows by one slot.
            self.locals_size += VARIABLE_SIZE;

            // And the offset of this location is at the new edge.
            let offset = -(self.locals_size as i32);

            let res = self.names.insert(String::from(name), offset);
            assert!(res.is_none());

            Ok(AsmLocation::RbpOffset(offset, String::from(name)))
        }
    }

    fn rsp_offset_from_rbp_offset(&self, rbp_offset: i32) -> u32 {
        let rsp_offset = self.size() as i32 + rbp_offset;
        assert!(rsp_offset >= 0);
        rsp_offset as u32
    }

    fn size(&self) -> u32 {
        self.locals_size + self.arguments_size
    }
}

impl<'e> TraversalContext<'e, ()> {
    fn new_blank(errors_opt: Option<&'e mut Vec<String>>) -> TraversalContext<'e, ()> {
        Self::new(None, errors_opt)
    }
}

impl<'e, T> TraversalContext<'e, T> {
    fn new(initial: Option<T>, errors_opt: Option<&'e mut Vec<String>>) -> Self {
        Self {
            inner: initial.map(Box::new),
            errors_len: errors_opt.as_deref().map(Vec::len).unwrap_or(0),
            errors_opt,
        }
    }

    /// Enters a deeper level of the context.
    fn nest(&mut self, mut val: T)
    where
        T: Nestable,
    {
        // The new value that is becoming the current context gets its parent reference set to the current context.
        *val.get_parent_opt_mut() = self.inner.take();

        // The current context is set to the new context.
        self.inner = Some(Box::new(val));
    }

    /// Exits a level of nesting.
    fn unnest(&mut self) -> Box<T>
    where
        T: Nestable,
    {
        // Extract the current context, which is becoming obsolete.
        let mut old_this = self.inner.take();

        // Grab the parent context out of the current context and set it to be the new current context.
        self.inner = old_this.as_mut().unwrap().get_parent_opt_mut().take();

        // Return the former current context (with no parent link, of course).
        old_this.unwrap()
    }

    /// Get a reference to the current context.
    fn get(&self) -> Option<&T> {
        self.inner.as_deref()
    }

    /// Get a mutable reference to the current context.
    fn get_mut(&mut self) -> Option<&mut T> {
        self.inner.as_deref_mut()
    }

    fn push_error<TError>(&mut self, result: Result<TError, String>) -> bool {
        if let Some(ref mut errors) = self.errors_opt {
            push_error(result, errors)
        } else {
            false
        }
    }

    fn had_errors(&self) -> bool {
        self.errors_opt.as_deref().map(Vec::len).unwrap_or(0) != self.errors_len
    }

    fn is_clean_traversal(&self) -> bool {
        // Either this object has gotten unnested to its original state or it encountered an error.
        self.inner.is_none() || self.had_errors()
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
            debug!(target: "lex", "match skipped {:?}: {}, {}", r, range.start, range.end);
            return Ok(("", input.split_at(range.end).1));
        }
    }

    for r in TOKEN_REGEXES.iter() {
        if let Some(mat) = r.find(input) {
            let range = mat.range();
            debug!(target: "lex", "match {:?}: {}, {}", r, range.start, range.end);
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
                debug!(target: "lex", "[{}], [{}]", split.0, split.1);
                info!(target: "lex", "token: {}", split.0);
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

fn try_parse_function<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<Option<AstFunction>, String> {
    let mut tokens = original_tokens.clone();

    if !tokens.try_consume_expected_next_token("int") {
        return Ok(None);
    }

    let Some(name) = tokens.try_consume_identifier() else {
        return Ok(None);
    };

    if !tokens.try_consume_expected_next_token("(") {
        return Ok(None);
    }

    // After the opening parenthesis for parameters, we're committed to parsing a function.
    *original_tokens = tokens;

    let mut parameters = vec![];

    // If the first parameter is void type, then it must be the only parameter. Otherwise, parse regular parameters.
    if !original_tokens.try_consume_expected_next_token("void") {
        while let Some(parameter_name) =
            try_parse_function_parameter(original_tokens, parameters.len() == 0)?
        {
            parameters.push(parameter_name);
        }
    }

    original_tokens.consume_expected_next_token(")")?;

    // This is either a function definition (body is present) or a function declaration (body absent, but semicolon
    // instead).
    let body_opt = try_parse_block(original_tokens)?;
    if body_opt.is_none() {
        original_tokens.consume_expected_next_token(";")?;
    }

    Ok(Some(AstFunction {
        name: AstIdentifier(String::from(name)),
        parameters,

        // We parsed the body as an AstBlock, but it's stored as a list of block items, because the body doesn't
        // introduce a new scope beyond the function itlself.
        body_opt: body_opt.map(|v| v.0),
    }))
}

#[instrument(target = "parse", level = "debug")]
fn try_parse_function_parameter<'i, 't>(
    tokens: &mut Tokens<'i, 't>,
    is_first_parameter: bool,
) -> Result<Option<AstIdentifier>, String> {
    if !is_first_parameter {
        if !tokens.try_consume_expected_next_token(",") {
            return Ok(None);
        }
    }

    if !tokens.try_consume_expected_next_token("int") {
        return Ok(None);
    }

    tokens
        .consume_identifier()
        .map(|i| Some(AstIdentifier(String::from(i))))
}

fn parse_block<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<AstBlock, String> {
    try_parse_block(tokens)?.ok_or(format!("required block not found"))
}

#[instrument(target = "parse", level = "debug")]
fn try_parse_block<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<Option<AstBlock>, String> {
    if !tokens.try_consume_expected_next_token("{") {
        return Ok(None);
    }

    let mut body = vec![];

    // Keep trying to parse block items until the end of the block is found.
    while !tokens.try_consume_expected_next_token("}") {
        body.push(parse_block_item(tokens)?);
    }

    Ok(Some(AstBlock(body)))
}

#[instrument(target = "parse", level = "debug")]
fn parse_block_item<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<AstBlockItem, String> {
    // Try parsing as a function first, because the parameter list makes it easier to differentiate from an identifier.
    if let Some(func) = try_parse_function(tokens)? {
        Ok(AstBlockItem::FuncDeclaration(func))
    } else if let Some(declaration) = try_parse_var_declaration(tokens)? {
        Ok(AstBlockItem::VarDeclaration(declaration))
    } else {
        Ok(AstBlockItem::Statement(parse_statement(tokens)?))
    }
}

#[instrument(target = "parse", level = "debug")]
fn parse_statement<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<AstStatement, String> {
    let labels = parse_statement_labels(tokens)?;

    let statement_type = if tokens.try_consume_expected_next_token("return") {
        let st = AstStatementType::Return(parse_expression(tokens)?);
        tokens.consume_expected_next_token(";")?;
        st
    } else if tokens.try_consume_expected_next_token(";") {
        AstStatementType::Null
    } else if tokens.try_consume_expected_next_token("if") {
        tokens.consume_expected_next_token("(")?;
        let condition_expr = parse_expression(tokens)?;
        tokens.consume_expected_next_token(")")?;
        let then_statement = parse_statement(tokens)?;

        let else_statement_opt = if tokens.try_consume_expected_next_token("else") {
            Some(Box::new(parse_statement(tokens)?))
        } else {
            None
        };

        AstStatementType::If(condition_expr, Box::new(then_statement), else_statement_opt)
    } else if tokens.try_consume_expected_next_token("goto") {
        let label_name = tokens.consume_identifier()?;
        tokens.consume_expected_next_token(";")?;
        AstStatementType::Goto(AstLabel(String::from(label_name)))
    } else if let Some(for_statement) = try_parse_for_statement(tokens)? {
        for_statement
    } else if tokens.try_consume_expected_next_token("while") {
        tokens.consume_expected_next_token("(")?;
        let condition_expr = parse_expression(tokens)?;
        tokens.consume_expected_next_token(")")?;
        let body_statement = parse_statement(tokens)?;

        AstStatementType::While(condition_expr, Box::new(body_statement), None, None)
    } else if tokens.try_consume_expected_next_token("do") {
        let body_statement = parse_statement(tokens)?;
        tokens.consume_expected_next_token("while")?;
        tokens.consume_expected_next_token("(")?;
        let condition_expr = parse_expression(tokens)?;
        tokens.consume_expected_next_token(")")?;
        tokens.consume_expected_next_token(";")?;
        AstStatementType::DoWhile(condition_expr, Box::new(body_statement), None, None)
    } else if let Some(block) = try_parse_block(tokens)? {
        AstStatementType::Compound(Box::new(block))
    } else if tokens.try_consume_expected_next_token("break") {
        let st = AstStatementType::Break(None);
        tokens.consume_expected_next_token(";")?;
        st
    } else if tokens.try_consume_expected_next_token("continue") {
        let st = AstStatementType::Continue(None);
        tokens.consume_expected_next_token(";")?;
        st
    } else if tokens.try_consume_expected_next_token("switch") {
        tokens.consume_expected_next_token("(")?;
        let condition_expr = parse_expression(tokens)?;
        tokens.consume_expected_next_token(")")?;
        let body_statement = parse_statement(tokens)?;

        AstStatementType::Switch(condition_expr, Box::new(body_statement), None, Vec::new())
    } else if tokens.try_consume_expected_next_token("case") {
        let expr = parse_expression(tokens)?;
        tokens.consume_expected_next_token(":")?;

        AstStatementType::SwitchCase(
            SwitchCase {
                expr_opt: Some(expr),
                label_opt: None,
            },
            Box::new(parse_statement(tokens)?),
        )
    } else if tokens.try_consume_expected_next_token("default") {
        tokens.consume_expected_next_token(":")?;

        AstStatementType::SwitchCase(
            SwitchCase {
                expr_opt: None,
                label_opt: None,
            },
            Box::new(parse_statement(tokens)?),
        )
    } else {
        let st = AstStatementType::Expr(parse_expression(tokens)?);
        tokens.consume_expected_next_token(";")?;
        st
    };

    let st = AstStatement::new(statement_type, labels);
    debug!(target: "parse", st = %DisplayFmtNode(&st), "parsed statement");
    Ok(st)
}

#[instrument(target = "parse", level = "debug")]
fn try_parse_for_statement<'i, 't>(
    tokens: &mut Tokens<'i, 't>,
) -> Result<Option<AstStatementType>, String> {
    if !tokens.try_consume_expected_next_token("for") {
        return Ok(None);
    }

    tokens.consume_expected_next_token("(")?;

    let initializer = if let Ok(Some(decl)) = try_parse_var_declaration(tokens) {
        AstForInit::Declaration(decl)
    } else {
        AstForInit::Expression(parse_optional_expression(tokens, ";")?)
    };

    let condition_opt = parse_optional_expression(tokens, ";")?;

    let final_expr_opt = parse_optional_expression(tokens, ")")?;

    let body = Box::new(parse_statement(tokens)?);

    Ok(Some(AstStatementType::For(
        initializer,
        condition_opt,
        final_expr_opt,
        body,
        None,
        None,
    )))
}

#[instrument(target = "parse", level = "debug")]
fn parse_statement_labels<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<Vec<AstLabel>, String> {
    fn parse_label<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstLabel, String> {
        let mut tokens = original_tokens.clone();

        // Label is a name followed by :
        let label_name = tokens.consume_identifier()?;
        tokens.consume_expected_next_token(":")?;

        *original_tokens = tokens;
        Ok(AstLabel(String::from(label_name)))
    }

    let mut labels = vec![];
    while let Ok(label) = parse_label(tokens) {
        labels.push(label);
    }
    Ok(labels)
}

#[instrument(target = "parse", level = "debug")]
fn try_parse_var_declaration<'i, 't>(
    tokens: &mut Tokens<'i, 't>,
) -> Result<Option<AstVarDeclaration>, String> {
    if !tokens.try_consume_expected_next_token("int") {
        return Ok(None);
    }

    let var_name = tokens.consume_identifier()?;

    // Optional initializer is present.
    let initializer_opt = if tokens.try_consume_expected_next_token("=") {
        Some(parse_expression(tokens)?)
    } else {
        None
    };

    tokens.consume_expected_next_token(";")?;

    let decl = AstVarDeclaration {
        identifier: AstIdentifier(String::from(var_name)),
        initializer_opt,
    };

    debug!(target: "parse", decl = %DisplayFmtNode(&decl), "parsed declaration");

    Ok(Some(decl))
}

fn parse_expression<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    #[instrument(target = "parse", level = "debug")]
    fn parse_expression_with_precedence<'i, 't>(
        original_tokens: &mut Tokens<'i, 't>,
        min_precedence_allowed: u8,
    ) -> Result<AstExpression, String> {
        // Always try for at least one factor in an expression.
        let mut left = parse_factor(original_tokens)?;
        debug!(target: "parse", expr = %DisplayFmtNode(&left), "parsed expression");

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
                            // Assignment is right-associative, not left-associative, so parse with the precedence of
                            // the operator so that further tokens of the same precedence would also go to the right
                            // side, not left side.
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
                            // strictly higher precedence, or else it shouldn't be part of the right-hand-side
                            // expression.
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

                    debug!(target: "parse", expr = %DisplayFmtNode(&left), "parsed expression");

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

    parse_expression_with_precedence(tokens, 0)
}

#[instrument(target = "parse", level = "debug")]
fn parse_optional_expression<'i, 't>(
    tokens: &mut Tokens<'i, 't>,
    terminator: &str,
) -> Result<Option<AstExpression>, String> {
    // Try to parse an expression and then the desired terminator.
    if let Ok(expr) = parse_expression(tokens) {
        tokens.consume_expected_next_token(terminator)?;
        Ok(Some(expr))
    } else if tokens.try_consume_expected_next_token(terminator) {
        // Well, the expression wasn't found. See if just the terminator is, in which case the optional expression is
        // valid, just with no expression found.
        Ok(None)
    } else {
        // If the terminator isn't found, then it's not valid.
        Err(format!(
            "failed to parse optional expression starting at {:?}",
            tokens
        ))
    }
}

#[instrument(target = "parse", level = "debug")]
fn parse_factor<'i, 't>(tokens: &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    let factor = if let Ok(integer_literal) = tokens.consume_and_parse_next_token::<u32>() {
        AstExpression::Constant(integer_literal)
    } else if tokens.try_consume_expected_next_token("(") {
        let inner = parse_expression(tokens)?;
        tokens.consume_expected_next_token(")")?;
        inner
    } else if let Ok(operator) = tokens.consume_and_parse_next_token::<AstUnaryOperator>() {
        let inner = parse_factor(tokens)?;
        AstExpression::UnaryOperator(operator, Box::new(inner))
    } else if let Some(func_call) = try_parse_function_call(tokens)? {
        func_call
    } else if let Ok(var_name) = tokens.consume_identifier() {
        AstExpression::Var(AstIdentifier(String::from(var_name)))
    } else if !tokens.is_empty() {
        return Err(format!("unknown factor \"{}\"", tokens[0]));
    } else {
        return Err(format!("end of file reached while parsing factor"));
    };

    // Immediately after parsing a factor, attempt to parse a postfix operator, because it's higher precedence than
    // anything else.
    Ok(
        if let Some(suffix_operator) = try_parse_postfix_operator(tokens) {
            AstExpression::UnaryOperator(suffix_operator, Box::new(factor))
        } else {
            factor
        },
    )
}

fn try_parse_function_call<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<Option<AstExpression>, String> {
    let mut tokens = original_tokens.clone();

    let Some(identifier) = tokens.try_consume_identifier() else {
        return Ok(None);
    };

    if !tokens.try_consume_expected_next_token("(") {
        return Ok(None);
    }

    // With an identifier and an open parenthesis, this is now committed to being a function call.
    *original_tokens = tokens;

    let mut arguments = vec![];
    while !original_tokens.try_consume_expected_next_token(")") {
        // Arguments are separated by commas.
        if !arguments.is_empty() {
            original_tokens.consume_expected_next_token(",")?;
        }

        arguments.push(parse_expression(original_tokens)?);
    }

    Ok(Some(AstExpression::FuncCall(
        AstIdentifier(String::from(identifier)),
        arguments,
    )))
}

#[instrument(target = "parse", level = "debug")]
fn try_parse_postfix_operator<'i, 't>(
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

fn generate_program_code(mode: Mode, ast_program: &AstProgram) -> Result<String, String> {
    let tac_program = ast_program.to_tac()?;

    info!(tac = %DisplayFmtNode(&tac_program));

    match mode {
        Mode::All | Mode::CodegenOnly => {
            let mut asm_program = tac_program.to_asm()?;

            info!(asm = %DisplayFmtNode(&asm_program));

            asm_program.finalize()?;

            info!(asm = %DisplayFmtNode(&asm_program), "asm after finalize");

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
    asm_path: &Path,
    output_path: &str,
    extra_link_args_opt: Option<&[String]>,
    should_suppress_output: bool,
    temp_dir: &Path,
) -> Option<i32> {
    let temp_dir = path::absolute(temp_dir).expect("failed to make absolute path");

    let mut command = Command::new("ml64.exe");

    let asm_full_path = path::absolute(asm_path).expect("failed to get full path to asm file");
    let asm_full_path_str = asm_full_path
        .to_str()
        .expect("failed to get full path string for asm file");

    // If link args are specified, even an empty list of them, it indicates the caller wants linking, i.e. not
    // compile-only.
    let mut command_args;
    let temp_output_path;
    let mut temp_pdb_path_opt = None;
    if let Some(extra_link_args) = extra_link_args_opt {
        temp_output_path = temp_dir.join("output.exe");

        let mut temp_pdb_path = temp_output_path.clone();
        temp_pdb_path.set_extension("pdb");

        command_args = vec![
            String::from("/Zi"),
            format!("/Fe{}", temp_output_path.display()),
            String::from(asm_full_path_str),
            String::from("/link"),
            String::from("/nodefaultlib:libcmt"),
            String::from("msvcrt.lib"),
            format!("/pdb:{}", temp_pdb_path.display()),
        ];

        // It's common to pass paths to object or library files in the link args, but it's also common to pass linker
        // arguments. For paths, make them absolute so they can be used even if we change current directory for the
        // assembler execution.
        for arg in extra_link_args.iter() {
            command_args.push(if let Ok(true) = std::fs::exists(arg) {
                String::from(path::absolute(arg).unwrap().to_str().unwrap())
            } else {
                arg.clone()
            });
        }

        temp_pdb_path_opt = Some(temp_pdb_path);
    } else {
        temp_output_path = temp_dir.join("code.obj");
        command_args = vec![
            String::from("/Zi"),
            String::from("/c"),
            format!("/Fo{}", temp_output_path.display()),
            String::from(asm_full_path_str),
        ];
    };

    command.args(&command_args);
    command.current_dir(&temp_dir);

    debug!("{}", format_command_args(&command));

    if should_suppress_output {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }

    let status = command.status().expect("failed to run ml64.exe");

    if status.success() {
        debug!(assembly_status = %status);

        std::fs::rename(&temp_output_path, &Path::new(output_path));

        // Also move the PDB
        if extra_link_args_opt.is_some() {
            let mut pdb_path = Path::new(output_path).to_path_buf();
            pdb_path.set_extension("pdb");
            std::fs::rename(temp_pdb_path_opt.as_ref().unwrap(), &pdb_path);
        }
    } else {
        error!(assembly_status = %status);
    }

    status.code()
}

// TODO should return line numbers with errors
#[instrument(target = "parse", level = "debug", skip_all)]
fn parse_and_validate(mode: Mode, input: &str) -> Result<AstProgram, Vec<String>> {
    let token_strings = lex_all_tokens(&input)?;
    let mut tokens = Tokens(&token_strings);

    if let Mode::LexOnly = mode {
        return Ok(AstProgram::new(vec![]));
    }

    // TODO all parsing should return a list of errors, not just one. for now, wrap it in a single error
    let mut ast = {
        let mut functions = vec![];
        while let Some(function) = try_parse_function(&mut tokens).map_err(|e| vec![e])? {
            functions.push(function);
        }

        AstProgram::new(functions)
    };

    info!(target: "parse", ast = %DisplayFmtNode(&ast));

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

    info!(target: "resolve", ast = %DisplayFmtNode(&ast), "AST after resolve");

    Ok(ast)
}

fn preprocess(
    input: &str,
    should_suppress_output: bool,
    temp_dir: &Path,
) -> Result<String, String> {
    let temp_dir = path::absolute(temp_dir).expect("failed to make absolute path");
    let preprocessed_output_path = temp_dir.join("input.i");

    let temp_input_path = temp_dir.join("input.c");
    std::fs::write(&temp_input_path, &input);

    let mut command = Command::new("cl.exe");
    let args = [
        String::from("/P"),
        format!("/Fi{}", preprocessed_output_path.display()),
        String::from(temp_input_path.to_str().unwrap()),
    ];
    command.args(&args);
    command.current_dir(temp_dir);

    info!("preprocess command: {}", format_command_args(&command));

    if should_suppress_output {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }

    let status = command.status().expect("failed to run cl.exe");

    info!("preprocess status: {:?}", status);
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
    should_expect_success: bool,
) -> Result<i32, String> {
    fn helper(
        args: &LcArgs,
        input: &str,
        should_suppress_output: bool,
        temp_dir: &Path,
    ) -> Result<i32, String> {
        let input = preprocess(input, should_suppress_output, temp_dir)?;

        match parse_and_validate(args.mode, &input) {
            Ok(ref ast) => match args.mode {
                Mode::All | Mode::TacOnly | Mode::CodegenOnly => {
                    let asm = generate_program_code(args.mode, ast)?;
                    info!("assembly:\n{}", &asm);

                    let asm_path = temp_dir.join("code.asm");
                    std::fs::write(&asm_path, &asm);

                    if let Mode::All = args.mode {
                        let exit_code = assemble_and_link(
                            &asm_path,
                            &args.output_path.as_ref().unwrap(),
                            if args.should_compile_only {
                                None
                            } else {
                                Some(&args.extra_link_args)
                            },
                            should_suppress_output,
                            temp_dir,
                        )
                        .expect("programs should always have an exit code");

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

    // If the user requested to keep the intermediate files, then don't clean them up. Also, if the build result was not
    // what the caller expected (mainly tests), don't clean them up.
    if !args.should_keep_intermediate_files && ret.is_ok() == should_expect_success {
        debug!("cleaning up temp dir {}", temp_dir.to_string_lossy());
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

    #[arg(short = 'v')]
    verbose: bool,

    /// Keep intermediate build files
    #[arg(short = 'k')]
    should_keep_intermediate_files: bool,

    /// Compile-only. Don't link.
    #[arg(short = 'c')]
    should_compile_only: bool,

    /// Additional arguments for the linker.
    #[arg(name = "link", long = "link", value_name = "LINK_ARGS")]
    extra_link_args: Vec<String>,
}

fn main() {
    let args = LcArgs::parse();

    if args.verbose {
        Registry::default()
            .with(tracing_forest::ForestLayer::default().with_filter(EnvFilter::from_default_env()))
            .init();
    }

    if let Mode::All = args.mode {
        if args.output_path.is_none() {
            println!("Must specify output path!");
            std::process::exit(1)
        }
    }

    info!("Loading {}", args.input_path);
    let input = std::fs::read_to_string(&args.input_path).unwrap();

    let exit_code = match compile_and_link(&args, &input, false, true) {
        Ok(inner_exit_code) => inner_exit_code,
        Err(msg) => {
            println!("error! {}", msg);
            1
        }
    };

    std::process::exit(exit_code)
}

#[cfg(test)]
mod test;
