#[macro_use]
extern crate lazy_static;
extern crate rand;
extern crate regex;

use {
    clap::Parser,
    rand::distributions::Alphanumeric,
    rand::{thread_rng, Rng},
    regex::Regex,
    std::{collections::HashMap, fmt, fmt::Display, ops::Deref, path::*, process::*},
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
}

impl<'i, 't> Deref for Tokens<'i, 't> {
    type Target = &'t [&'i str];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(PartialEq, Clone, Debug)]
struct AstProgram<'i> {
    functions: Vec<AstFunction<'i>>,
}

#[derive(PartialEq, Clone, Debug)]
struct AstFunction<'i> {
    name: &'i str,
    parameters: Vec<String>,
    body: AstStatement,
}

// TODO: use a string slice instead of a string
#[derive(PartialEq, Clone, Debug)]
enum AstStatement {
    Return(AstExpression),
}

#[derive(PartialEq, Clone, Debug)]
enum AstExpression {
    Constant(u32),
    UnaryOperator(AstUnaryOperator, Box<AstExpression>),
}

#[derive(PartialEq, Clone, Debug)]
enum AstUnaryOperator {
    Negation,
    BitwiseNot,
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

#[derive(Debug)]
enum TacInstruction {
    Return(TacVal),
    UnaryOp(TacUnaryOperator, TacVal, TacVar),
}

#[derive(Debug, Clone)]
struct TacVar(String);

#[derive(Debug)]
enum TacVal {
    Constant(u32),
    Var(TacVar),
}

#[derive(Debug)]
enum TacUnaryOperator {
    Negation,
    BitwiseNot,
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
enum AsmInstruction {
    Mov(AsmVal, AsmLocation),
    UnaryOp(AsmUnaryOperator, AsmLocation),
    AllocateStack(u32),
    Ret(u32),
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

struct TacGenState {
    next_temporary_id: u32,
}

struct FuncStackFrame {
    names: HashMap<String, i32>,
    max_base_offset: u32,
}

impl<'i> AstProgram<'i> {
    fn lookup_function_definition(&'i self, name: &str) -> Option<&'i AstFunction<'i>> {
        self.functions.iter().find(|func| func.name == name)
    }

    fn to_tac(&self) -> Result<TacProgram, String> {
        let mut tacgen_state = TacGenState::new();

        let mut functions = vec![];
        for func in self.functions.iter() {
            functions.push(func.to_tac(&mut tacgen_state)?);
        }

        Ok(TacProgram { functions })
    }
}

impl<'i> FmtNode for AstProgram<'i> {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        Self::fmt_nodelist(f, self.functions.iter(), "\n\n", 0)
    }
}

impl<'i> AstFunction<'i> {
    fn to_tac(&self, tacgen_state: &mut TacGenState) -> Result<TacFunction, String> {
        Ok(TacFunction {
            name: String::from(self.name),
            body: self.body.to_tac(tacgen_state)?,
        })
    }
}

impl<'i> FmtNode for AstFunction<'i> {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;
        write!(f, "FUNC {}(", self.name)?;
        fmt_list(f, self.parameters.iter(), ", ")?;
        writeln!(f, "):")?;
        self.body.fmt_node(f, indent_levels + 1)
    }
}

impl AstStatement {
    fn to_tac(&self, tacgen_state: &mut TacGenState) -> Result<Vec<TacInstruction>, String> {
        let mut instructions = vec![];

        match self {
            AstStatement::Return(ast_exp) => {
                let tac_val = ast_exp.to_tac(tacgen_state, &mut instructions)?;
                instructions.push(TacInstruction::Return(tac_val));
            }
        }

        Ok(instructions)
    }
}

impl FmtNode for AstStatement {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;

        match self {
            AstStatement::Return(expr) => {
                write!(f, "return ")?;
                expr.fmt_node(f, indent_levels + 1)?;
            }
        }

        Ok(())
    }
}

impl AstUnaryOperator {
    fn to_tac(&self) -> TacUnaryOperator {
        match self {
            AstUnaryOperator::Negation => TacUnaryOperator::Negation,
            AstUnaryOperator::BitwiseNot => TacUnaryOperator::BitwiseNot,
        }
    }
}

impl FmtNode for AstUnaryOperator {
    fn fmt_node(&self, f: &mut fmt::Formatter, _indent_levels: u32) -> fmt::Result {
        f.write_str(match self {
            AstUnaryOperator::Negation => "-",
            AstUnaryOperator::BitwiseNot => "~",
        })
    }
}

impl std::str::FromStr for AstUnaryOperator {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "-" => Ok(AstUnaryOperator::Negation),
            "~" => Ok(AstUnaryOperator::BitwiseNot),
            _ => Err(format!("unknown operator {}", s)),
        }
    }
}

impl AstExpression {
    fn to_tac(
        &self,
        tacgen_state: &mut TacGenState,
        instructions: &mut Vec<TacInstruction>,
    ) -> Result<TacVal, String> {
        Ok(match self {
            AstExpression::Constant(num) => TacVal::Constant(*num),
            AstExpression::UnaryOperator(ast_unary_op, ast_exp_inner) => {
                let tac_exp_inner_var = ast_exp_inner.to_tac(tacgen_state, instructions)?;
                let tempvar = tacgen_state.allocate_temporary();
                instructions.push(TacInstruction::UnaryOp(
                    ast_unary_op.to_tac(),
                    tac_exp_inner_var,
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
                operator.fmt_node(f, 0)?;
                expr.fmt_node(f, 0)?;
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

impl TacInstruction {
    fn to_asm(&self, func_body: &mut Vec<AsmInstruction>) -> Result<(), String> {
        match self {
            TacInstruction::Return(val) => {
                func_body.push(AsmInstruction::Mov(val.to_asm()?, AsmLocation::Reg("eax")));
                func_body.push(AsmInstruction::Ret(0));
            }
            TacInstruction::UnaryOp(unary_op, src_val, dest_var) => {
                let dest_asm_loc = dest_var.to_asm()?;
                func_body.push(AsmInstruction::Mov(src_val.to_asm()?, dest_asm_loc.clone()));
                func_body.push(AsmInstruction::UnaryOp(unary_op.to_asm()?, dest_asm_loc));
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
                writeln!(f)?;
            }
            TacInstruction::UnaryOp(unary_op, src, dest) => {
                dest.fmt_node(f, 0)?;
                write!(f, " = ")?;
                unary_op.fmt(f)?;
                write!(f, " ")?;
                src.fmt_node(f, 0)?;
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
        })
    }
}

impl fmt::Display for TacUnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            TacUnaryOperator::Negation => "Negate",
            TacUnaryOperator::BitwiseNot => "BitwiseNot",
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
                    if let AsmVal::Loc(src_loc) = src_val {
                        src_loc.resolve_pseudoregister(&mut frame)?;
                    }

                    dest_loc.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::UnaryOp(_, dest_loc) => {
                    dest_loc.resolve_pseudoregister(&mut frame)?;
                }
                AsmInstruction::AllocateStack(_) => {}
                AsmInstruction::Ret(_) => {}
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

        let mut i = 0;
        while i < self.body.len() {
            // For any Mov that uses a stack offset for both src and dest, x64 assembly requires that we first store it
            // in a temporary register.
            if let AsmInstruction::Mov(
                ref src_loc @ AsmVal::Loc(AsmLocation::RspOffset(_, _)),
                ref mut dest_loc @ AsmLocation::RspOffset(_, _),
            ) = &mut self.body[i]
            {
                let real_dest = dest_loc.clone();
                *dest_loc = AsmLocation::Reg("r10d");

                self.body.insert(
                    i + 1,
                    AsmInstruction::Mov(AsmVal::Loc(AsmLocation::Reg("r10d")), real_dest),
                );

                // We don't need to examine the next instruction since we just inserted it and know it's correct.
                i += 1;
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

impl AsmInstruction {
    fn convert_to_rsp_offset(&mut self, frame: &FuncStackFrame) {
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                if let AsmVal::Loc(src_loc) = src_val {
                    src_loc.convert_to_rsp_offset(frame);
                }

                dest_loc.convert_to_rsp_offset(frame);
            }
            AsmInstruction::UnaryOp(_, dest_loc) => {
                dest_loc.convert_to_rsp_offset(frame);
            }
            AsmInstruction::AllocateStack(_) => {}
            AsmInstruction::Ret(_) => {}
        }
    }

    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                format_code_and_comment(
                    f,
                    |f| {
                        write!(f, "mov ")?;
                        dest_loc.emit_code(f)?;
                        write!(f, ",")?;
                        src_val.emit_code(f)
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
                        dest_loc.emit_code(f)
                    },
                    |f| {
                        unary_op.emit_code(f)?;
                        write!(f, " ")?;
                        dest_loc.fmt_asm_comment(f)
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
        }

        Ok(())
    }
}

impl FmtNode for AsmInstruction {
    fn fmt_node(&self, f: &mut fmt::Formatter, indent_levels: u32) -> fmt::Result {
        Self::write_indent(f, indent_levels)?;
        match self {
            AsmInstruction::Mov(src_val, dest_loc) => {
                write!(f, "Mov ")?;
                src_val.fmt_node(f, 0)?;
                write!(f, " -> ")?;
                dest_loc.fmt_node(f, 0)?;
            }
            AsmInstruction::UnaryOp(unary_op, dest_loc) => {
                unary_op.fmt_node(f, 0)?;
                write!(f, " ")?;
                dest_loc.fmt_node(f, 0)?;
            }
            AsmInstruction::AllocateStack(size) => {
                write!(f, "AllocateStack {}", size)?;
            }
            AsmInstruction::Ret(size) => {
                write!(f, "Ret (dealloc {} stack)", size)?;
            }
        }

        Ok(())
    }
}

impl AsmVal {
    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmVal::Imm(num) => {
                write!(f, "{}", num)?;
            }
            AsmVal::Loc(loc) => {
                loc.emit_code(f)?;
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

    fn emit_code(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AsmLocation::Reg(name) => {
                f.write_str(name)?;
            }
            AsmLocation::RspOffset(rsp_offset, _name) => {
                write!(f, "DWORD PTR [rsp+{}]", rsp_offset)?;
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
            AsmLocation::RspOffset(rsp_offset, name) => {
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

impl TacGenState {
    fn new() -> Self {
        TacGenState {
            next_temporary_id: 0,
        }
    }

    fn allocate_temporary(&mut self) -> TacVar {
        self.next_temporary_id += 1;
        TacVar(format!("{:03}_tmp", self.next_temporary_id - 1))
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
            Regex::new(r"^&&").expect("failed to compile regex"),
            Regex::new(r"^\|\|").expect("failed to compile regex"),
            Regex::new(r"^==").expect("failed to compile regex"),
            Regex::new(r"^!=").expect("failed to compile regex"),
            Regex::new(r"^<=").expect("failed to compile regex"),
            Regex::new(r"^>=").expect("failed to compile regex"),
            Regex::new(r"^--").expect("failed to compile regex"),
            Regex::new(r"^\+\+").expect("failed to compile regex"),
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
            Regex::new(r"^[a-zA-Z]\w*\b").expect("failed to compile regex"),
            Regex::new(r"^[0-9]+\b").expect("failed to compile regex"),
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

    tokens.consume_expected_next_token("{")?;

    let body = parse_statement(&mut tokens)?;

    tokens.consume_expected_next_token("}")?;

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

    let var_name = tokens.consume_next_token()?;
    *original_tokens = tokens;
    Ok(var_name)
}

fn parse_statement<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstStatement, String> {
    let mut tokens = original_tokens.clone();

    tokens.consume_expected_next_token("return")?;

    let statement = AstStatement::Return(parse_expression(&mut tokens)?);
    tokens.consume_expected_next_token(";")?;

    *original_tokens = tokens;
    Ok(statement)
}

fn parse_expression<'i, 't>(original_tokens: &mut Tokens<'i, 't>) -> Result<AstExpression, String> {
    parse_unary_expression(original_tokens)
}

fn parse_unary_expression<'i, 't>(
    original_tokens: &mut Tokens<'i, 't>,
) -> Result<AstExpression, String> {
    let mut tokens = original_tokens.clone();

    let ret = if let Ok(integer_literal) = tokens.consume_and_parse_next_token::<u32>() {
        Ok(AstExpression::Constant(integer_literal))
    } else if let Ok(_) = tokens.consume_expected_next_token("(") {
        let inner = parse_expression(&mut tokens)?;
        tokens.consume_expected_next_token(")")?;
        Ok(inner)
    } else if let Ok(operator) = tokens.consume_and_parse_next_token::<AstUnaryOperator>() {
        let inner = parse_unary_expression(&mut tokens)?;
        Ok(AstExpression::UnaryOperator(operator, Box::new(inner)))
    } else {
        Err(String::from("unknown unary expression"))
    };

    if ret.is_ok() {
        *original_tokens = tokens;
    }

    ret
}

fn generate_program_code(mode: Mode, ast_program: &AstProgram) -> Result<String, String> {
    let tac_program = ast_program.to_tac()?;

    println!("tac:\n{}", display_with(|f| { tac_program.fmt_node(f, 0) }));

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

fn validate_ast(ast_program: &AstProgram) -> Result<(), Vec<String>> {
    let mut errors = vec![];

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
    let token_strings = lex_all_tokens(&input)?;
    let mut tokens = Tokens(&token_strings);
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
    let ast = {
        let mut functions = vec![];
        while let Ok(function) = parse_function(&mut tokens) {
            functions.push(function);
        }

        if tokens.0.len() == 0 {
            Ok(AstProgram { functions })
        } else {
            Err(vec![format!(
                "extra tokens after main function end: {:?}",
                tokens
            )])
        }
    }?;

    println!("AST:\n{}\n", display_with(|f| ast.fmt_node(f, 0)));

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
                Mode::All | Mode::TacOnly | Mode::CodegenOnly => {
                    let asm = generate_program_code(args.mode, &ast)?;

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

    /// Lex, parse, and generate TAC only. Exit code is zero if successful.
    #[value(name = "tac", alias = "t")]
    TacOnly,

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

    #[test]
    fn lex_no_at() {
        assert!(lex_all_tokens("int main() { return 0@1; }").is_err());
    }

    #[test]
    fn lex_no_backslash() {
        assert!(lex_all_tokens("\\").is_err());
    }

    #[test]
    fn lex_no_backtick() {
        assert!(lex_all_tokens("`").is_err());
    }

    #[test]
    fn lex_bad_identifier() {
        assert!(lex_all_tokens("int main() { return 1foo; }").is_err());
    }

    #[test]
    fn lex_no_at_identifier() {
        assert!(lex_all_tokens("int main() { return @b; }").is_err());
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
    fn test_codegen_unary_neg() {
        test_codegen_expression("-5", -5);
    }

    #[test]
    fn test_codegen_unary_not() {
        test_codegen_expression("~12", -13);
    }

    #[test]
    fn test_codegen_unary_neg_zero() {
        test_codegen_expression("-0", 0);
    }

    #[test]
    fn test_codegen_unary_not_zero() {
        test_codegen_expression("~0", -1);
    }

    #[test]
    fn test_codegen_unary_neg_min_val() {
        test_codegen_expression("-2147483647", -2147483647);
    }

    #[test]
    fn test_codegen_unary_not_and_neg() {
        test_codegen_expression("~-3", 2);
    }

    #[test]
    fn test_codegen_unary_not_and_neg_zero() {
        test_codegen_expression("-~0", 1);
    }

    #[test]
    fn test_codegen_unary_not_and_neg_min_val() {
        test_codegen_expression("~-2147483647", 2147483646);
    }

    #[test]
    fn test_codegen_unary_grouping_outside() {
        test_codegen_expression("(-2)", -2);
    }

    #[test]
    fn test_codegen_unary_grouping_inside() {
        test_codegen_expression("~(2)", -3);
    }

    #[test]
    fn test_codegen_unary_grouping_inside_and_outside() {
        test_codegen_expression("-(-4)", 4);
    }

    #[test]
    fn test_codegen_unary_grouping_several() {
        test_codegen_expression("-((((((10))))))", -10);
    }

    #[test]
    fn test_parse_fail_extra_paren() {
        test_codegen_mainfunc_failure("return (3));");
    }

    #[test]
    fn test_parse_fail_unclosed_paren() {
        test_codegen_mainfunc_failure("return (3;");
    }

    #[test]
    fn test_parse_fail_missing_immediate() {
        test_codegen_mainfunc_failure("return ~;");
    }

    #[test]
    fn test_parse_fail_missing_immediate_2() {
        test_codegen_mainfunc_failure("return -~;");
    }

    #[test]
    fn test_parse_fail_missing_semicolon() {
        test_codegen_mainfunc_failure("return 5");
    }

    #[test]
    fn test_parse_fail_parens_around_operator() {
        test_codegen_mainfunc_failure("return (-)5;");
    }

    #[test]
    fn test_parse_fail_operator_wrong_order() {
        test_codegen_mainfunc_failure("return 5-;");
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
