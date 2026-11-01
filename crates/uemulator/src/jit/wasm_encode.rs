//! A minimal WebAssembly binary encoder: the function body instructions the block generator
//! (`emit_wasm.rs`) uses and the one fixed module shape every block is wrapped in. Nothing here
//! knows about the guest. Pure `alloc`, so it also builds and is unit-tested on hosts that cannot
//! run the modules (where nothing but those tests uses it).

#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

#[cfg(freestanding)]
use crate::prelude::*;

pub(crate) const I32: u8 = 0x7f;
pub(crate) const I64: u8 = 0x7e;

/// Block types of `if`: no result, or one i64.
pub(crate) const BLOCK_VOID: u8 = 0x40;
pub(crate) const BLOCK_I64: u8 = I64;

/// Opcodes (WebAssembly core spec, "Binary Format: Instructions"), named as in the text format.
pub(crate) mod op {
    pub const BLOCK: u8 = 0x02;
    pub const IF: u8 = 0x04;
    pub const ELSE: u8 = 0x05;
    pub const END: u8 = 0x0b;
    pub const BR_IF: u8 = 0x0d;
    pub const RETURN: u8 = 0x0f;
    pub const CALL_INDIRECT: u8 = 0x11;
    /// Tail call (WebAssembly 3.0, shipped in Chrome 112, Firefox 121 and Safari 18.2).
    pub const RETURN_CALL_INDIRECT: u8 = 0x13;
    pub const SELECT: u8 = 0x1b;
    pub const LOCAL_GET: u8 = 0x20;
    pub const LOCAL_SET: u8 = 0x21;
    pub const LOCAL_TEE: u8 = 0x22;
    pub const I32_LOAD: u8 = 0x28;
    pub const I64_LOAD: u8 = 0x29;
    pub const I32_LOAD8_U: u8 = 0x2d;
    pub const I64_LOAD8_S: u8 = 0x30;
    pub const I64_LOAD8_U: u8 = 0x31;
    pub const I64_LOAD16_S: u8 = 0x32;
    pub const I64_LOAD16_U: u8 = 0x33;
    pub const I64_LOAD32_S: u8 = 0x34;
    pub const I64_LOAD32_U: u8 = 0x35;
    pub const I32_STORE: u8 = 0x36;
    pub const I64_STORE: u8 = 0x37;
    pub const I32_STORE16: u8 = 0x3b;
    pub const I64_STORE8: u8 = 0x3c;
    pub const I64_STORE16: u8 = 0x3d;
    pub const I64_STORE32: u8 = 0x3e;
    pub const I32_CONST: u8 = 0x41;
    pub const I64_CONST: u8 = 0x42;
    pub const I32_EQZ: u8 = 0x45;
    pub const I64_GE_U: u8 = 0x5a;
    pub const I64_EQZ: u8 = 0x50;
    pub const I64_EQ: u8 = 0x51;
    pub const I64_NE: u8 = 0x52;
    pub const I64_LT_S: u8 = 0x53;
    pub const I64_LT_U: u8 = 0x54;
    pub const I64_GE_S: u8 = 0x59;
    pub const I64_LE_U: u8 = 0x58;
    pub const I32_ADD: u8 = 0x6a;
    pub const I32_AND: u8 = 0x71;
    pub const I32_OR: u8 = 0x72;
    pub const I32_XOR: u8 = 0x73;
    pub const I32_SHL: u8 = 0x74;
    pub const I32_ROTL: u8 = 0x77;
    pub const I32_ROTR: u8 = 0x78;
    pub const I64_POPCNT: u8 = 0x7b;
    pub const I64_ADD: u8 = 0x7c;
    pub const I64_SUB: u8 = 0x7d;
    pub const I64_MUL: u8 = 0x7e;
    pub const I64_AND: u8 = 0x83;
    pub const I64_OR: u8 = 0x84;
    pub const I64_XOR: u8 = 0x85;
    pub const I64_SHL: u8 = 0x86;
    pub const I64_SHR_S: u8 = 0x87;
    pub const I64_SHR_U: u8 = 0x88;
    pub const I64_ROTL: u8 = 0x89;
    pub const I64_ROTR: u8 = 0x8a;
    pub const I32_WRAP_I64: u8 = 0xa7;
    pub const I64_EXTEND_I32_U: u8 = 0xad;
    pub const I64_EXTEND8_S: u8 = 0xc2;
    pub const I64_EXTEND16_S: u8 = 0xc3;
    pub const I64_EXTEND32_S: u8 = 0xc4;
}

/// Appends `value` as unsigned LEB128.
pub(crate) fn leb_u32(out: &mut Vec<u8>, mut value: u32) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Appends `value` as signed LEB128 (the encoding of `i32.const` and `i64.const` operands).
pub(crate) fn leb_i64(out: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        if done {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// The instructions of one function body.
#[derive(Default)]
pub(crate) struct Code {
    pub bytes: Vec<u8>,
}

impl Code {
    pub fn op(&mut self, opcode: u8) -> &mut Self {
        self.bytes.push(opcode);
        self
    }

    pub fn i32_const(&mut self, value: i32) -> &mut Self {
        self.bytes.push(op::I32_CONST);
        leb_i64(&mut self.bytes, value as i64);
        self
    }

    pub fn i64_const(&mut self, value: i64) -> &mut Self {
        self.bytes.push(op::I64_CONST);
        leb_i64(&mut self.bytes, value);
        self
    }

    pub fn local_get(&mut self, index: u32) -> &mut Self {
        self.bytes.push(op::LOCAL_GET);
        leb_u32(&mut self.bytes, index);
        self
    }

    pub fn local_set(&mut self, index: u32) -> &mut Self {
        self.bytes.push(op::LOCAL_SET);
        leb_u32(&mut self.bytes, index);
        self
    }

    pub fn local_tee(&mut self, index: u32) -> &mut Self {
        self.bytes.push(op::LOCAL_TEE);
        leb_u32(&mut self.bytes, index);
        self
    }

    /// A load or store: the opcode, then the memarg. The alignment is only a hint to the engine,
    /// and guest accesses are unaligned, so it is always 0 (byte alignment).
    pub fn mem(&mut self, opcode: u8, offset: u32) -> &mut Self {
        self.bytes.push(opcode);
        leb_u32(&mut self.bytes, 0);
        leb_u32(&mut self.bytes, offset);
        self
    }

    pub fn begin_if(&mut self, block_type: u8) -> &mut Self {
        self.bytes.push(op::IF);
        self.bytes.push(block_type);
        self
    }

    pub fn call_indirect(&mut self, type_index: u32) -> &mut Self {
        self.bytes.push(op::CALL_INDIRECT);
        leb_u32(&mut self.bytes, type_index);
        leb_u32(&mut self.bytes, 0);
        self
    }

    /// Jumps to the function in the table slot on top of the stack with the arguments below it,
    /// replacing this frame, so a chain of blocks does not grow the stack.
    pub fn return_call_indirect(&mut self, type_index: u32) -> &mut Self {
        self.bytes.push(op::RETURN_CALL_INDIRECT);
        leb_u32(&mut self.bytes, type_index);
        leb_u32(&mut self.bytes, 0);
        self
    }

    /// `block` with no result; `br_if(0)` inside it leaves it.
    pub fn begin_block(&mut self) -> &mut Self {
        self.bytes.push(op::BLOCK);
        self.bytes.push(BLOCK_VOID);
        self
    }

    pub fn br_if(&mut self, label: u32) -> &mut Self {
        self.bytes.push(op::BR_IF);
        leb_u32(&mut self.bytes, label);
        self
    }
}

/// The function types every block module declares, in this order; `emit_wasm.rs` names them by
/// index. A helper's type must match the Rust `extern "C"` signature exactly (a pointer is i32,
/// a `u64` is i64) or `call_indirect` traps, which would take the whole emulator down.
pub(crate) const BLOCK_TYPE: u32 = 0;
pub(crate) const EXEC_TYPE: u32 = 1;
pub(crate) const COND_TYPE: u32 = 2;
pub(crate) const LOAD_TYPE: u32 = 3;
pub(crate) const STORE_TYPE: u32 = 4;
/// The linking chain stub: `(cpu: i32, link_cell: i32) -> i64` (`emit_wasm::compile_chain_stub`).
/// A tail call only needs the callee's result to match the caller's, so a block can tail-call it.
pub(crate) const LINK_TYPE: u32 = 5;
const FUNCTION_TYPES: [(&[u8], &[u8]); 6] = [
    (&[I32], &[I64]),
    (&[I32, I32, I64], &[I64]),
    (&[I32, I64], &[I64]),
    (&[I32, I64, I64, I32], &[I64]),
    (&[I32, I64, I64, I64], &[I64]),
    (&[I32, I32], &[I64]),
];

const SECTION_TYPE: u8 = 1;
const SECTION_IMPORT: u8 = 2;
const SECTION_FUNCTION: u8 = 3;
const SECTION_EXPORT: u8 = 7;
const SECTION_CODE: u8 = 10;
const FUNC_TYPE_FORM: u8 = 0x60;
const EXTERN_FUNC: u8 = 0;
const EXTERN_TABLE: u8 = 1;
const EXTERN_MEMORY: u8 = 2;
const REF_FUNCREF: u8 = 0x70;
/// Limits with a minimum and no maximum. The minimum is 0 so any memory or table matches the
/// import; the engine checks the real one against it.
const LIMITS_MIN_ONLY: u8 = 0;

/// The export the host takes the block from (`www/jit_host.js`).
pub(crate) const EXPORT_NAME: &str = "block";

fn name(out: &mut Vec<u8>, text: &str) {
    leb_u32(out, text.len() as u32);
    out.extend_from_slice(text.as_bytes());
}

fn section(module: &mut Vec<u8>, id: u8, contents: &[u8]) {
    module.push(id);
    leb_u32(module, contents.len() as u32);
    module.extend_from_slice(contents);
}

/// A module that exports `block`, a `(cpu: i32) -> i64` function whose body is `code` (which must
/// end with its own `end`). It imports `env.memory` and `env.table`, the emulator's linear memory
/// and function table, so the code reads the `Cpu` directly and reaches the helpers through
/// `call_indirect`. `i32_locals` and `i64_locals` are declared after the parameter, in that order.
pub(crate) fn block_module(code: &[u8], i32_locals: u32, i64_locals: u32) -> Vec<u8> {
    module_of_type(BLOCK_TYPE, code, i32_locals, i64_locals)
}

/// `block_module` for a function of another of the `FUNCTION_TYPES`; its locals follow all of
/// that type's parameters.
pub(crate) fn module_of_type(
    function_type: u32,
    code: &[u8],
    i32_locals: u32,
    i64_locals: u32,
) -> Vec<u8> {
    let mut module = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

    let mut types = Vec::new();
    leb_u32(&mut types, FUNCTION_TYPES.len() as u32);
    for (params, results) in FUNCTION_TYPES {
        types.push(FUNC_TYPE_FORM);
        leb_u32(&mut types, params.len() as u32);
        types.extend_from_slice(params);
        leb_u32(&mut types, results.len() as u32);
        types.extend_from_slice(results);
    }
    section(&mut module, SECTION_TYPE, &types);

    let mut imports = Vec::new();
    leb_u32(&mut imports, 2);
    name(&mut imports, "env");
    name(&mut imports, "memory");
    imports.extend_from_slice(&[EXTERN_MEMORY, LIMITS_MIN_ONLY, 0]);
    name(&mut imports, "env");
    name(&mut imports, "table");
    imports.extend_from_slice(&[EXTERN_TABLE, REF_FUNCREF, LIMITS_MIN_ONLY, 0]);
    section(&mut module, SECTION_IMPORT, &imports);

    let mut functions = Vec::new();
    leb_u32(&mut functions, 1);
    leb_u32(&mut functions, function_type);
    section(&mut module, SECTION_FUNCTION, &functions);

    let mut exports = Vec::new();
    leb_u32(&mut exports, 1);
    name(&mut exports, EXPORT_NAME);
    exports.push(EXTERN_FUNC);
    leb_u32(&mut exports, 0);
    section(&mut module, SECTION_EXPORT, &exports);

    let mut body = Vec::new();
    let local_groups = [(i32_locals, I32), (i64_locals, I64)];
    let declared: Vec<_> = local_groups
        .iter()
        .filter(|(count, _)| *count > 0)
        .collect();
    leb_u32(&mut body, declared.len() as u32);
    for (count, ty) in declared {
        leb_u32(&mut body, *count);
        body.push(*ty);
    }
    body.extend_from_slice(code);
    let mut code_section = Vec::new();
    leb_u32(&mut code_section, 1);
    leb_u32(&mut code_section, body.len() as u32);
    code_section.extend_from_slice(&body);
    section(&mut module, SECTION_CODE, &code_section);
    module
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unsigned(value: u32) -> Vec<u8> {
        let mut out = Vec::new();
        leb_u32(&mut out, value);
        out
    }

    fn signed(value: i64) -> Vec<u8> {
        let mut out = Vec::new();
        leb_i64(&mut out, value);
        out
    }

    #[test]
    fn unsigned_leb128_matches_the_spec_examples() {
        assert_eq!(unsigned(0), [0x00]);
        assert_eq!(unsigned(127), [0x7f]);
        assert_eq!(unsigned(128), [0x80, 0x01]);
        assert_eq!(unsigned(624485), [0xe5, 0x8e, 0x26]);
        assert_eq!(unsigned(u32::MAX), [0xff, 0xff, 0xff, 0xff, 0x0f]);
    }

    #[test]
    fn signed_leb128_matches_the_spec_examples() {
        assert_eq!(signed(0), [0x00]);
        assert_eq!(signed(63), [0x3f]);
        assert_eq!(signed(64), [0xc0, 0x00]);
        assert_eq!(signed(-1), [0x7f]);
        assert_eq!(signed(-64), [0x40]);
        assert_eq!(signed(-65), [0xbf, 0x7f]);
        assert_eq!(signed(-123456), [0xc0, 0xbb, 0x78]);
        assert_eq!(
            signed(i64::MIN),
            [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x7f]
        );
        assert_eq!(
            signed(i64::MAX),
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00]
        );
    }

    #[test]
    fn constants_and_memory_operands_are_encoded_with_their_opcode() {
        let mut code = Code::default();
        code.i32_const(-1)
            .i64_const(300)
            .local_get(2)
            .mem(op::I64_LOAD, 200);
        assert_eq!(
            code.bytes,
            [
                0x41, 0x7f, 0x42, 0xac, 0x02, 0x20, 0x02, 0x29, 0x00, 0xc8, 0x01
            ]
        );
    }

    #[test]
    fn a_block_module_has_the_header_and_the_sections_in_order() {
        let module = block_module(&[op::END], 2, 3);
        assert_eq!(
            &module[..8],
            [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]
        );
        let mut ids = Vec::new();
        let mut offset = 8;
        while offset < module.len() {
            ids.push(module[offset]);
            let mut length = 0usize;
            let mut shift = 0;
            offset += 1;
            loop {
                let byte = module[offset];
                offset += 1;
                length |= ((byte & 0x7f) as usize) << shift;
                shift += 7;
                if byte & 0x80 == 0 {
                    break;
                }
            }
            offset += length;
        }
        assert_eq!(offset, module.len(), "every section length is exact");
        assert_eq!(
            ids,
            [
                SECTION_TYPE,
                SECTION_IMPORT,
                SECTION_FUNCTION,
                SECTION_EXPORT,
                SECTION_CODE
            ]
        );
        // two local groups: 2 x i32, 3 x i64, then the body
        assert!(module.ends_with(&[0x02, 0x02, I32, 0x03, I64, op::END]));
    }

    #[test]
    fn locals_that_are_not_used_are_not_declared() {
        let module = block_module(&[op::END], 0, 1);
        assert!(module.ends_with(&[0x01, 0x01, I64, op::END]));
        let module = block_module(&[op::END], 0, 0);
        assert!(module.ends_with(&[0x00, op::END]));
    }
}
