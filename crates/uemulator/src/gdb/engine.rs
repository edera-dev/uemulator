//! The packets: what each one asks of the stopped machine and what it answers.
//!
//! Threads are vCPUs, numbered from 1 (gdb reserves 0 for "any" and -1 for "all"). The machine
//! is all-stop: a stop stops every vCPU, and a resume names what each one does, with vCPUs a
//! `vCont` leaves out held (gdb steps one thread over a breakpoint that way).

#[cfg(freestanding)]
use crate::prelude::*;

use super::packet::{
    self, ACK, Decoder, Incoming, MAX_PACKET_BYTES, NACK, from_hex, number, parse_number, to_hex,
};
use super::{
    BreakpointKinds, DebugConfig, ResumeAction, StopEvent, StopKind, WatchKind, Watchpoint,
};
use crate::cpu::gdb::{self as target, FEATURES, REG_XMM0, REGISTER_COUNT, REGISTERS, Vcpus};
use alloc::collections::VecDeque;

/// The most memory one `m` answers: its hex has to fit in a packet.
const MAX_READ_BYTES: usize = MAX_PACKET_BYTES / 2 - 16;
/// The largest watchpoint accepted; gdb splits bigger ones itself or falls back to stepping.
const MAX_WATCH_BYTES: u64 = 4096;

const BREAKPOINT_SOFTWARE: u8 = b'0';
const BREAKPOINT_HARDWARE: u8 = b'1';
const WATCH_WRITE: u8 = b'2';
const WATCH_READ: u8 = b'3';
const WATCH_ACCESS: u8 = b'4';

/// The errors replies carry. gdb shows the number and nothing else depends on it.
const ERROR_BAD_PACKET: &[u8] = b"E01";
const ERROR_NO_THREAD: &[u8] = b"E02";
const ERROR_REFUSED: &[u8] = b"E03";
/// EFAULT, which gdb prints as "Cannot access memory".
const ERROR_MEMORY: &[u8] = b"E14";
/// A `qXfer` annex this target does not have.
const ERROR_NO_ANNEX: &[u8] = b"E00";

/// What a packet asks of the machine beyond its reply.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    /// Run again, each vCPU as its entry says.
    Resume(Vec<ResumeAction>),
    /// Clear breakpoints and watchpoints and run.
    Detach,
    /// End the machine.
    Kill,
}

/// A thread id in a packet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ThreadId {
    All,
    Any,
    Vcpu(usize),
}

/// The protocol state of one connection.
pub(super) struct Engine {
    decoder: Decoder,
    /// Decoded but not handled yet: what came after a resume in the same read.
    pending: VecDeque<Incoming>,
    no_ack: bool,
    /// The last reply as sent, for a `-` from the client.
    last_reply: Vec<u8>,
    /// `Hg`: the vCPU `g`, `p`, `m` and friends look at; `None` is the one that stopped.
    general: Option<usize>,
    /// `Hc`: the vCPU a legacy `s` steps; `None` is the one that stopped.
    continued: Option<usize>,
    /// The client understands `swbreak`/`hwbreak` in a stop reply (`qSupported`).
    swbreak: bool,
    hwbreak: bool,
    /// The client resumed the machine and waits for a stop reply.
    resume_pending: bool,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            decoder: Decoder::default(),
            pending: VecDeque::new(),
            no_ack: false,
            last_reply: Vec::new(),
            general: None,
            continued: None,
            swbreak: false,
            hwbreak: false,
            resume_pending: false,
        }
    }

    pub fn take_resume_pending(&mut self) -> bool {
        core::mem::take(&mut self.resume_pending)
    }

    /// Frames `payload` into `out`, kept for a `-` from the client.
    pub fn send(&mut self, payload: &[u8], out: &mut Vec<u8>) {
        self.last_reply.clear();
        packet::frame(payload, &mut self.last_reply);
        out.extend_from_slice(&self.last_reply);
    }

    /// Drops interrupts decoded behind a resume and says whether there were any: they arrived
    /// while the machine was still stopped, but are meant for it running.
    pub fn take_pending_interrupt(&mut self) -> bool {
        let before = self.pending.len();
        self.pending.retain(|item| *item != Incoming::Interrupt);
        self.pending.len() != before
    }

    /// The `T` packet for a stop: signal, the vCPU, and why.
    pub fn stop_reply(&self, event: &StopEvent, config: &DebugConfig) -> Vec<u8> {
        let mut reply = format!("T{:02x}thread:{:x};", event.signal(), event.vcpu + 1);
        match event.kind {
            StopKind::Watch(kind, address) => {
                let name = match kind {
                    WatchKind::Write => "watch",
                    WatchKind::Read => "rwatch",
                    WatchKind::Access => "awatch",
                };
                reply += &format!("{name}:{address:x};");
            }
            StopKind::Breakpoint(address) => {
                let kinds = config
                    .breakpoints
                    .get(&address)
                    .copied()
                    .unwrap_or_default();
                if kinds.software && self.swbreak {
                    reply += "swbreak:;";
                } else if kinds.hardware && self.hwbreak {
                    reply += "hwbreak:;";
                }
            }
            _ => {}
        }
        reply.into_bytes()
    }

    /// Handles what the client sent, appending what goes back to `out`. Stops at the first
    /// packet that resumes, detaches or kills, and returns it; later packets wait.
    pub fn process(
        &mut self,
        bytes: &[u8],
        vcpus: &mut dyn Vcpus,
        config: &mut DebugConfig,
        last_stop: Option<StopEvent>,
        out: &mut Vec<u8>,
    ) -> Option<Outcome> {
        let mut incoming = Vec::new();
        self.decoder.feed(bytes, &mut incoming);
        self.pending.extend(incoming);
        let stopped = last_stop.map_or(0, |event| event.vcpu);
        while let Some(item) = self.pending.pop_front() {
            match item {
                // An interrupt while stopped has nothing left to stop.
                Incoming::Ack | Incoming::Interrupt => {}
                Incoming::Nack => out.extend_from_slice(&self.last_reply),
                Incoming::Corrupt => {
                    if !self.no_ack {
                        out.push(NACK);
                    }
                }
                Incoming::Packet(data) => {
                    if !self.no_ack {
                        out.push(ACK);
                    }
                    let mut context = Context {
                        vcpus: &mut *vcpus,
                        config: &mut *config,
                        last_stop,
                        stopped,
                    };
                    let (reply, outcome) = self.handle(&data, &mut context);
                    if let Some(reply) = reply {
                        self.send(&reply, out);
                    }
                    if let Some(outcome) = outcome {
                        if matches!(outcome, Outcome::Resume(_)) {
                            self.resume_pending = true;
                        }
                        self.general = None;
                        self.continued = None;
                        return Some(outcome);
                    }
                }
            }
        }
        None
    }

    fn handle(
        &mut self,
        data: &[u8],
        context: &mut Context<'_>,
    ) -> (Option<Vec<u8>>, Option<Outcome>) {
        let reply = |bytes: &[u8]| (Some(bytes.to_vec()), None);
        let Some((&command, arguments)) = data.split_first() else {
            return reply(b"");
        };
        match command {
            b'?' => {
                let event = context.last_stop.unwrap_or(StopEvent {
                    vcpu: 0,
                    kind: StopKind::Quiet,
                    resumes: 0,
                });
                (Some(self.stop_reply(&event, context.config)), None)
            }
            b'q' | b'Q' => (Some(self.query(data, context)), None),
            b'H' => {
                let Some((&which, thread)) = arguments.split_first() else {
                    return reply(ERROR_BAD_PACKET);
                };
                let selection = match parse_thread(thread, context.vcpus.count()) {
                    Some(ThreadId::Vcpu(index)) => Some(index),
                    Some(ThreadId::All | ThreadId::Any) => None,
                    None => return reply(ERROR_NO_THREAD),
                };
                match which {
                    b'g' => self.general = selection,
                    b'c' => self.continued = selection,
                    _ => return reply(ERROR_BAD_PACKET),
                }
                reply(b"OK")
            }
            b'T' => match parse_thread(arguments, context.vcpus.count()) {
                Some(ThreadId::Vcpu(_)) => reply(b"OK"),
                _ => reply(ERROR_NO_THREAD),
            },
            b'g' => {
                match target::read_registers(context.vcpus, self.general.unwrap_or(context.stopped))
                {
                    Some(bytes) => reply(&to_hex(&bytes)),
                    None => reply(ERROR_NO_THREAD),
                }
            }
            b'G' => {
                let index = self.general_vcpu(context);
                match from_hex(arguments) {
                    Some(bytes) if target::write_registers(context.vcpus, index, &bytes) => {
                        reply(b"OK")
                    }
                    Some(_) => reply(ERROR_REFUSED),
                    None => reply(ERROR_BAD_PACKET),
                }
            }
            b'p' => {
                let register = parse_number(arguments).map(|value| value as usize);
                let index = self.general_vcpu(context);
                match register
                    .and_then(|register| target::read_register(context.vcpus, index, register))
                {
                    Some(bytes) => reply(&to_hex(&bytes)),
                    None => reply(ERROR_BAD_PACKET),
                }
            }
            b'P' => {
                let parsed = split_once(arguments, b'=').and_then(|(register, value)| {
                    Some((parse_number(register)? as usize, from_hex(value)?))
                });
                let index = self.general_vcpu(context);
                match parsed {
                    Some((register, value))
                        if target::write_register(context.vcpus, index, register, &value) =>
                    {
                        reply(b"OK")
                    }
                    Some(_) => reply(ERROR_REFUSED),
                    None => reply(ERROR_BAD_PACKET),
                }
            }
            b'm' => {
                let Some((address, length)) = parse_address_length(arguments) else {
                    return reply(ERROR_BAD_PACKET);
                };
                let length = (length as usize).min(MAX_READ_BYTES);
                let index = self.general_vcpu(context);
                let bytes = target::read_memory(context.vcpus, index, address, length);
                if bytes.is_empty() && length > 0 {
                    reply(ERROR_MEMORY)
                } else {
                    reply(&to_hex(&bytes))
                }
            }
            b'M' | b'X' => {
                let Some((header, payload)) = split_once(arguments, b':') else {
                    return reply(ERROR_BAD_PACKET);
                };
                let Some((address, length)) = parse_address_length(header) else {
                    return reply(ERROR_BAD_PACKET);
                };
                // `X` carries the bytes themselves (unescaped by the decoder), `M` their hex.
                let bytes = if command == b'X' {
                    Some(payload.to_vec())
                } else {
                    from_hex(payload)
                };
                let Some(bytes) = bytes.filter(|bytes| bytes.len() as u64 == length) else {
                    return reply(ERROR_BAD_PACKET);
                };
                let index = self.general_vcpu(context);
                if target::write_memory(context.vcpus, index, address, &bytes) {
                    reply(b"OK")
                } else {
                    reply(ERROR_MEMORY)
                }
            }
            b'c' | b's' | b'C' | b'S' => {
                // `C sig;addr` and `S sig;addr`: the signal means nothing to a machine.
                let address = if command.is_ascii_uppercase() {
                    split_once(arguments, b';').map(|(_, address)| address)
                } else {
                    (!arguments.is_empty()).then_some(arguments)
                };
                let target_vcpu = self.continued.unwrap_or(context.stopped);
                if let Some(address) = address {
                    let Some(rip) = parse_number(address) else {
                        return reply(ERROR_BAD_PACKET);
                    };
                    target::set_rip(context.vcpus, target_vcpu, rip);
                }
                let count = context.vcpus.count();
                let actions = if command.eq_ignore_ascii_case(&b's') {
                    (0..count)
                        .map(|index| match index == target_vcpu {
                            true => ResumeAction::Step,
                            false => ResumeAction::Hold,
                        })
                        .collect()
                } else {
                    vec![ResumeAction::Continue; count]
                };
                (None, Some(Outcome::Resume(actions)))
            }
            b'v' => self.v_packet(data, context),
            b'Z' | b'z' => reply(&self.breakpoint(command == b'Z', arguments, context.config)),
            b'D' => (Some(b"OK".to_vec()), Some(Outcome::Detach)),
            // No reply: the client does not wait for one.
            b'k' => (None, Some(Outcome::Kill)),
            _ => reply(b""),
        }
    }

    fn general_vcpu(&self, context: &Context<'_>) -> usize {
        self.general.unwrap_or(context.stopped)
    }

    fn query(&mut self, data: &[u8], context: &mut Context<'_>) -> Vec<u8> {
        if let Some(features) = data.strip_prefix(b"qSupported") {
            for feature in features
                .strip_prefix(b":")
                .unwrap_or_default()
                .split(|&byte| byte == b';')
            {
                match feature {
                    b"swbreak+" => self.swbreak = true,
                    b"hwbreak+" => self.hwbreak = true,
                    _ => {}
                }
            }
            return format!(
                "PacketSize={MAX_PACKET_BYTES:x};qXfer:features:read+;QStartNoAckMode+;\
                 vContSupported+;swbreak+;hwbreak+"
            )
            .into_bytes();
        }
        if data == b"QStartNoAckMode" {
            // This packet itself was still acknowledged; from its reply on, nothing is.
            self.no_ack = true;
            return b"OK".to_vec();
        }
        if let Some(request) = data.strip_prefix(b"qXfer:features:read:") {
            return features_read(request);
        }
        if data == b"qAttached" {
            return b"1".to_vec();
        }
        if data == b"qC" {
            let mut reply = b"QC".to_vec();
            reply.extend(number(context.stopped as u64 + 1));
            return reply;
        }
        if data == b"qfThreadInfo" {
            let ids: Vec<String> = (1..=context.vcpus.count())
                .map(|id| format!("{id:x}"))
                .collect();
            return format!("m{}", ids.join(",")).into_bytes();
        }
        if data == b"qsThreadInfo" {
            return b"l".to_vec();
        }
        if let Some(thread) = data.strip_prefix(b"qThreadExtraInfo,") {
            return match parse_thread(thread, context.vcpus.count()) {
                Some(ThreadId::Vcpu(index)) => {
                    to_hex(target::describe(context.vcpus, index).as_bytes())
                }
                _ => ERROR_NO_THREAD.to_vec(),
            };
        }
        if data.starts_with(b"qSymbol") {
            return b"OK".to_vec();
        }
        Vec::new()
    }

    fn v_packet(
        &mut self,
        data: &[u8],
        context: &mut Context<'_>,
    ) -> (Option<Vec<u8>>, Option<Outcome>) {
        if data == b"vCont?" {
            return (Some(b"vCont;c;C;s;S;t".to_vec()), None);
        }
        if let Some(actions) = data.strip_prefix(b"vCont;") {
            return match parse_vcont(actions, context.vcpus.count()) {
                Some(actions) => (None, Some(Outcome::Resume(actions))),
                None => (Some(ERROR_BAD_PACKET.to_vec()), None),
            };
        }
        if data.starts_with(b"vKill") {
            return (Some(b"OK".to_vec()), Some(Outcome::Kill));
        }
        // Everything else, `vMustReplyEmpty` included, is not supported: an empty reply.
        (Some(Vec::new()), None)
    }

    /// `Z`/`z` type,address,kind: insert or remove a breakpoint or watchpoint.
    fn breakpoint(&mut self, insert: bool, arguments: &[u8], config: &mut DebugConfig) -> Vec<u8> {
        let mut fields = arguments.split(|&byte| byte == b',');
        let (Some(&[kind_digit]), Some(address), Some(size)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return ERROR_BAD_PACKET.to_vec();
        };
        // Target conditions and commands (`;X...`) are not offered, so the size field ends here.
        let size = size.split(|&byte| byte == b';').next().unwrap_or_default();
        let (Some(address), Some(size)) = (parse_number(address), parse_number(size)) else {
            return ERROR_BAD_PACKET.to_vec();
        };
        let watch_kind = match kind_digit {
            BREAKPOINT_SOFTWARE | BREAKPOINT_HARDWARE => {
                let software = kind_digit == BREAKPOINT_SOFTWARE;
                let entry = config.breakpoints.entry(address).or_default();
                let flag = if software {
                    &mut entry.software
                } else {
                    &mut entry.hardware
                };
                *flag = insert;
                if *entry == BreakpointKinds::default() {
                    config.breakpoints.remove(&address);
                }
                return b"OK".to_vec();
            }
            WATCH_WRITE => WatchKind::Write,
            WATCH_READ => WatchKind::Read,
            WATCH_ACCESS => WatchKind::Access,
            _ => return Vec::new(),
        };
        if size == 0 || size > MAX_WATCH_BYTES {
            return ERROR_REFUSED.to_vec();
        }
        let watchpoint = Watchpoint {
            address,
            length: size,
            kind: watch_kind,
        };
        if insert {
            if !config.watchpoints.contains(&watchpoint) {
                config.watchpoints.push(watchpoint);
            }
        } else {
            config
                .watchpoints
                .retain(|existing| *existing != watchpoint);
        }
        b"OK".to_vec()
    }
}

/// What a packet works on.
struct Context<'a> {
    vcpus: &'a mut dyn Vcpus,
    config: &'a mut DebugConfig,
    last_stop: Option<StopEvent>,
    /// The vCPU that stopped: the default for every packet that takes a thread.
    stopped: usize,
}

fn split_once(bytes: &[u8], separator: u8) -> Option<(&[u8], &[u8])> {
    let at = bytes.iter().position(|&byte| byte == separator)?;
    Some((&bytes[..at], &bytes[at + 1..]))
}

fn parse_address_length(text: &[u8]) -> Option<(u64, u64)> {
    let (address, length) = split_once(text, b',')?;
    Some((parse_number(address)?, parse_number(length)?))
}

fn parse_thread(text: &[u8], count: usize) -> Option<ThreadId> {
    match text {
        b"-1" => Some(ThreadId::All),
        b"0" => Some(ThreadId::Any),
        _ => {
            let id = parse_number(text)? as usize;
            (1..=count).contains(&id).then(|| ThreadId::Vcpu(id - 1))
        }
    }
}

/// `vCont;action[:thread];...`: each vCPU takes the leftmost action that names it or names no
/// thread; one that none matches is held.
fn parse_vcont(text: &[u8], count: usize) -> Option<Vec<ResumeAction>> {
    let mut actions: Vec<Option<ResumeAction>> = vec![None; count];
    for entry in text.split(|&byte| byte == b';') {
        let (action, thread) = match split_once(entry, b':') {
            Some((action, thread)) => (action, Some(parse_thread(thread, count)?)),
            None => (entry, None),
        };
        let action = match action.first()? {
            b'c' | b'C' => ResumeAction::Continue,
            b's' | b'S' => ResumeAction::Step,
            b't' => ResumeAction::Hold,
            _ => return None,
        };
        for (index, slot) in actions.iter_mut().enumerate() {
            let named = match thread {
                None | Some(ThreadId::All | ThreadId::Any) => true,
                Some(ThreadId::Vcpu(vcpu)) => vcpu == index,
            };
            if named && slot.is_none() {
                *slot = Some(action);
            }
        }
    }
    Some(
        actions
            .into_iter()
            .map(|action| action.unwrap_or(ResumeAction::Hold))
            .collect(),
    )
}

/// `qXfer:features:read:annex:offset,length`: a window of the target description, `m` if more
/// follows and `l` for the last part.
fn features_read(request: &[u8]) -> Vec<u8> {
    let Some((annex, window)) = split_once(request, b':') else {
        return ERROR_BAD_PACKET.to_vec();
    };
    if annex != b"target.xml" {
        return ERROR_NO_ANNEX.to_vec();
    }
    let Some((offset, length)) = parse_address_length(window) else {
        return ERROR_BAD_PACKET.to_vec();
    };
    let document = target_description();
    let start = (offset as usize).min(document.len());
    // Escaping can double a byte, so stay within half a packet.
    let end = start
        + (length as usize)
            .min(MAX_PACKET_BYTES / 2)
            .min(document.len() - start);
    let mut reply = vec![if end < document.len() { b'm' } else { b'l' }];
    reply.extend_from_slice(&document.as_bytes()[start..end]);
    reply
}

/// The target description: gdb's i386 features, which gdb needs to treat the target as amd64,
/// and one of our own for the system registers.
pub(super) fn target_description() -> String {
    let mut xml = String::from(
        "<?xml version=\"1.0\"?>\n<!DOCTYPE target SYSTEM \"gdb-target.dtd\">\n\
         <target version=\"1.0\">\n<architecture>i386:x86-64</architecture>\n",
    );
    for (name, registers) in FEATURES {
        xml += &format!("<feature name=\"{name}\">\n");
        if registers.start == REG_XMM0 {
            // The type of the xmm registers, as gdb's own description of the SSE unit has it.
            xml += "<vector id=\"v4f\" type=\"ieee_single\" count=\"4\"/>\n\
                    <vector id=\"v2d\" type=\"ieee_double\" count=\"2\"/>\n\
                    <vector id=\"v16i8\" type=\"int8\" count=\"16\"/>\n\
                    <vector id=\"v8i16\" type=\"int16\" count=\"8\"/>\n\
                    <vector id=\"v4i32\" type=\"int32\" count=\"4\"/>\n\
                    <vector id=\"v2i64\" type=\"int64\" count=\"2\"/>\n\
                    <union id=\"vec128\"><field name=\"v4_float\" type=\"v4f\"/>\
                    <field name=\"v2_double\" type=\"v2d\"/><field name=\"v16_int8\" type=\"v16i8\"/>\
                    <field name=\"v8_int16\" type=\"v8i16\"/><field name=\"v4_int32\" type=\"v4i32\"/>\
                    <field name=\"v2_int64\" type=\"v2i64\"/><field name=\"uint128\" type=\"uint128\"/>\
                    </union>\n";
        }
        for number in registers {
            let register = &REGISTERS[number];
            let group = register
                .group
                .map(|group| format!(" group=\"{group}\""))
                .unwrap_or_default();
            xml += &format!(
                "<reg name=\"{}\" bitsize=\"{}\" type=\"{}\" regnum=\"{number}\"{group}/>\n",
                register.name, register.bits, register.kind
            );
        }
        xml += "</feature>\n";
    }
    xml += "</target>\n";
    debug_assert_eq!(xml.matches("<reg ").count(), REGISTER_COUNT);
    xml
}
