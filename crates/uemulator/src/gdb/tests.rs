//! The framing and every packet, against a real `Cpu` (a `Harness`) as the stopped machine.

use super::engine::{Engine, Outcome, target_description};
use super::packet::{Decoder, Incoming, frame, from_hex, to_hex};
use super::*;
use crate::cpu::gdb::{OneVcpu, SharedVcpus};
use crate::testkit::{CODE_BASE, Harness, RAX, RBX, RCX, SCRATCH};

fn framed(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    frame(payload, &mut out);
    out
}

fn decode(bytes: &[u8]) -> Vec<Incoming> {
    let mut out = Vec::new();
    Decoder::default().feed(bytes, &mut out);
    out
}

#[test]
fn a_frame_carries_the_sum_of_its_bytes() {
    assert_eq!(framed(b"OK"), b"$OK#9a");
    assert_eq!(framed(b""), b"$#00");
}

#[test]
fn framing_escapes_what_would_read_as_framing_or_run_length() {
    let out = framed(b"a$b#c}d*e");
    assert!(out.starts_with(b"$a}\x04b}\x03c}]d}\x0ae#"), "{out:?}");
    // And the decoder takes it back to what was framed.
    assert_eq!(decode(&out), vec![Incoming::Packet(b"a$b#c}d*e".to_vec())]);
}

#[test]
fn the_decoder_reassembles_packets_split_anywhere() {
    let stream = [
        b"+".as_slice(),
        &framed(b"qSupported:swbreak+"),
        b"-",
        &[0x03],
        &framed(b"g"),
    ]
    .concat();
    let whole = decode(&stream);
    let mut decoder = Decoder::default();
    let mut pieces = Vec::new();
    for byte in &stream {
        decoder.feed(core::slice::from_ref(byte), &mut pieces);
    }
    assert_eq!(whole, pieces);
    assert_eq!(
        whole,
        vec![
            Incoming::Ack,
            Incoming::Packet(b"qSupported:swbreak+".to_vec()),
            Incoming::Nack,
            Incoming::Interrupt,
            Incoming::Packet(b"g".to_vec()),
        ]
    );
}

#[test]
fn a_wrong_checksum_or_a_runaway_packet_is_corrupt() {
    assert_eq!(decode(b"$OK#00"), vec![Incoming::Corrupt]);
    assert_eq!(decode(b"$OK#zz"), vec![Incoming::Corrupt]);
    let mut huge = vec![b'$'];
    huge.extend(vec![b'a'; packet::MAX_PACKET_BYTES + 1]);
    let sum = (b'a' as usize * (packet::MAX_PACKET_BYTES + 1)) as u8;
    huge.push(b'#');
    huge.extend(packet::hex_byte(sum));
    assert_eq!(decode(&huge), vec![Incoming::Corrupt]);
    // The decoder recovers at the next packet.
    assert_eq!(
        decode(&[b"$OK#00".as_slice(), &framed(b"?")].concat()),
        vec![Incoming::Corrupt, Incoming::Packet(b"?".to_vec())]
    );
}

#[test]
fn hex_helpers_round_trip_and_reject_garbage() {
    assert_eq!(to_hex(&[0x00, 0xab, 0x7f]), b"00ab7f");
    assert_eq!(from_hex(b"00AB7f"), Some(vec![0x00, 0xab, 0x7f]));
    assert_eq!(from_hex(b"abc"), None);
    assert_eq!(from_hex(b"zz"), None);
    assert_eq!(
        packet::parse_number(b"ffffffff81000000"),
        Some(0xffff_ffff_8100_0000)
    );
    assert_eq!(packet::parse_number(b""), None);
    assert_eq!(packet::parse_number(b"10000000000000000"), None);
}

/// A connection to a stopped one-vCPU machine.
struct Session {
    engine: Engine,
    harness: Harness,
    config: DebugConfig,
    stop: StopEvent,
}

impl Session {
    fn new(code: &[u8]) -> Self {
        let mut session = Session {
            engine: Engine::new(),
            harness: Harness::new(code),
            config: DebugConfig::default(),
            stop: StopEvent {
                vcpu: 0,
                kind: StopKind::Quiet,
            },
        };
        // Acknowledgments are covered on their own; everywhere else they would be noise.
        let (reply, _) = session.exchange(b"QStartNoAckMode");
        assert_eq!(reply, b"OK");
        session
    }

    /// Sends one packet and returns the payload of the reply (empty for none) and what it asks
    /// of the machine.
    fn exchange(&mut self, payload: &[u8]) -> (Vec<u8>, Option<Outcome>) {
        let mut out = Vec::new();
        let outcome = self.engine.process(
            &framed(payload),
            &mut OneVcpu(&mut self.harness.cpu),
            &mut self.config,
            Some(self.stop),
            &mut out,
        );
        let reply = decode(&out)
            .into_iter()
            .find_map(|item| match item {
                Incoming::Packet(data) => Some(data),
                _ => None,
            })
            .unwrap_or_default();
        (reply, outcome)
    }

    fn ask(&mut self, payload: &str) -> String {
        let (reply, outcome) = self.exchange(payload.as_bytes());
        assert_eq!(outcome, None, "{payload}");
        String::from_utf8(reply).unwrap()
    }

    fn register(&mut self, number: usize) -> u64 {
        let bytes = from_hex(self.ask(&format!("p{number:x}")).as_bytes()).unwrap();
        let mut wide = [0u8; 8];
        wide[..bytes.len().min(8)].copy_from_slice(&bytes[..bytes.len().min(8)]);
        u64::from_le_bytes(wide)
    }
}

#[test]
fn every_packet_is_acknowledged_until_no_ack_mode_and_a_nack_resends() {
    let mut harness = Harness::new(&[0xf4]);
    let mut engine = Engine::new();
    let mut config = DebugConfig::default();
    let mut run = |engine: &mut Engine, bytes: &[u8]| {
        let mut out = Vec::new();
        engine.process(
            bytes,
            &mut OneVcpu(&mut harness.cpu),
            &mut config,
            None,
            &mut out,
        );
        out
    };
    assert_eq!(run(&mut engine, &framed(b"qAttached")), b"+$1#31");
    assert_eq!(run(&mut engine, b"-"), b"$1#31");
    assert_eq!(run(&mut engine, b"$qAttached#00"), b"-");
    assert_eq!(run(&mut engine, &framed(b"QStartNoAckMode")), b"+$OK#9a");
    assert_eq!(run(&mut engine, &framed(b"qAttached")), b"$1#31");
    assert_eq!(run(&mut engine, b"$qAttached#00"), b"");
    assert_eq!(run(&mut engine, b"+"), b"");
}

#[test]
fn q_supported_offers_the_target_description_and_learns_the_stop_reasons() {
    let mut session = Session::new(&[0xf4]);
    let reply = session.ask("qSupported:multiprocess+;swbreak+;hwbreak+;xmlRegisters=i386");
    for feature in [
        "PacketSize=4000",
        "qXfer:features:read+",
        "QStartNoAckMode+",
        "vContSupported+",
    ] {
        assert!(reply.contains(feature), "{reply}");
    }
    session.config.breakpoints.insert(
        0x1000,
        BreakpointKinds {
            software: true,
            hardware: false,
        },
    );
    session.stop = StopEvent {
        vcpu: 0,
        kind: StopKind::Breakpoint(0x1000),
    };
    assert_eq!(session.ask("?"), "T05thread:1;swbreak:;");
}

#[test]
fn stop_replies_name_the_thread_and_the_reason() {
    let mut session = Session::new(&[0xf4]);
    let engine = Engine::new();
    let config = DebugConfig::default();
    let reply =
        |kind| String::from_utf8(engine.stop_reply(&StopEvent { vcpu: 2, kind }, &config)).unwrap();
    assert_eq!(reply(StopKind::Quiet), "T05thread:3;");
    assert_eq!(reply(StopKind::Interrupted), "T02thread:3;");
    assert_eq!(reply(StopKind::Step), "T05thread:3;");
    assert_eq!(
        reply(StopKind::Watch(WatchKind::Write, 0x40)),
        "T05thread:3;watch:40;"
    );
    assert_eq!(
        reply(StopKind::Watch(WatchKind::Read, 0x40)),
        "T05thread:3;rwatch:40;"
    );
    assert_eq!(
        reply(StopKind::Watch(WatchKind::Access, 0x40)),
        "T05thread:3;awatch:40;"
    );
    // Without `swbreak+` from the client a breakpoint is a plain SIGTRAP.
    assert_eq!(reply(StopKind::Breakpoint(0x40)), "T05thread:3;");
    assert_eq!(session.ask("?"), "T05thread:1;");
}

#[test]
fn the_target_description_is_served_in_windows_and_names_every_register() {
    let mut session = Session::new(&[0xf4]);
    let mut document = String::new();
    loop {
        let reply = session.ask(&format!(
            "qXfer:features:read:target.xml:{:x},200",
            document.len()
        ));
        let (marker, data) = reply.split_at(1);
        document += data;
        if marker == "l" {
            break;
        }
        assert_eq!(marker, "m");
    }
    assert_eq!(document, target_description());
    for name in [
        "rax",
        "rip",
        "eflags",
        "fs_base",
        "gs_base",
        "k_gs_base",
        "cr3",
        "efer",
        "xmm15",
        "mxcsr",
        "st7",
        "fop",
    ] {
        assert!(
            document.contains(&format!("<reg name=\"{name}\"")),
            "{name}"
        );
    }
    assert!(document.contains("<architecture>i386:x86-64</architecture>"));
    assert!(document.contains("org.gnu.gdb.i386.core"));
    assert_eq!(session.ask("qXfer:features:read:other.xml:0,100"), "E00");
}

/// `count` vCPUs of one machine, the first owning RAM and devices as after a time-shared run.
fn vcpus(count: usize) -> Vec<crate::cpu::Cpu> {
    let devices = crate::dev::DevConfig {
        cpus: count,
        ..Default::default()
    };
    let bootstrap = Harness::with_devices(&[0xf4], devices).cpu;
    crate::cpu::smp::secondary_cpus(bootstrap)
}

fn two_vcpus() -> Vec<crate::cpu::Cpu> {
    vcpus(2)
}

#[test]
fn thread_queries_list_every_vcpu_from_one() {
    let mut cpus = two_vcpus();
    cpus[1].halted = true;
    cpus[1].gpr[RAX] = 0x2222;
    let mut engine = Engine::new();
    let mut config = DebugConfig::default();
    let stop = StopEvent {
        vcpu: 1,
        kind: StopKind::Quiet,
    };
    let mut ask = |payload: &str| {
        let mut out = Vec::new();
        let outcome = engine.process(
            &framed(payload.as_bytes()),
            &mut SharedVcpus {
                cpus: &mut cpus,
                active: 0,
            },
            &mut config,
            Some(stop),
            &mut out,
        );
        let reply = decode(&out)
            .into_iter()
            .find_map(|item| match item {
                Incoming::Packet(data) => Some(String::from_utf8(data).unwrap()),
                _ => None,
            })
            .unwrap_or_default();
        (reply, outcome)
    };
    ask("QStartNoAckMode");
    assert_eq!(ask("qfThreadInfo").0, "m1,2");
    assert_eq!(ask("qsThreadInfo").0, "l");
    assert_eq!(ask("qC").0, "QC2");
    assert_eq!(ask("T2").0, "OK");
    assert_eq!(ask("T3").0, "E02");
    let info = ask("qThreadExtraInfo,2").0;
    assert_eq!(
        from_hex(info.as_bytes()).unwrap(),
        b"CPU#1 [waiting for start-up]"
    );
    // `g` defaults to the vCPU that stopped; `Hg` picks another.
    assert!(ask("g").0.starts_with("2222000000000000"));
    assert_eq!(ask("Hg1").0, "OK");
    assert!(ask("g").0.starts_with("0000000000000000"));
    assert_eq!(ask("Hg5").0, "E02");
    // A legacy `s` steps the `Hc` thread and holds the rest.
    assert_eq!(ask("Hc1").0, "OK");
    assert_eq!(
        ask("s").1,
        Some(Outcome::Resume(vec![
            ResumeAction::Step,
            ResumeAction::Hold
        ]))
    );
    // A resume forgets the selection: the next stop picks the stopped vCPU again.
    assert_eq!(
        ask("s").1,
        Some(Outcome::Resume(vec![
            ResumeAction::Hold,
            ResumeAction::Step
        ]))
    );
}

#[test]
fn registers_read_in_gdbs_order() {
    let mut session = Session::new(&[0xf4]);
    let cpu = &mut session.harness.cpu;
    cpu.gpr[RAX] = 0x1111;
    cpu.gpr[RBX] = 0x2222;
    cpu.gpr[RCX] = 0x3333;
    cpu.xmm[3] = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
    cpu.mxcsr = 0x1f80;
    cpu.kgsbase = 0xffff_8880_0000_0000;
    cpu.seg[crate::cpu::SEG_FS].base = 0x7000;
    let (rip, cr3, efer) = (cpu.rip, cpu.cr3, cpu.efer);
    assert_eq!(session.register(0), 0x1111);
    assert_eq!(session.register(1), 0x2222);
    assert_eq!(session.register(2), 0x3333);
    assert_eq!(session.register(16), rip);
    assert_eq!(session.register(57), 0x7000);
    assert_eq!(session.register(59), 0xffff_8880_0000_0000);
    assert_eq!(session.register(62), cr3);
    assert_eq!(session.register(65), efer);
    assert_eq!(session.register(56), 0x1f80);
    assert_eq!(session.ask("p2b"), "1032547698badcfeefcdab8967452301");
    // The `g` packet is every register in this order: 16 gprs, rip, eflags, 6 selectors, ...
    let all = from_hex(session.ask("g").as_bytes()).unwrap();
    assert_eq!(
        all.len(),
        16 * 8 + 8 + 4 + 6 * 4 + 8 * 10 + 8 * 4 + 16 * 16 + 4 + 9 * 8
    );
    assert_eq!(&all[16 * 8..16 * 8 + 8], &rip.to_le_bytes());
    assert_eq!(session.ask("p42"), "E01");
}

#[test]
fn register_writes_change_state_and_refuse_what_needs_the_cpu_to_react() {
    let mut session = Session::new(&[0xf4]);
    assert_eq!(session.ask("P0=efbeadde00000000"), "OK");
    assert_eq!(session.harness.cpu.gpr[RAX], 0xdead_beef);
    assert_eq!(session.ask(&format!("P10={}", hex64(0x1234_5678))), "OK");
    assert_eq!(session.harness.cpu.rip, 0x1234_5678);
    // Changing CR3 is refused; writing back the value it holds is not.
    let cr3 = session.harness.cpu.cr3;
    assert_eq!(session.ask(&format!("P3e={}", hex64(cr3 + 0x1000))), "E03");
    assert_eq!(session.ask(&format!("P3e={}", hex64(cr3))), "OK");
    // The flags, through the lazy-flag state.
    assert_eq!(session.ask("P11=c3000000"), "OK");
    assert_eq!(session.harness.cpu.eflags_full() & 0xfff, 0xc3);
    assert_eq!(session.ask("P11=02000200"), "E03", "setting VM");
    assert_eq!(session.ask("P0=1234"), "E03", "a wrong width");
    assert_eq!(session.ask("Pzz=00"), "E01");
    // `G` writes everything at once, refusing as a whole.
    let all = session.ask("g");
    let mut bytes = from_hex(all.as_bytes()).unwrap();
    bytes[8..16].copy_from_slice(&0x77u64.to_le_bytes());
    assert_eq!(
        session.ask(&format!("G{}", String::from_utf8(to_hex(&bytes)).unwrap())),
        "OK"
    );
    assert_eq!(session.harness.cpu.gpr[RBX], 0x77);
    let cr0_offset = 16 * 8 + 8 + 4 + 6 * 4 + 8 * 10 + 8 * 4 + 16 * 16 + 4 + 3 * 8;
    bytes[0..8].copy_from_slice(&0x99u64.to_le_bytes());
    bytes[cr0_offset] ^= 0x20;
    assert_eq!(
        session.ask(&format!("G{}", String::from_utf8(to_hex(&bytes)).unwrap())),
        "E03"
    );
    assert_eq!(
        session.harness.cpu.gpr[RAX], 0xdead_beef,
        "nothing was written"
    );
}

fn hex64(value: u64) -> String {
    String::from_utf8(to_hex(&value.to_le_bytes())).unwrap()
}

#[test]
fn x87_registers_are_relative_to_top_with_the_full_tag_word() {
    let mut session = Session::new(&[0xf4]);
    let cpu = &mut session.harness.cpu;
    // TOP = 6: ST(0) is physical 6, ST(2) physical 0.
    cpu.fsw = 6 << 11;
    cpu.st[6] = [0, 0, 0, 0, 0, 0, 0, 0x80, 0xff, 0x3f]; // 1.0, valid
    cpu.st[7] = [0; 10]; // +0, zero
    cpu.st[0] = [0, 0, 0, 0, 0, 0, 0, 0xc0, 0xff, 0x7f]; // a NaN, special
    cpu.ftw = (1 << 6) | (1 << 7) | 1;
    assert_eq!(session.ask("p18"), "0000000000000080ff3f");
    assert_eq!(session.ask("p1a"), "00000000000000c0ff7f");
    // Physical 0 special, 1-5 empty, 6 valid, 7 zero.
    assert_eq!(session.register(34), 0b0100_1111_1111_1110);
    // Writing the tag word back keeps the abridged form.
    assert_eq!(session.ask("P22=ff3f0000"), "OK");
    assert_eq!(session.harness.cpu.ftw, 0x80);
}

#[test]
fn memory_reads_and_writes_go_through_the_page_tables_without_touching_them() {
    let mut session = Session::new(&[0xf4]);
    // The fake kernel identity-maps low memory.
    session.harness.write(SCRATCH, b"hello");
    assert_eq!(session.ask(&format!("m{SCRATCH:x},5")), "68656c6c6f");
    // A higher-half mapping of our own: PML4 -> PDPT -> PD -> PT at SCRATCH + 0x10000 on.
    const VIRTUAL: u64 = 0xffff_8000_0020_3000;
    const TARGET: u64 = SCRATCH + 0x8000;
    const PRESENT_WRITABLE: u64 = 3;
    let tables = SCRATCH + 0x10000;
    let cr3 = session.harness.cpu.cr3 & !0xfff;
    let index = |level: u32| (VIRTUAL >> (12 + 9 * level)) & 0x1ff;
    session
        .harness
        .write_u64(cr3 + 8 * index(3), tables | PRESENT_WRITABLE);
    session
        .harness
        .write_u64(tables + 8 * index(2), (tables + 0x1000) | PRESENT_WRITABLE);
    session.harness.write_u64(
        tables + 0x1000 + 8 * index(1),
        (tables + 0x2000) | PRESENT_WRITABLE,
    );
    session
        .harness
        .write_u64(tables + 0x2000 + 8 * index(0), TARGET | PRESENT_WRITABLE);
    session.harness.write(TARGET + 0xff0, b"0123456789abcdefXY");
    let cr2 = session.harness.cpu.cr2;
    assert_eq!(
        session.ask(&format!("m{:x},4", VIRTUAL + 0xff0)),
        "30313233"
    );
    // Accessed was not set at any level, and no fault left its address.
    assert_eq!(
        session.harness.read_u64(tables + 0x2000 + 8 * index(0)) & 0x20,
        0
    );
    assert_eq!(session.harness.read_u64(tables + 8 * index(2)) & 0x20, 0);
    assert_eq!(session.harness.cpu.cr2, cr2);
    // A read that runs into an unmapped page stops there; one that starts there fails.
    assert_eq!(session.ask(&format!("m{:x},20", VIRTUAL + 0xffe)), "6566");
    assert_eq!(session.ask(&format!("m{:x},4", VIRTUAL + 0x1000)), "E14");
    assert_eq!(session.ask("m0,0"), "");
    // A 2 MiB leaf.
    session.harness.write_u64(
        tables + 0x1000 + 8 * (index(1) + 1),
        SCRATCH | 0x80 | PRESENT_WRITABLE,
    );
    session.harness.write(SCRATCH + 0x1234, b"\x5a");
    assert_eq!(
        session.ask(&format!("m{:x},1", VIRTUAL - 0x3000 + 0x20_0000 + 0x1234)),
        "5a"
    );
    // Writes, hex and binary, land in the page the mapping names.
    assert_eq!(session.ask(&format!("M{:x},2:abcd", VIRTUAL)), "OK");
    assert_eq!(session.harness.read(TARGET, 2), [0xab, 0xcd]);
    let mut binary = format!("X{:x},3:", VIRTUAL + 2).into_bytes();
    binary.extend_from_slice(b"#}*");
    let (reply, _) = session.exchange(&binary);
    assert_eq!(reply, b"OK");
    assert_eq!(session.harness.read(TARGET + 2, 3), b"#}*");
    // A write that would run off the mapping writes nothing.
    assert_eq!(
        session.ask(&format!("M{:x},4:01020304", VIRTUAL + 0xffe)),
        "E14"
    );
    assert_eq!(session.harness.read(TARGET + 0xffe, 2), b"ef");
    assert_eq!(session.ask("M0,2:01"), "E01", "length and data disagree");
}

#[test]
fn a_memory_write_to_translated_code_drops_it() {
    let mut session = Session::new(&[0x48, 0xc7, 0xc0, 0x01, 0, 0, 0, 0xf4]);
    if !session.harness.enable_jit() {
        return;
    }
    session.harness.run_to_halt();
    assert_eq!(session.harness.cpu.gpr[RAX], 1);
    let page = (CODE_BASE >> 12) as usize;
    assert_ne!(session.harness.cpu.page_has_code.get(page), 0);
    // mov rax, 1 becomes mov rax, 2.
    assert_eq!(session.ask(&format!("M{:x},1:02", CODE_BASE + 3)), "OK");
    assert_eq!(session.harness.cpu.page_has_code.get(page), 0);
    session.harness.cpu.rip = CODE_BASE;
    session.harness.cpu.halted = false;
    session.harness.run_to_halt();
    assert_eq!(session.harness.cpu.gpr[RAX], 2);
}

#[test]
fn breakpoint_and_watchpoint_packets_edit_the_configuration() {
    let mut session = Session::new(&[0xf4]);
    assert_eq!(session.ask("Z0,1000,1"), "OK");
    assert_eq!(session.ask("Z1,1000,1"), "OK");
    assert_eq!(
        session.config.breakpoints[&0x1000],
        BreakpointKinds {
            software: true,
            hardware: true
        }
    );
    assert_eq!(session.ask("z0,1000,1"), "OK");
    assert!(session.config.breakpoints.contains_key(&0x1000));
    assert_eq!(session.ask("z1,1000,1"), "OK");
    assert!(session.config.breakpoints.is_empty());
    assert_eq!(
        session.ask("Z0,2000,1;X1,ff"),
        "OK",
        "conditions are ignored"
    );
    assert_eq!(session.ask("Z2,3000,8"), "OK");
    assert_eq!(session.ask("Z3,3000,4"), "OK");
    assert_eq!(session.ask("Z4,3000,2"), "OK");
    assert_eq!(session.ask("Z2,3000,8"), "OK", "twice is once");
    assert_eq!(session.config.watchpoints.len(), 3);
    assert_eq!(session.ask("z3,3000,4"), "OK");
    assert_eq!(
        session.config.watchpoints,
        vec![
            Watchpoint {
                address: 0x3000,
                length: 8,
                kind: WatchKind::Write
            },
            Watchpoint {
                address: 0x3000,
                length: 2,
                kind: WatchKind::Access
            },
        ]
    );
    assert_eq!(session.ask("Z2,3000,0"), "E03");
    assert_eq!(session.ask("Z2,3000,100000"), "E03");
    assert_eq!(session.ask("Z9,3000,1"), "");
    assert_eq!(session.ask("Z0,zz,1"), "E01");
}

#[test]
fn watchpoints_match_overlapping_accesses_of_their_kind() {
    let watch = |kind| Watchpoint {
        address: 0x1000,
        length: 4,
        kind,
    };
    assert!(watch(WatchKind::Write).matches(0x1003, 1, true));
    assert!(watch(WatchKind::Write).matches(0x0ffc, 8, true));
    assert!(!watch(WatchKind::Write).matches(0x0ffc, 4, true));
    assert!(!watch(WatchKind::Write).matches(0x1004, 8, true));
    assert!(!watch(WatchKind::Write).matches(0x1000, 1, false));
    assert!(watch(WatchKind::Read).matches(0x1000, 1, false));
    assert!(!watch(WatchKind::Read).matches(0x1000, 1, true));
    assert!(watch(WatchKind::Access).matches(0x1002, 2, true));
    let top = Watchpoint {
        address: u64::MAX - 1,
        length: 2,
        kind: WatchKind::Access,
    };
    assert!(top.matches(u64::MAX, 1, false));
}

#[test]
fn resume_packets_say_what_each_vcpu_does() {
    let mut session = Session::new(&[0xf4]);
    let resume = |actions: &[ResumeAction]| Some(Outcome::Resume(actions.to_vec()));
    assert_eq!(
        session.exchange(b"c"),
        (Vec::new(), resume(&[ResumeAction::Continue]))
    );
    assert_eq!(
        session.exchange(b"s"),
        (Vec::new(), resume(&[ResumeAction::Step]))
    );
    assert_eq!(
        session.exchange(b"C05"),
        (Vec::new(), resume(&[ResumeAction::Continue]))
    );
    assert_eq!(session.ask("vCont?"), "vCont;c;C;s;S;t");
    assert_eq!(
        session.exchange(b"vCont;s:1;c"),
        (Vec::new(), resume(&[ResumeAction::Step]))
    );
    assert_eq!(
        session.exchange(b"vCont;c:1"),
        (Vec::new(), resume(&[ResumeAction::Continue]))
    );
    assert_eq!(
        session.exchange(b"vCont;t"),
        (Vec::new(), resume(&[ResumeAction::Hold]))
    );
    assert_eq!(session.ask("vCont;x"), "E01");
    assert_eq!(session.ask("vCont;c:9"), "E01");
    // `c addr` resumes at addr.
    assert_eq!(
        session.exchange(b"c1000").1,
        resume(&[ResumeAction::Continue])
    );
    assert_eq!(session.harness.cpu.rip, 0x1000);
    assert_eq!(
        session.exchange(b"S05;2000").1,
        resume(&[ResumeAction::Step])
    );
    assert_eq!(session.harness.cpu.rip, 0x2000);
    assert_eq!(
        session.exchange(b"D"),
        (b"OK".to_vec(), Some(Outcome::Detach))
    );
    assert_eq!(
        session.exchange(b"D;1"),
        (b"OK".to_vec(), Some(Outcome::Detach))
    );
    assert_eq!(session.exchange(b"k"), (Vec::new(), Some(Outcome::Kill)));
    assert_eq!(
        session.exchange(b"vKill;1"),
        (b"OK".to_vec(), Some(Outcome::Kill))
    );
}

#[test]
fn vcont_takes_the_leftmost_action_for_each_vcpu_and_holds_the_rest() {
    let mut cpus = vcpus(3);
    let mut engine = Engine::new();
    let mut config = DebugConfig::default();
    let mut resume = |payload: &str| {
        let mut out = Vec::new();
        engine.process(
            &framed(payload.as_bytes()),
            &mut SharedVcpus {
                cpus: &mut cpus,
                active: 0,
            },
            &mut config,
            None,
            &mut out,
        )
    };
    use ResumeAction::{Continue, Hold, Step};
    assert_eq!(
        resume("vCont;s:2;c"),
        Some(Outcome::Resume(vec![Continue, Step, Continue]))
    );
    assert_eq!(
        resume("vCont;s:2"),
        Some(Outcome::Resume(vec![Hold, Step, Hold]))
    );
    assert_eq!(
        resume("vCont;c:3;s:-1"),
        Some(Outcome::Resume(vec![Step, Step, Continue]))
    );
}

#[test]
fn packets_after_a_resume_wait_for_the_next_stop() {
    let mut session = Session::new(&[0xf4]);
    let mut out = Vec::new();
    let bytes = [framed(b"c"), framed(b"qAttached")].concat();
    let outcome = session.engine.process(
        &bytes,
        &mut OneVcpu(&mut session.harness.cpu),
        &mut session.config,
        Some(session.stop),
        &mut out,
    );
    assert_eq!(outcome, Some(Outcome::Resume(vec![ResumeAction::Continue])));
    assert!(out.is_empty());
    assert!(session.engine.take_resume_pending());
    assert_eq!(session.ask("qAttached").as_str(), "1");
}

#[test]
fn unknown_packets_get_the_empty_reply() {
    let mut session = Session::new(&[0xf4]);
    for packet in ["vMustReplyEmpty", "qTStatus", "Xfoo", "!", "qOffsets"] {
        let reply = session.ask(packet);
        assert!(
            reply.is_empty() || reply.starts_with('E'),
            "{packet}: {reply}"
        );
    }
    assert_eq!(session.ask("qSymbol::"), "OK");
    assert_eq!(session.ask("qAttached"), "1");
}
