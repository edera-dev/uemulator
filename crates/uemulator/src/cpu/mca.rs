//! Machine-check architecture: AMD's Scalable MCA as on Zen 5 (AMD APM vol. 2 sec. 9.3, the
//! family 1Ah PPR's MCA chapter). The emulated machine has no hardware to fail, so no bank ever
//! logs an error on its own: the registers exist so an OS that finds MCE/MCA/ScalableMca in CPUID
//! can initialize its banks (enable reporting, clear status, name the banks from their IPIDs, set
//! up thresholding and the deferred-error interrupt) without a #GP. `Cpu::machine_check` logs an
//! error and raises #MC the way the hardware would, for tests and for any device model that wants
//! to report one later.
//!
//! Register layout. Each bank has sixteen MSRs at `MSR_SMCA_FIRST + 0x10 * bank`; the first four
//! (CTL, STATUS, ADDR, MISC0) are also reachable at the legacy `0x400 + 4 * bank`, which is what
//! an OS that ignores SMCA (and Xen's vMCE) uses. Linux uses only the SMCA addresses once it sees
//! ScalableMca, and sets CONFIG.McaXEn on every bank to say so.

use super::Cpu;
use super::exception::VECTOR_MACHINE_CHECK;
use super::sys::CR4_MCE;
use crate::machine::ExitReason;
use crate::snapshot::Snapshot;

pub(crate) const MSR_MCG_CAP: u32 = 0x179;
pub(crate) const MSR_MCG_STATUS: u32 = 0x17a;
pub(crate) const MSR_MCG_CTL: u32 = 0x17b;
const MSR_LEGACY_BANK_FIRST: u32 = 0x400;
const LEGACY_REGISTERS_PER_BANK: u32 = 4;
const MSR_SMCA_FIRST: u32 = 0xc000_2000;
const SMCA_REGISTERS_PER_BANK: u32 = 0x10;
/// MCA_INTR_CFG (Linux's MSR_CU_DEF_ERR): the extended-LVT offsets of the deferred-error and
/// thresholding interrupts, which firmware fills in and the OS reads to program the local APIC.
const MSR_MCA_INTR_CFG: u32 = 0xc000_0410;
/// MCA_CTL_MASK, one per bank: error types that are logged but never signaled. Storage only.
const MSR_CTL_MASK_FIRST: u32 = 0xc001_0400;
/// HWCR, kept in `Cpu::msr_misc`. McStatusWrEn lets software write nonzero values to the status
/// registers, which Linux's mce-inject does before `int 18`.
pub(crate) const MSR_HWCR: u32 = 0xc001_0015;
pub(crate) const HWCR_MC_STATUS_WRITE_ENABLE: u64 = 1 << 18;

// Register offsets inside a bank's SMCA block. Offset 7 does not exist (#GP). The legacy aliases
// use offsets 0-3 in the same order.
const CTL: usize = 0;
const STATUS: usize = 1;
const ADDR: usize = 2;
const MISC0: usize = 3;
const CONFIG: usize = 4;
const IPID: usize = 5;
#[cfg(test)]
const SYND: usize = 6;
const NO_REGISTER: usize = 7;
const DESTAT: usize = 8;
const DEADDR: usize = 9;

pub(crate) const BANKS: usize = BANK_LAYOUT.len();
const MSR_LEGACY_BANK_LAST: u32 =
    MSR_LEGACY_BANK_FIRST + BANKS as u32 * LEGACY_REGISTERS_PER_BANK - 1;
const MSR_SMCA_LAST: u32 = MSR_SMCA_FIRST + BANKS as u32 * SMCA_REGISTERS_PER_BANK - 1;
const MSR_CTL_MASK_LAST: u32 = MSR_CTL_MASK_FIRST + BANKS as u32 - 1;

/// MCG_CAP: bank count in 7:0 and MCG_CTL_P (bit 8), the MCG_CTL register exists. AMD sets none
/// of Intel's other capability bits.
const MCG_CAP_CTL_PRESENT: u64 = 1 << 8;
const MCG_CAP_VALUE: u64 = BANKS as u64 | MCG_CAP_CTL_PRESENT;

/// MCG_STATUS: restart IP valid, error IP valid, machine check in progress. A second #MC while
/// MCIP is set shuts the processor down. Other bits are reserved and ignored on write.
pub(crate) const MCG_STATUS_RIPV: u64 = 1 << 0;
pub(crate) const MCG_STATUS_EIPV: u64 = 1 << 1;
pub(crate) const MCG_STATUS_MCIP: u64 = 1 << 2;
const MCG_STATUS_WRITABLE: u64 = MCG_STATUS_RIPV | MCG_STATUS_EIPV | MCG_STATUS_MCIP;

// MCA_STATUS bits (AMD APM vol. 2 sec. 9.3.2.2 and the PPR).
pub(crate) const STATUS_VALID: u64 = 1 << 63;
pub(crate) const STATUS_OVERFLOW: u64 = 1 << 62;
pub(crate) const STATUS_UNCORRECTED: u64 = 1 << 61;
#[cfg(test)]
const STATUS_ENABLED: u64 = 1 << 60;
pub(crate) const STATUS_ADDRESS_VALID: u64 = 1 << 58;
/// Processor context corrupt: the interrupted context cannot be restarted (RIPV stays clear).
pub(crate) const STATUS_CONTEXT_CORRUPT: u64 = 1 << 57;
pub(crate) const STATUS_DEFERRED: u64 = 1 << 44;

/// MCA_CONFIG low half, read-only capabilities: McaX (the SMCA registers exist, bit 0),
/// DeferredErrorLoggingSupported (2) and DeferredIntTypeSupported (5). Linux sets McaXEn (32) and
/// DeferredIntType = APIC (38:37) in the high half, which is stored as written.
const CONFIG_MCAX: u64 = 1 << 0;
const CONFIG_DEFERRED_LOGGING_SUPPORTED: u64 = 1 << 2;
const CONFIG_DEFERRED_INT_TYPE_SUPPORTED: u64 = 1 << 5;
const CONFIG_CAPABILITIES: u64 =
    CONFIG_MCAX | CONFIG_DEFERRED_LOGGING_SUPPORTED | CONFIG_DEFERRED_INT_TYPE_SUPPORTED;
const CONFIG_WRITABLE: u64 = 0xffff_ffff_0000_0000;

/// MCA_MISC0: Valid (63), CntP (62, an error counter exists), IntP (60, it can interrupt
/// through the extended LVT), Locked (61) clear and BlkPtr (31:24) zero, so MISC1-4 are not
/// announced. Linux then programs a threshold into ErrCnt (43:32), clears Ovrflw (48), and sets
/// IntType (50:49), CntEn (51) and, outside SMCA, LvtOffset (55:52): the writable bits.
const MISC0_VALID: u64 = 1 << 63;
const MISC0_COUNTER_PRESENT: u64 = 1 << 62;
const MISC0_INTERRUPT_PRESENT: u64 = 1 << 60;
const MISC0_RESET: u64 = MISC0_VALID | MISC0_COUNTER_PRESENT | MISC0_INTERRUPT_PRESENT;
const MISC0_WRITABLE: u64 = 0x00ff_ffff_0000_0000;

/// MCA_INTR_CFG fields: deferred-error LVT offset (7:4) and threshold LVT offset (15:12). Turin's
/// firmware leaves 2 and 1 there; Linux reports a firmware bug when the deferred one is zero.
const INTR_CFG_DEFERRED_LVT_SHIFT: u32 = 4;
const INTR_CFG_THRESHOLD_LVT_SHIFT: u32 = 12;
const DEFERRED_ERROR_LVT_OFFSET: u64 = 2;
const THRESHOLD_LVT_OFFSET: u64 = 1;
const INTR_CFG_RESET: u64 = DEFERRED_ERROR_LVT_OFFSET << INTR_CFG_DEFERRED_LVT_SHIFT
    | THRESHOLD_LVT_OFFSET << INTR_CFG_THRESHOLD_LVT_SHIFT;
const INTR_CFG_WRITABLE: u64 = 0xffff_ffff;

/// Leaf 0x8000_0007 EBX: McaOverflowRecov (0, an overflowed status is still recoverable), SUCCOR
/// (1, software uncorrectable error containment and recovery, which also makes Linux enable the
/// deferred-error interrupt) and ScalableMca (3).
const ADVANCED_POWER_MCA_OVERFLOW_RECOVERY: u32 = 1 << 0;
const ADVANCED_POWER_SUCCOR: u32 = 1 << 1;
const ADVANCED_POWER_SCALABLE_MCA: u32 = 1 << 3;
pub(crate) const ADVANCED_POWER_EBX: u32 =
    ADVANCED_POWER_MCA_OVERFLOW_RECOVERY | ADVANCED_POWER_SUCCOR | ADVANCED_POWER_SCALABLE_MCA;

/// What a bank's MCA_IPID says it is. Linux names banks by (HardwareID, McaType) from its
/// `smca_hwid_mcatypes` table, so these pairs must be ones it knows. IPID layout: InstanceId in
/// 31:0, HardwareID in 43:32, McaType in 63:48.
#[derive(Clone, Copy)]
struct BankKind {
    hardware_id: u64,
    mca_type: u64,
    instance: u64,
}

const IPID_HARDWARE_ID_SHIFT: u32 = 32;
const IPID_MCA_TYPE_SHIFT: u32 = 48;

impl BankKind {
    const fn ipid(&self) -> u64 {
        self.mca_type << IPID_MCA_TYPE_SHIFT
            | self.hardware_id << IPID_HARDWARE_ID_SHIFT
            | self.instance
    }
}

const HARDWARE_ID_CORE: u64 = 0xb0;
const HARDWARE_ID_DATA_FABRIC: u64 = 0x2e;
const HARDWARE_ID_MEMORY_CONTROLLER: u64 = 0x96;
const HARDWARE_ID_SMU: u64 = 0x01;
const HARDWARE_ID_PARAMETER_BLOCK: u64 = 0x05;
const HARDWARE_ID_NBIO: u64 = 0x18;
const HARDWARE_ID_PCIE: u64 = 0x46;

const fn bank(hardware_id: u64, mca_type: u64, instance: u64) -> Option<BankKind> {
    Some(BankKind {
        hardware_id,
        mca_type,
        instance,
    })
}

/// The banks one thread sees, in Zen 5's order: the core's load-store (McaType 0x10 is Zen 3's
/// LS_V2), instruction fetch, L2, decode, a reserved bank 4 (no hardware: every register reads
/// zero and ignores writes, as on every Zen), execution and floating-point units, the eight L3
/// slices of its complex, then a share of the SoC: MP5, parameter block, two memory controllers
/// (UMC_V2), two coherent slaves (CS_V2), the PIE, NBIO and PCIe (PCIE_V2). A real socket shows
/// up to 64 banks, most of them more instances of the same SoC units; fewer is still a valid
/// Turin to Linux, which sizes everything from MCG_CAP.
const BANK_LAYOUT: [Option<BankKind>; 24] = [
    bank(HARDWARE_ID_CORE, 0x10, 0),
    bank(HARDWARE_ID_CORE, 0x1, 0),
    bank(HARDWARE_ID_CORE, 0x2, 0),
    bank(HARDWARE_ID_CORE, 0x3, 0),
    None,
    bank(HARDWARE_ID_CORE, 0x5, 0),
    bank(HARDWARE_ID_CORE, 0x6, 0),
    bank(HARDWARE_ID_CORE, 0x7, 0),
    bank(HARDWARE_ID_CORE, 0x7, 1),
    bank(HARDWARE_ID_CORE, 0x7, 2),
    bank(HARDWARE_ID_CORE, 0x7, 3),
    bank(HARDWARE_ID_CORE, 0x7, 4),
    bank(HARDWARE_ID_CORE, 0x7, 5),
    bank(HARDWARE_ID_CORE, 0x7, 6),
    bank(HARDWARE_ID_CORE, 0x7, 7),
    bank(HARDWARE_ID_SMU, 0x2, 0),
    bank(HARDWARE_ID_PARAMETER_BLOCK, 0x0, 0),
    bank(HARDWARE_ID_MEMORY_CONTROLLER, 0x1, 0),
    bank(HARDWARE_ID_MEMORY_CONTROLLER, 0x1, 1),
    bank(HARDWARE_ID_DATA_FABRIC, 0x2, 0),
    bank(HARDWARE_ID_DATA_FABRIC, 0x2, 1),
    bank(HARDWARE_ID_DATA_FABRIC, 0x1, 0),
    bank(HARDWARE_ID_NBIO, 0x0, 0),
    bank(HARDWARE_ID_PCIE, 0x1, 0),
];

/// What the bank-register write path needs from outside the MCA: HWCR.McStatusWrEn.
#[derive(Clone, Copy)]
pub(crate) struct WriteContext {
    pub status_writable: bool,
}

#[derive(Clone, Snapshot)]
pub(crate) struct Mca {
    global_status: u64,
    global_control: u64,
    interrupt_config: u64,
    control_mask: [u64; BANKS],
    /// The sixteen SMCA registers of each bank, indexed by offset. CONFIG holds only its high
    /// half (the low half is `CONFIG_CAPABILITIES`), IPID is not stored.
    bank: [[u64; SMCA_REGISTERS_PER_BANK as usize]; BANKS],
}

/// A bank register an MSR number names.
#[derive(Clone, Copy)]
enum Register {
    Bank(usize, usize),
    ControlMask(usize),
    GlobalCapability,
    GlobalStatus,
    GlobalControl,
    InterruptConfig,
}

impl Mca {
    pub fn new() -> Self {
        let mut bank = [[0; SMCA_REGISTERS_PER_BANK as usize]; BANKS];
        for (registers, kind) in bank.iter_mut().zip(BANK_LAYOUT) {
            if kind.is_some() {
                registers[MISC0] = MISC0_RESET;
            }
        }
        Mca {
            global_status: 0,
            global_control: 0,
            interrupt_config: INTR_CFG_RESET,
            control_mask: [0; BANKS],
            bank,
        }
    }

    fn register(msr: u32) -> Option<Register> {
        Some(match msr {
            MSR_MCG_CAP => Register::GlobalCapability,
            MSR_MCG_STATUS => Register::GlobalStatus,
            MSR_MCG_CTL => Register::GlobalControl,
            MSR_MCA_INTR_CFG => Register::InterruptConfig,
            MSR_LEGACY_BANK_FIRST..=MSR_LEGACY_BANK_LAST => {
                let offset = msr - MSR_LEGACY_BANK_FIRST;
                Register::Bank(
                    (offset / LEGACY_REGISTERS_PER_BANK) as usize,
                    (offset % LEGACY_REGISTERS_PER_BANK) as usize,
                )
            }
            MSR_SMCA_FIRST..=MSR_SMCA_LAST => {
                let offset = msr - MSR_SMCA_FIRST;
                let register = (offset % SMCA_REGISTERS_PER_BANK) as usize;
                if register == NO_REGISTER {
                    return None;
                }
                Register::Bank((offset / SMCA_REGISTERS_PER_BANK) as usize, register)
            }
            MSR_CTL_MASK_FIRST..=MSR_CTL_MASK_LAST => {
                Register::ControlMask((msr - MSR_CTL_MASK_FIRST) as usize)
            }
            _ => return None,
        })
    }

    pub fn owns(msr: u32) -> bool {
        Self::register(msr).is_some()
    }

    pub fn read(&self, msr: u32) -> Option<u64> {
        Some(match Self::register(msr)? {
            Register::GlobalCapability => MCG_CAP_VALUE,
            Register::GlobalStatus => self.global_status,
            Register::GlobalControl => self.global_control,
            Register::InterruptConfig => self.interrupt_config,
            Register::ControlMask(bank) => self.control_mask[bank],
            Register::Bank(bank, register) => {
                let Some(kind) = BANK_LAYOUT[bank] else {
                    return Some(0);
                };
                match register {
                    CONFIG => CONFIG_CAPABILITIES | self.bank[bank][CONFIG],
                    IPID => kind.ipid(),
                    _ => self.bank[bank][register],
                }
            }
        })
    }

    /// `false` is #GP: MCG_CAP and IPID are read-only, and a status register accepts a nonzero
    /// value only while HWCR.McStatusWrEn is set (AMD PPR, MCA_STATUS). Read-only fields of
    /// CONFIG and MISC0 keep their value; the reserved bank ignores everything.
    pub fn write(&mut self, msr: u32, value: u64, context: WriteContext) -> bool {
        let Some(target) = Self::register(msr) else {
            return false;
        };
        match target {
            Register::GlobalCapability => return false,
            Register::GlobalStatus => self.global_status = value & MCG_STATUS_WRITABLE,
            Register::GlobalControl => self.global_control = value,
            Register::InterruptConfig => {
                if value & !INTR_CFG_WRITABLE != 0 {
                    return false;
                }
                self.interrupt_config = value;
            }
            Register::ControlMask(bank) => self.control_mask[bank] = value,
            Register::Bank(bank, register) => {
                if BANK_LAYOUT[bank].is_none() {
                    return true;
                }
                let stored = &mut self.bank[bank][register];
                match register {
                    IPID => return false,
                    STATUS | DESTAT => {
                        if value != 0 && !context.status_writable {
                            return false;
                        }
                        *stored = value;
                    }
                    CONFIG => *stored = value & CONFIG_WRITABLE,
                    MISC0 => *stored = *stored & !MISC0_WRITABLE | value & MISC0_WRITABLE,
                    _ => *stored = value,
                }
            }
        }
        true
    }

    /// Records an error in `bank` as the hardware logs it and says whether it is signaled as #MC.
    /// A bank that already holds a valid error gets the overflow bit; an uncorrected error
    /// replaces a corrected one but never the other way round. A deferred error also lands in
    /// DESTAT/DEADDR. Only an uncorrected error whose bank is enabled in both MCG_CTL and its
    /// MCi_CTL is signaled; everything else is just logged, as with reporting turned off.
    fn log(&mut self, bank: usize, status: u64, address: u64) -> bool {
        if bank >= BANKS || BANK_LAYOUT[bank].is_none() {
            return false;
        }
        let registers = &mut self.bank[bank];
        let previous = registers[STATUS];
        let mut status = status | STATUS_VALID;
        if previous & STATUS_VALID != 0 {
            let keep_previous =
                previous & STATUS_UNCORRECTED != 0 && status & STATUS_UNCORRECTED == 0;
            if keep_previous {
                registers[STATUS] = previous | STATUS_OVERFLOW;
                return false;
            }
            status |= STATUS_OVERFLOW;
        }
        registers[STATUS] = status;
        if status & STATUS_ADDRESS_VALID != 0 {
            registers[ADDR] = address;
        }
        if status & STATUS_DEFERRED != 0 {
            registers[DESTAT] = status;
            registers[DEADDR] = address;
        }
        let enabled = self.global_control & (1 << bank) != 0 && registers[CTL] != 0;
        status & STATUS_UNCORRECTED != 0 && enabled
    }
}

/// An error for `Cpu::machine_check`: the bank it is logged in, its MCA_STATUS (VAL is implied)
/// and, with ADDRV, its address.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MachineCheck {
    pub bank: usize,
    pub status: u64,
    pub address: u64,
}

impl Cpu {
    /// Logs `error` and, when it is signaled, raises #MC (vector 18, no error code). MCG_STATUS
    /// gets MCIP, EIPV and, unless the error corrupted the context (PCC), RIPV. A machine check
    /// with CR4.MCE clear, or while MCIP is still set from the last one, shuts the processor down
    /// (APM vol. 2 sec. 9.3.4), which the run loop reports as a triple fault. In an SVM guest the
    /// #MC intercept and HOST_MCE_OVERRIDE come first (`svm_machine_check_exit`), and a shutdown
    /// is the guest's (`svm_shutdown_exit`). `Some` ends the run.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn machine_check(&mut self, error: MachineCheck) -> Option<ExitReason> {
        if !self.mca.log(error.bank, error.status, error.address) {
            return None;
        }
        let shutdown =
            |cpu: &mut Cpu| (!cpu.svm_shutdown_exit()).then_some(ExitReason::TripleFault);
        if self.mca.global_status & MCG_STATUS_MCIP != 0 {
            return shutdown(self);
        }
        let restartable = if error.status & STATUS_CONTEXT_CORRUPT == 0 {
            MCG_STATUS_RIPV
        } else {
            0
        };
        let signaled = MCG_STATUS_MCIP | MCG_STATUS_EIPV | restartable;
        if self.svm_machine_check_exit() {
            self.mca.global_status = signaled;
            return None;
        }
        if self.cr4 & CR4_MCE == 0 {
            return shutdown(self);
        }
        self.mca.global_status = signaled;
        self.deliver_event(VECTOR_MACHINE_CHECK, None, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::test_support::{handler_for, install_idt, install_tss};
    use crate::testkit::{HLT, Harness};

    const WRITABLE: WriteContext = WriteContext {
        status_writable: true,
    };
    const LOCKED: WriteContext = WriteContext {
        status_writable: false,
    };

    fn smca(bank: u32, register: usize) -> u32 {
        0xc000_2000 + 0x10 * bank + register as u32
    }

    #[test]
    fn capability_reports_24_banks_and_the_global_control_register() {
        assert_eq!(Mca::new().read(MSR_MCG_CAP), Some(0x118));
        assert!(!Mca::new().write(MSR_MCG_CAP, 0, WRITABLE));
    }

    #[test]
    fn legacy_registers_alias_the_first_four_smca_registers() {
        let mut mca = Mca::new();
        // bank 5 is legacy 0x414..=0x417
        for (legacy, register, reset) in [
            (0x414, CTL, 0),
            (0x416, ADDR, 0),
            (0x417, MISC0, MISC0_RESET),
        ] {
            assert!(mca.write(legacy, 0x1234 << 32, LOCKED));
            assert_eq!(
                mca.read(smca(5, register)),
                Some(reset | 0x1234 << 32),
                "{legacy:#x}"
            );
        }
        assert!(mca.write(smca(5, STATUS), 0x55, WRITABLE));
        assert_eq!(mca.read(0x415), Some(0x55));
    }

    #[test]
    fn status_accepts_nonzero_only_with_mc_status_write_enable() {
        let mut mca = Mca::new();
        for register in [STATUS, DESTAT] {
            assert!(mca.write(smca(0, register), 0, LOCKED));
            assert!(!mca.write(smca(0, register), 1, LOCKED));
            assert_eq!(mca.read(smca(0, register)), Some(0));
            assert!(mca.write(smca(0, register), STATUS_VALID | 7, WRITABLE));
            assert_eq!(mca.read(smca(0, register)), Some(STATUS_VALID | 7));
        }
        assert!(!mca.write(0x401, 1, LOCKED));
    }

    #[test]
    fn config_keeps_its_capabilities_and_stores_the_enables() {
        let mut mca = Mca::new();
        let config = smca(3, CONFIG);
        assert_eq!(mca.read(config), Some(0b10_0101));
        // what Linux writes: McaXEn and DeferredIntType = APIC on top of what it read
        let mcax_enable_and_apic_interrupt = (1 << 32) | (1 << 37);
        assert!(mca.write(config, 0b10_0101 | mcax_enable_and_apic_interrupt, LOCKED));
        assert_eq!(
            mca.read(config),
            Some(0b10_0101 | mcax_enable_and_apic_interrupt)
        );
        assert!(mca.write(config, 0, LOCKED));
        assert_eq!(mca.read(config), Some(0b10_0101));
    }

    #[test]
    fn ipid_names_each_bank_as_linux_expects_and_is_read_only() {
        let mca = Mca::new();
        // (HardwareID, McaType) pairs from Linux's smca_hwid_mcatypes: LS_V2, IF, L2, DE, EX, FP
        let core = [(0, 0x10), (1, 1), (2, 2), (3, 3), (5, 5), (6, 6)];
        for (bank, mca_type) in core {
            let ipid = mca.read(smca(bank, IPID)).unwrap();
            assert_eq!(
                (ipid >> 32 & 0xfff, ipid >> 48),
                (0xb0, mca_type),
                "bank {bank}"
            );
        }
        for (slice, bank) in (7..15).enumerate() {
            assert_eq!(
                mca.read(smca(bank, IPID)),
                Some(7 << 48 | 0xb0 << 32 | slice as u64)
            );
        }
        // UMC_V2 and CS_V2
        assert_eq!(mca.read(smca(17, IPID)).unwrap() >> 32, 1 << 16 | 0x96);
        assert_eq!(mca.read(smca(20, IPID)).unwrap() >> 32, 2 << 16 | 0x2e);
        assert!(!Mca::new().write(smca(0, IPID), 0, WRITABLE));
    }

    #[test]
    fn misc0_offers_a_threshold_counter_and_keeps_its_read_only_bits() {
        let mut mca = Mca::new();
        let misc = smca(2, MISC0);
        assert_eq!(mca.read(misc), Some(MISC0_RESET));
        // ErrCnt, IntType = APIC and CntEn as Linux's threshold_restart_bank writes them, with a
        // try at clearing Valid and setting BlkPtr
        let programmed = (0xff5 << 32) | (1 << 49) | (1 << 51);
        assert!(mca.write(misc, programmed | 0xff00_0000, LOCKED));
        assert_eq!(mca.read(misc), Some(MISC0_RESET | programmed));
    }

    #[test]
    fn the_reserved_bank_reads_zero_and_ignores_writes() {
        let mut mca = Mca::new();
        for register in [CTL, STATUS, MISC0, CONFIG, IPID, SYND] {
            assert!(mca.write(smca(4, register), u64::MAX, WRITABLE));
            assert_eq!(mca.read(smca(4, register)), Some(0));
        }
    }

    #[test]
    fn the_other_bank_registers_store_what_is_written() {
        let mut mca = Mca::new();
        for register in [ADDR, SYND, DEADDR, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf] {
            assert!(mca.write(smca(23, register), 0xdead_beef, LOCKED));
            assert_eq!(mca.read(smca(23, register)), Some(0xdead_beef));
        }
    }

    #[test]
    fn registers_stop_after_the_last_bank_and_offset_7_does_not_exist() {
        assert!(Mca::owns(0x400 + 24 * 4 - 1));
        assert!(!Mca::owns(0x400 + 24 * 4));
        assert!(Mca::owns(smca(23, 0xf)));
        assert!(!Mca::owns(smca(24, 0)));
        assert!(!Mca::owns(smca(0, NO_REGISTER)));
        assert!(Mca::owns(0xc001_0400 + 23));
        assert!(!Mca::owns(0xc001_0400 + 24));
    }

    #[test]
    fn interrupt_config_has_turins_lvt_offsets() {
        let mut mca = Mca::new();
        assert_eq!(mca.read(0xc000_0410), Some(0x1020));
        assert!(mca.write(0xc000_0410, 0x3040, LOCKED));
        assert_eq!(mca.read(0xc000_0410), Some(0x3040));
        assert!(!mca.write(0xc000_0410, 1 << 32, LOCKED));
    }

    #[test]
    fn global_status_keeps_only_its_three_bits() {
        let mut mca = Mca::new();
        assert!(mca.write(MSR_MCG_STATUS, u64::MAX, LOCKED));
        assert_eq!(mca.read(MSR_MCG_STATUS), Some(7));
    }

    const UNCORRECTED: u64 = STATUS_UNCORRECTED | STATUS_ENABLED | STATUS_ADDRESS_VALID;

    fn armed() -> Harness {
        let mut h = Harness::new(&[HLT]);
        install_idt(&mut h);
        install_tss(&mut h, [0x70_0000; 3], [0x61_0000; 7]);
        h.cpu.cr4 |= CR4_MCE;
        h.cpu.mca.write(MSR_MCG_CTL, u64::MAX, LOCKED);
        h.cpu.mca.write(smca(2, CTL), u64::MAX, LOCKED);
        h
    }

    fn error(status: u64) -> MachineCheck {
        MachineCheck {
            bank: 2,
            status,
            address: 0x1234_5000,
        }
    }

    #[test]
    fn an_uncorrected_error_raises_machine_check_with_mcip() {
        let mut h = armed();
        assert!(h.cpu.machine_check(error(UNCORRECTED)).is_none());
        assert_eq!(h.cpu.rip, handler_for(VECTOR_MACHINE_CHECK));
        assert_eq!(
            h.cpu.mca.read(MSR_MCG_STATUS).unwrap(),
            MCG_STATUS_MCIP | MCG_STATUS_EIPV | MCG_STATUS_RIPV
        );
        assert_eq!(
            h.cpu.mca.read(smca(2, STATUS)),
            Some(STATUS_VALID | UNCORRECTED)
        );
        assert_eq!(h.cpu.mca.read(0x40a), Some(0x1234_5000));
    }

    #[test]
    fn a_corrupt_context_clears_ripv() {
        let mut h = armed();
        h.cpu
            .machine_check(error(UNCORRECTED | STATUS_CONTEXT_CORRUPT));
        assert_eq!(
            h.cpu.mca.read(MSR_MCG_STATUS).unwrap(),
            MCG_STATUS_MCIP | MCG_STATUS_EIPV
        );
    }

    #[test]
    fn machine_check_without_cr4_mce_or_during_another_shuts_down() {
        let mut h = armed();
        h.cpu.cr4 &= !CR4_MCE;
        assert_eq!(
            h.cpu.machine_check(error(UNCORRECTED)),
            Some(ExitReason::TripleFault)
        );
        let mut h = armed();
        assert!(h.cpu.machine_check(error(UNCORRECTED)).is_none());
        assert_eq!(
            h.cpu.machine_check(error(UNCORRECTED)),
            Some(ExitReason::TripleFault)
        );
        // the handler clears MCIP, after which the next one is delivered again
        let mut h = armed();
        h.cpu.machine_check(error(UNCORRECTED));
        h.cpu.mca.write(MSR_MCG_STATUS, 0, LOCKED);
        h.cpu.mca.write(smca(2, STATUS), 0, LOCKED);
        assert!(h.cpu.machine_check(error(UNCORRECTED)).is_none());
    }

    #[test]
    fn corrected_and_disabled_errors_are_logged_but_not_signaled() {
        let mut h = armed();
        let rip = h.cpu.rip;
        assert!(h.cpu.machine_check(error(STATUS_ENABLED)).is_none());
        assert_eq!(h.cpu.rip, rip);
        assert_eq!(
            h.cpu.mca.read(smca(2, STATUS)),
            Some(STATUS_VALID | STATUS_ENABLED)
        );
        let mut h = armed();
        h.cpu.mca.write(MSR_MCG_CTL, !(1 << 2), LOCKED);
        assert!(h.cpu.machine_check(error(UNCORRECTED)).is_none());
        assert_eq!(h.cpu.rip, rip);
        assert_eq!(h.cpu.mca.read(MSR_MCG_STATUS).unwrap(), 0);
    }

    #[test]
    fn a_second_error_sets_overflow_and_an_uncorrected_one_wins() {
        let mut h = armed();
        h.cpu.mca.write(MSR_MCG_CTL, 0, LOCKED);
        h.cpu.machine_check(error(STATUS_ENABLED));
        h.cpu.machine_check(error(UNCORRECTED));
        assert_eq!(
            h.cpu.mca.read(smca(2, STATUS)),
            Some(STATUS_VALID | STATUS_OVERFLOW | UNCORRECTED)
        );
        h.cpu.machine_check(error(STATUS_ENABLED));
        assert_eq!(
            h.cpu.mca.read(smca(2, STATUS)),
            Some(STATUS_VALID | STATUS_OVERFLOW | UNCORRECTED)
        );
    }

    #[test]
    fn a_deferred_error_is_also_logged_in_destat() {
        let mut h = armed();
        let deferred = STATUS_DEFERRED | STATUS_ADDRESS_VALID;
        assert!(h.cpu.machine_check(error(deferred)).is_none());
        assert_eq!(
            h.cpu.mca.read(smca(2, DESTAT)),
            Some(STATUS_VALID | deferred)
        );
        assert_eq!(h.cpu.mca.read(smca(2, DEADDR)), Some(0x1234_5000));
    }

    #[test]
    fn errors_in_missing_banks_are_dropped() {
        let mut h = armed();
        for bank in [4, BANKS] {
            let missing = MachineCheck {
                bank,
                ..error(UNCORRECTED)
            };
            assert!(h.cpu.machine_check(missing).is_none());
        }
        assert_eq!(h.cpu.mca.read(MSR_MCG_STATUS).unwrap(), 0);
    }
}
