//! Zen 5 model-specific registers that describe the processor more than they control it: the
//! effective-frequency counters (MPERF/APERF, which `rdpru` also reads), the hardware P-state
//! registers, RAPL energy counters, the protected processor inventory number, the CPUID
//! feature-override masks, and a handful of configuration MSRs that Linux, Xen and KVM read or
//! read-modify-write on family 1Ah and that only need to hold a value (`STORED_ONLY`).
//!
//! The machine has one clock: the TSC counts guest nanoseconds (`Cpu::tsc`), so P0, the frequency
//! MPERF counts at and the TSC rate are all 1 GHz. Every counter here derives from that clock, so a
//! run with a fixed `--time` and no wall clock reads the same values every time.

use super::exception::Exception;
use super::{Cpu, ExecResult};

/// Maximum and actual performance clocks (AMD names them MPERF and APERF, Linux "aperfmperf"),
/// advertised by leaf 6 ECX bit 0. Writable: Linux zeroes neither, but the hardware allows it.
pub(super) const MSR_MPERF: u32 = 0xe7;
pub(super) const MSR_APERF: u32 = 0xe8;
/// The same counters, read-only, at the addresses leaf 0x8000_0007 EDX bit 10 (EffFreqRO)
/// promises. A write is #GP.
const MSR_MPERF_READ_ONLY: u32 = 0xc000_00e7;
const MSR_APERF_READ_ONLY: u32 = 0xc000_00e8;

/// Hardware P-states (leaf 0x8000_0007 EDX bit 7, HwPstate): the current-limit and status
/// registers are read-only, the control register takes a P-state number, and eight definitions
/// follow. Nothing changes speed: every P-state runs at the one guest clock.
const MSR_PSTATE_CURRENT_LIMIT: u32 = 0xc001_0061;
const MSR_PSTATE_CONTROL: u32 = 0xc001_0062;
const MSR_PSTATE_STATUS: u32 = 0xc001_0063;
const MSR_PSTATE_DEFINITION_FIRST: u32 = 0xc001_0064;
const MSR_PSTATE_DEFINITION_LAST: u32 = 0xc001_006b;
/// PStateCurLim bits 6:4: the highest (slowest) P-state number software may request.
const PSTATE_MAX_VALUE_SHIFT: u32 = 4;
/// The P-state field of PStateCtl and PStateStat (bits 2:0).
const PSTATE_NUMBER_MASK: u64 = 0b111;
/// PStateDef bit 63: the definition is valid.
const PSTATE_ENABLED: u64 = 1 << 63;
/// PStateDef CpuFid (bits 11:0). Family 1Ah gives the core clock directly as CpuFid times 5 MHz
/// (earlier families divided a 25 MHz multiple by a DFS divisor), per the family 1Ah PPR.
const PSTATE_FID_MHZ: u64 = 5;
/// P0 is the TSC rate (1 GHz, see the module comment), as on every Zen; P1 and P2 are slower
/// steps that exist only on paper.
/// `cppc.rs` reports P0 as nominal and the last as lowest nonlinear performance.
const PSTATE_MHZ: [u64; 3] = [1000, 800, 600];
pub(super) const FASTEST_PSTATE_MHZ: u64 = PSTATE_MHZ[0];
pub(super) const SLOWEST_PSTATE_MHZ: u64 = PSTATE_MHZ[PSTATE_MHZ.len() - 1];

/// MMIO_CONF_BASE, the ECAM window: base address (47:20), bus range as log2 of the bus count
/// (5:2), enable (0). It must agree with the MCFG table, which describes one bus at `ECAM_BASE`.
const MSR_MMIO_CONFIGURATION_BASE: u32 = 0xc001_0058;
const MMIO_CONFIGURATION_ENABLE: u64 = 1;
const MMIO_CONFIGURATION_VALUE: u64 = crate::dev::pci::ECAM_BASE | MMIO_CONFIGURATION_ENABLE;

/// RAPL (leaf 0x8000_0007 EDX bit 14): the unit register and the core and package energy
/// counters, all read-only. Units: power 1/2^3 W (3:0), energy 1/2^16 J (12:8), time 1/2^10 s
/// (19:16), the values every Zen reports.
const MSR_RAPL_POWER_UNIT: u32 = 0xc001_0299;
const MSR_CORE_ENERGY_STATUS: u32 = 0xc001_029a;
const MSR_PACKAGE_ENERGY_STATUS: u32 = 0xc001_029b;
const RAPL_POWER_UNIT_VALUE: u64 = 0x000a_1003;
const RAPL_ENERGY_UNITS_PER_JOULE: u128 = 1 << 16;
/// The energy counters are 32 bits and wrap, as Linux's RAPL PMU expects.
const RAPL_ENERGY_MASK: u64 = 0xffff_ffff;
/// A constant draw: each core uses `CORE_WATTS` whenever time passes, and the package adds
/// `UNCORE_WATTS` for the fabric and I/O die. Plausible for a Zen 5 server core, not a model.
const CORE_WATTS: u128 = 5;
const UNCORE_WATTS: u128 = 30;
const NS_PER_SECOND: u128 = 1_000_000_000;

/// Protected processor inventory number (leaf 0x8000_0008 EBX bit 23). PPIN_CTL bit 0 locks the
/// control register and bit 1 enables reading PPIN, which is #GP while disabled.
const MSR_PPIN_CONTROL: u32 = 0xc001_02f0;
const MSR_PPIN: u32 = 0xc001_02f1;
const PPIN_CONTROL_LOCK_OUT: u64 = 1 << 0;
const PPIN_CONTROL_ENABLE: u64 = 1 << 1;
/// One package, so one number for every vCPU. Fixed so runs are reproducible.
const PPIN_VALUE: u64 = 0x0b00_2155_0e5c_0001;

/// SEV_STATUS: SEV is not implemented (user decision), so it reads zero and cannot be written.
/// Linux reads it only when leaf 0x8000_001F says SEV exists, which it does not.
const MSR_SEV_STATUS: u32 = 0xc001_0131;

/// The CPUID feature-override MSRs Linux and Xen write to hide features (Linux clears RDSEED in
/// Fn7 on Zen 5 parts with old microcode and RDRAND in Fn1 when it is broken; Xen levels a pool
/// of hosts to a common subset with all three). Each covers two registers of one leaf, high half
/// then low half: Fn0000_0007 EAX:EBX (subleaf 0), Fn0000_0001 ECX:EDX, Fn8000_0001 ECX:EDX.
/// A read returns what CPUID reports; a write is a mask on what it reports: a bit can be taken
/// away and given back, never added beyond what the CPU implements.
const MSR_CPUID_STRUCTURED: u32 = 0xc001_1002;
const MSR_CPUID_FEATURES: u32 = 0xc001_1004;
const MSR_CPUID_EXTENDED_FEATURES: u32 = 0xc001_1005;

/// MSRs that hold whatever was last written and affect nothing. Seeded into `Cpu::msr_misc`.
pub(super) const STORED_ONLY: [(u32, u64); 8] = [
    // OS visible workarounds (leaf 0x8000_0001 ECX bit 9): ID length 0 says no erratum is
    // described, so the status register has no meaningful bits. KVM and Linux read both.
    (0xc001_0140, 0),
    (0xc001_0141, 0),
    // SMM_BASE (its reset value, 0x30000), SMM_ADDR and SMM_MASK: there is no SMM, but Xen and
    // KVM read the base and firmware writes the TSEG pair.
    (0xc001_0111, 0x3_0000),
    (0xc001_0112, 0),
    (0xc001_0113, 0),
    // TW_CFG and FP_CFG: table-walker and floating-point configuration, which Linux and Xen
    // set erratum bits in.
    (0xc001_1023, 0),
    (0xc001_1028, 0),
    // PREFETCH_CONTROL (leaf 0x8000_0021 EAX bit 13, PrefetchCtlMsr): hardware prefetcher
    // disables. There are no prefetchers.
    (0xc000_0108, 0),
];

/// Which CPUID override MSR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CpuidMask {
    Structured,
    Features,
    ExtendedFeatures,
}

/// Per-vCPU state behind the MSRs above.
#[derive(Clone, Debug)]
pub(crate) struct ModelMsrs {
    /// Added to guest time to make the TSC: a write of IA32_TIME_STAMP_COUNTER (0x10) sets it.
    pub tsc_adjust: u64,
    mperf_offset: u64,
    aperf_offset: u64,
    pstate_command: u64,
    ppin_control: u64,
    /// The override masks, `CpuidMask` order; all ones until software clears a bit.
    cpuid_masks: [u64; 3],
}

impl Default for ModelMsrs {
    fn default() -> Self {
        ModelMsrs {
            tsc_adjust: 0,
            mperf_offset: 0,
            aperf_offset: 0,
            pstate_command: 0,
            ppin_control: 0,
            cpuid_masks: [u64::MAX; 3],
        }
    }
}

impl ModelMsrs {
    /// The (high, low) halves of the mask that applies to the two registers `which` covers.
    pub(super) fn cpuid_mask(&self, which: CpuidMask) -> (u32, u32) {
        let mask = self.cpuid_masks[which as usize];
        ((mask >> 32) as u32, mask as u32)
    }
}

pub(super) fn owns(msr: u32) -> bool {
    matches!(
        msr,
        MSR_MPERF
            | MSR_APERF
            | MSR_MPERF_READ_ONLY
            | MSR_APERF_READ_ONLY
            | MSR_PSTATE_CURRENT_LIMIT
            | MSR_PSTATE_CONTROL
            | MSR_PSTATE_STATUS
            | MSR_MMIO_CONFIGURATION_BASE
            | MSR_RAPL_POWER_UNIT
            | MSR_CORE_ENERGY_STATUS
            | MSR_PACKAGE_ENERGY_STATUS
            | MSR_PPIN_CONTROL
            | MSR_PPIN
            | MSR_SEV_STATUS
            | MSR_CPUID_STRUCTURED
            | MSR_CPUID_FEATURES
            | MSR_CPUID_EXTENDED_FEATURES
    ) || (MSR_PSTATE_DEFINITION_FIRST..=MSR_PSTATE_DEFINITION_LAST).contains(&msr)
}

fn cpuid_mask_of(msr: u32) -> Option<CpuidMask> {
    match msr {
        MSR_CPUID_STRUCTURED => Some(CpuidMask::Structured),
        MSR_CPUID_FEATURES => Some(CpuidMask::Features),
        MSR_CPUID_EXTENDED_FEATURES => Some(CpuidMask::ExtendedFeatures),
        _ => None,
    }
}

/// PStateDef for P-state `index`: enabled with its clock, or all zero past the last one.
fn pstate_definition(index: usize) -> u64 {
    match PSTATE_MHZ.get(index) {
        Some(mhz) => PSTATE_ENABLED | (mhz / PSTATE_FID_MHZ),
        None => 0,
    }
}

impl Cpu {
    /// MPERF: counts at P0 (the TSC rate) since reset, plus whatever a write moved it by.
    pub(super) fn mperf(&self) -> u64 {
        self.now_ns().wrapping_add(self.model_msrs.mperf_offset)
    }

    /// APERF: counts at the actual clock, which is always P0 here, so it runs with MPERF and
    /// Linux computes an effective frequency equal to the TSC rate.
    pub(super) fn aperf(&self) -> u64 {
        self.now_ns().wrapping_add(self.model_msrs.aperf_offset)
    }

    /// A write of the TSC: later reads continue from `value`. Under SVM the write changes the one
    /// real counter, as on hardware, so that the guest reads `value` through its ratio and offset.
    pub(super) fn set_tsc(&mut self, value: u64) {
        let host_tsc = self.svm_host_tsc_for(value);
        self.model_msrs.tsc_adjust = host_tsc.wrapping_sub(self.now_ns());
    }

    /// Energy in RAPL units since reset for a constant draw of `watts`.
    fn rapl_energy(&self, watts: u128) -> u64 {
        let joules_scaled = self.now_ns() as u128 * watts * RAPL_ENERGY_UNITS_PER_JOULE;
        (joules_scaled / NS_PER_SECOND) as u64 & RAPL_ENERGY_MASK
    }

    pub(super) fn model_msr_read(&mut self, msr: u32) -> ExecResult<u64> {
        if let Some(which) = cpuid_mask_of(msr) {
            let (high, low) = self.cpuid_overridable(which);
            let reported = (high as u64) << 32 | low as u64;
            return Ok(reported & self.model_msrs.cpuid_masks[which as usize]);
        }
        Ok(match msr {
            MSR_MPERF | MSR_MPERF_READ_ONLY => self.mperf(),
            MSR_APERF | MSR_APERF_READ_ONLY => self.aperf(),
            MSR_PSTATE_CURRENT_LIMIT => (PSTATE_MHZ.len() as u64 - 1) << PSTATE_MAX_VALUE_SHIFT,
            MSR_PSTATE_CONTROL => self.model_msrs.pstate_command,
            MSR_PSTATE_STATUS => self
                .model_msrs
                .pstate_command
                .min(PSTATE_MHZ.len() as u64 - 1),
            MSR_PSTATE_DEFINITION_FIRST..=MSR_PSTATE_DEFINITION_LAST => {
                pstate_definition((msr - MSR_PSTATE_DEFINITION_FIRST) as usize)
            }
            MSR_MMIO_CONFIGURATION_BASE => MMIO_CONFIGURATION_VALUE,
            MSR_RAPL_POWER_UNIT => RAPL_POWER_UNIT_VALUE,
            MSR_CORE_ENERGY_STATUS => self.rapl_energy(CORE_WATTS),
            MSR_PACKAGE_ENERGY_STATUS => self
                .rapl_energy(CORE_WATTS * self.topology.logical_processors as u128 + UNCORE_WATTS),
            MSR_PPIN_CONTROL => self.model_msrs.ppin_control,
            MSR_PPIN => {
                if self.model_msrs.ppin_control & PPIN_CONTROL_ENABLE == 0 {
                    return Err(Exception::general_protection(0));
                }
                PPIN_VALUE
            }
            MSR_SEV_STATUS => 0,
            _ => return Err(Exception::general_protection(0)),
        })
    }

    pub(super) fn model_msr_write(&mut self, msr: u32, value: u64) -> ExecResult<()> {
        if let Some(which) = cpuid_mask_of(msr) {
            self.model_msrs.cpuid_masks[which as usize] = value;
            return Ok(());
        }
        let now = self.now_ns();
        match msr {
            MSR_MPERF => self.model_msrs.mperf_offset = value.wrapping_sub(now),
            MSR_APERF => self.model_msrs.aperf_offset = value.wrapping_sub(now),
            MSR_PSTATE_CONTROL if value & !PSTATE_NUMBER_MASK == 0 => {
                self.model_msrs.pstate_command = value;
            }
            // The firmware's definitions stand: a write that changes nothing is accepted.
            MSR_PSTATE_DEFINITION_FIRST..=MSR_PSTATE_DEFINITION_LAST
                if value == pstate_definition((msr - MSR_PSTATE_DEFINITION_FIRST) as usize) => {}
            // The ECAM window is fixed by the platform; rewriting the same value is all firmware
            // ever does.
            MSR_MMIO_CONFIGURATION_BASE if value == MMIO_CONFIGURATION_VALUE => {}
            MSR_PPIN_CONTROL
                if self.model_msrs.ppin_control & PPIN_CONTROL_LOCK_OUT == 0
                    && value & !(PPIN_CONTROL_LOCK_OUT | PPIN_CONTROL_ENABLE) == 0 =>
            {
                self.model_msrs.ppin_control = value;
            }
            _ => return Err(Exception::general_protection(0)),
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "model_msr_tests.rs"]
mod tests;
