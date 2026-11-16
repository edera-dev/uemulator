//! Collaborative Processor Performance Control through MSRs (CPUID 0x8000_0008 EBX bit 27, the
//! "MSR-based CPPC" of Zen 2 and later; register layout as Linux's `msr-index.h` names it). The
//! ACPI `_CPC` package of every processor device points at these registers (`acpi/dsdt.rs`),
//! which is what `amd_pstate` needs to load in active or passive mode.
//!
//! The emulated CPU has one speed, so the capabilities describe exactly that: nominal and highest
//! performance are the same level (no boost, no preferred cores), nominal frequency is the TSC
//! rate, and requests are stored without changing anything.
//!
//! The hardware P-state registers, which amd_pstate also writes (`amd_perf_ctl_reset`, an
//! unchecked wrmsr of PERF_CTL that warns on #GP, on every CPU it takes over), are
//! `model_msr.rs`; the lowest nonlinear level here is its slowest P-state.
use crate::snapshot::Snapshot;

/// Read-only: lowest (7:0), lowest nonlinear (15:8), nominal (23:16) and highest (31:24)
/// performance.
pub(crate) const MSR_CPPC_CAPABILITY_1: u32 = 0xc001_02b0;
/// Bit 0 turns CPPC on; once set it stays set until reset.
pub(crate) const MSR_CPPC_ENABLE: u32 = 0xc001_02b1;
/// Read-only and zero: Linux never reads it, and nothing in it is described for software.
pub(crate) const MSR_CPPC_CAPABILITY_2: u32 = 0xc001_02b2;
/// Maximum (7:0), minimum (15:8) and desired (23:16) performance and the energy-performance
/// preference (31:24) the OS asks for.
pub(crate) const MSR_CPPC_REQUEST: u32 = 0xc001_02b3;
/// Read-only and zero: the performance never had to drop below what was requested.
pub(crate) const MSR_CPPC_STATUS: u32 = 0xc001_02b4;

pub(crate) const ENABLE_BIT: u64 = 1;
/// The request fields; bits 63:32 are reserved and #GP when set.
const REQUEST_WRITABLE: u64 = 0xffff_ffff;

/// Bit positions of the fields in CAPABILITY_1 and REQUEST, which `_CPC` refers to.
pub(crate) const LOWEST_SHIFT: u8 = 0;
pub(crate) const LOWEST_NONLINEAR_SHIFT: u8 = 8;
pub(crate) const NOMINAL_SHIFT: u8 = 16;
pub(crate) const HIGHEST_SHIFT: u8 = 24;
pub(crate) const REQUEST_MAXIMUM_SHIFT: u8 = 0;
pub(crate) const REQUEST_MINIMUM_SHIFT: u8 = 8;
pub(crate) const REQUEST_DESIRED_SHIFT: u8 = 16;
pub(crate) const REQUEST_ENERGY_PREFERENCE_SHIFT: u8 = 24;
pub(crate) const PERFORMANCE_FIELD_BITS: u8 = 8;

/// Abstract performance levels, 10 MHz each. Nominal is the TSC rate, 1 GHz (`Cpu::tsc` counts
/// nanoseconds; the UEFI handoff's `TSC_HZ` says the same). Linux's `amd_pstate_init_freq` fails
/// the driver unless lowest < lowest nonlinear <= nominal <= highest, with the frequencies
/// derived from them positive. Lowest nonlinear is the slowest P-state, as on hardware, where the
/// levels below it are reached by throttling rather than by a lower clock.
const MHZ_PER_PERFORMANCE_LEVEL: u32 = 10;
const LOWEST_PERFORMANCE: u32 = 20;
const LOWEST_NONLINEAR_PERFORMANCE: u32 =
    super::model_msr::SLOWEST_PSTATE_MHZ as u32 / MHZ_PER_PERFORMANCE_LEVEL;
const NOMINAL_PERFORMANCE: u32 =
    super::model_msr::FASTEST_PSTATE_MHZ as u32 / MHZ_PER_PERFORMANCE_LEVEL;
const HIGHEST_PERFORMANCE: u32 = NOMINAL_PERFORMANCE;
/// The integers `_CPC` gives next to the registers (entries 21 and 22 of revision 3), in MHz.
pub(crate) const LOWEST_FREQUENCY_MHZ: u32 = LOWEST_PERFORMANCE * MHZ_PER_PERFORMANCE_LEVEL;
pub(crate) const NOMINAL_FREQUENCY_MHZ: u32 = NOMINAL_PERFORMANCE * MHZ_PER_PERFORMANCE_LEVEL;

const CAPABILITY_1: u64 = (LOWEST_PERFORMANCE << LOWEST_SHIFT
    | LOWEST_NONLINEAR_PERFORMANCE << LOWEST_NONLINEAR_SHIFT
    | NOMINAL_PERFORMANCE << NOMINAL_SHIFT
    | HIGHEST_PERFORMANCE << HIGHEST_SHIFT) as u64;
/// What firmware leaves in REQUEST: the full range allowed, no desired level (autonomous), and
/// the balanced energy preference.
const ENERGY_PREFERENCE_BALANCED: u32 = 0x80;
const REQUEST_RESET: u64 = (HIGHEST_PERFORMANCE << REQUEST_MAXIMUM_SHIFT
    | LOWEST_PERFORMANCE << REQUEST_MINIMUM_SHIFT
    | ENERGY_PREFERENCE_BALANCED << REQUEST_ENERGY_PREFERENCE_SHIFT)
    as u64;

/// The writable registers, per logical processor.
#[derive(Clone, Snapshot)]
pub(crate) struct Cppc {
    enable: u64,
    request: u64,
}

impl Cppc {
    pub fn new() -> Self {
        Cppc {
            enable: 0,
            request: REQUEST_RESET,
        }
    }

    pub fn owns(msr: u32) -> bool {
        (MSR_CPPC_CAPABILITY_1..=MSR_CPPC_STATUS).contains(&msr)
    }

    pub fn read(&self, msr: u32) -> Option<u64> {
        match msr {
            MSR_CPPC_CAPABILITY_1 => Some(CAPABILITY_1),
            MSR_CPPC_ENABLE => Some(self.enable),
            MSR_CPPC_CAPABILITY_2 | MSR_CPPC_STATUS => Some(0),
            MSR_CPPC_REQUEST => Some(self.request),
            _ => None,
        }
    }

    /// `false` is #GP: the capability and status registers are read-only, and reserved bits of
    /// the others must be zero. Clearing the enable bit is ignored, as on hardware.
    pub fn write(&mut self, msr: u32, value: u64) -> bool {
        match msr {
            MSR_CPPC_ENABLE if value & !ENABLE_BIT == 0 => {
                self.enable |= value;
                true
            }
            MSR_CPPC_REQUEST if value & !REQUEST_WRITABLE == 0 => {
                self.request = value;
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(value: u64, shift: u8) -> u32 {
        (value >> shift) as u32 & ((1 << PERFORMANCE_FIELD_BITS) - 1)
    }

    #[test]
    fn capabilities_satisfy_the_ordering_amd_pstate_checks() {
        let capability = Cppc::new().read(MSR_CPPC_CAPABILITY_1).unwrap();
        let lowest = field(capability, LOWEST_SHIFT);
        let nonlinear = field(capability, LOWEST_NONLINEAR_SHIFT);
        let nominal = field(capability, NOMINAL_SHIFT);
        let highest = field(capability, HIGHEST_SHIFT);
        assert!(0 < lowest && lowest < nonlinear && nonlinear <= nominal && nominal <= highest);
        assert_eq!(capability >> 32, 0);
        // The _CPC frequencies scale with the same factor as the levels.
        assert_eq!(
            LOWEST_FREQUENCY_MHZ * nominal,
            NOMINAL_FREQUENCY_MHZ * lowest
        );
        assert_eq!(NOMINAL_FREQUENCY_MHZ, 1000, "the TSC rate");
    }

    #[test]
    fn capability_and_status_registers_are_read_only() {
        let mut cppc = Cppc::new();
        for msr in [
            MSR_CPPC_CAPABILITY_1,
            MSR_CPPC_CAPABILITY_2,
            MSR_CPPC_STATUS,
        ] {
            let before = cppc.read(msr);
            assert!(!cppc.write(msr, 0), "{msr:#x}");
            assert_eq!(cppc.read(msr), before);
        }
        assert_eq!(cppc.read(MSR_CPPC_STATUS), Some(0));
    }

    #[test]
    fn enable_is_sticky_and_has_one_bit() {
        let mut cppc = Cppc::new();
        assert_eq!(cppc.read(MSR_CPPC_ENABLE), Some(0));
        assert!(!cppc.write(MSR_CPPC_ENABLE, 2));
        assert!(cppc.write(MSR_CPPC_ENABLE, 1));
        assert!(cppc.write(MSR_CPPC_ENABLE, 0));
        assert_eq!(cppc.read(MSR_CPPC_ENABLE), Some(1));
    }

    #[test]
    fn request_stores_its_low_dword_and_rejects_reserved_bits() {
        let mut cppc = Cppc::new();
        let initial = cppc.read(MSR_CPPC_REQUEST).unwrap();
        assert_eq!(field(initial, REQUEST_MAXIMUM_SHIFT), HIGHEST_PERFORMANCE);
        assert_eq!(field(initial, REQUEST_MINIMUM_SHIFT), LOWEST_PERFORMANCE);
        assert!(cppc.write(MSR_CPPC_REQUEST, 0xff64_1464));
        assert_eq!(cppc.read(MSR_CPPC_REQUEST), Some(0xff64_1464));
        assert!(!cppc.write(MSR_CPPC_REQUEST, 1 << 32));
        assert_eq!(cppc.read(MSR_CPPC_REQUEST), Some(0xff64_1464));
    }

    #[test]
    fn the_range_is_exactly_the_cppc_registers() {
        assert!(!Cppc::owns(MSR_CPPC_CAPABILITY_1 - 1));
        assert!((MSR_CPPC_CAPABILITY_1..=MSR_CPPC_STATUS).all(Cppc::owns));
        assert!(!Cppc::owns(MSR_CPPC_STATUS + 1));
        // the P-state registers are `model_msr.rs`'s
        assert!(!(0xc001_0061..=0xc001_006b).any(Cppc::owns));
    }

    /// The levels and the P-state definitions describe the same clock: nominal is P0 and lowest
    /// nonlinear the slowest P-state.
    #[test]
    fn nominal_and_lowest_nonlinear_are_the_fastest_and_slowest_pstates() {
        let capability = Cppc::new().read(MSR_CPPC_CAPABILITY_1).unwrap();
        let mhz = |shift| field(capability, shift) * MHZ_PER_PERFORMANCE_LEVEL;
        assert_eq!(
            mhz(NOMINAL_SHIFT) as u64,
            super::super::model_msr::FASTEST_PSTATE_MHZ
        );
        assert_eq!(
            mhz(LOWEST_NONLINEAR_SHIFT) as u64,
            super::super::model_msr::SLOWEST_PSTATE_MHZ
        );
    }
}
