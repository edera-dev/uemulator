//! What a machine's CPU looks like beyond the fixed Turin model in `cpuid.rs`: how its vCPUs pair
//! up into cores and which EPYC 9005 part the brand string names (`docs/plans/21-ryzen-7000-parity.md`
//! section F; signature, caches and features stay those of the one model).
//!
//! vCPU `n` is logical processor `n` with x2APIC id `n` whatever the shape: AMD numbers the threads
//! of a core consecutively (`core << smt_shift | thread`), so with two threads per core and an even
//! count the ids are still 0..count, and the local APICs, the MADT and the IPI routing never need
//! a translation. Breaking that (sparse ids, a socket field) means touching all of them.

use super::cpuid::CpuTopology;

/// Zen 5 has SMT2; Turin parts ship with it on.
const SMT_THREADS_PER_CORE: u32 = 2;

/// Brand strings of EPYC 9005 parts by core count, smallest first: one per count, the
/// general-purpose part where several share it. The guest gets the smallest part with at least
/// as many cores as it sees, so it is never told about fewer cores than it has, as in a VM on
/// that part. The last three are Turin Dense (Zen 5c, family 1Ah model 11h) while the signature
/// stays the classic die's model 02h: no classic part has more than 128 cores, and a brand
/// string that undercounts the cores is the more visible lie (`lscpu`, `/proc/cpuinfo`). Linux
/// keys nothing on the brand.
const TURIN_PARTS: &[(u32, &str)] = &[
    (8, "AMD EPYC 9015 8-Core Processor"),
    (16, "AMD EPYC 9115 16-Core Processor"),
    (24, "AMD EPYC 9255 24-Core Processor"),
    (32, "AMD EPYC 9335 32-Core Processor"),
    (36, "AMD EPYC 9365 36-Core Processor"),
    (48, "AMD EPYC 9455 48-Core Processor"),
    (64, "AMD EPYC 9535 64-Core Processor"),
    (72, "AMD EPYC 9565 72-Core Processor"),
    (96, "AMD EPYC 9655 96-Core Processor"),
    (128, "AMD EPYC 9755 128-Core Processor"),
    (144, "AMD EPYC 9825 144-Core Processor"),
    (160, "AMD EPYC 9845 160-Core Processor"),
    (192, "AMD EPYC 9965 192-Core Processor"),
];

/// Leaves 0x8000_0002..=0x8000_0004 hold 48 bytes, the last of which must stay NUL.
pub(crate) const BRAND_BYTES: usize = 48;
const _: () = {
    let mut index = 0;
    while index < TURIN_PARTS.len() {
        assert!(TURIN_PARTS[index].1.len() < BRAND_BYTES);
        index += 1;
    }
};

/// The largest part's thread count, which is what `MachineConfig::cpus` allows.
pub(crate) const MAX_LOGICAL_PROCESSORS: u32 =
    TURIN_PARTS[TURIN_PARTS.len() - 1].0 * SMT_THREADS_PER_CORE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CpuProfile {
    pub logical_processors: u32,
    pub threads_per_core: u32,
    pub brand: &'static str,
}

impl CpuProfile {
    /// `cpus` vCPUs in one package. With `smt` an even count becomes cores of two threads; an odd
    /// count cannot be split that way and gets one thread per core, as does `smt` off.
    pub(crate) fn new(cpus: u32, smt: bool) -> Self {
        let logical_processors = cpus.max(1);
        let threads_per_core = if smt && logical_processors.is_multiple_of(SMT_THREADS_PER_CORE) {
            SMT_THREADS_PER_CORE
        } else {
            1
        };
        let cores = logical_processors / threads_per_core;
        let brand = TURIN_PARTS
            .iter()
            .find(|(part_cores, _)| *part_cores >= cores)
            .unwrap_or(&TURIN_PARTS[TURIN_PARTS.len() - 1])
            .1;
        CpuProfile {
            logical_processors,
            threads_per_core,
            brand,
        }
    }

    #[cfg(test)]
    pub(crate) fn cores(&self) -> u32 {
        self.logical_processors / self.threads_per_core
    }

    /// What CPUID tells vCPU `index`.
    pub(crate) fn topology(&self, index: u32) -> CpuTopology {
        CpuTopology {
            logical_processors: self.logical_processors,
            threads_per_core: self.threads_per_core,
            x2apic_id: index,
            brand: self.brand,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_counts_pair_into_cores_and_odd_ones_do_not() {
        for (cpus, smt, cores, threads) in [
            (1, true, 1, 1),
            (2, true, 1, 2),
            (3, true, 3, 1),
            (6, true, 3, 2),
            (6, false, 6, 1),
            (384, true, 192, 2),
            (383, true, 383, 1),
        ] {
            let profile = CpuProfile::new(cpus, smt);
            assert_eq!(
                (profile.cores(), profile.threads_per_core),
                (cores, threads),
                "{cpus} smt {smt}"
            );
            assert_eq!(profile.cores() * profile.threads_per_core, cpus);
        }
        assert_eq!(CpuProfile::new(0, true).logical_processors, 1);
    }

    #[test]
    fn the_brand_names_the_smallest_part_with_enough_cores() {
        let brand = |cpus, smt| CpuProfile::new(cpus, smt).brand;
        assert_eq!(brand(1, true), "AMD EPYC 9015 8-Core Processor");
        assert_eq!(brand(16, true), "AMD EPYC 9015 8-Core Processor");
        assert_eq!(brand(16, false), "AMD EPYC 9115 16-Core Processor");
        assert_eq!(brand(18, true), "AMD EPYC 9115 16-Core Processor");
        assert_eq!(brand(256, true), "AMD EPYC 9755 128-Core Processor");
        assert_eq!(brand(384, true), "AMD EPYC 9965 192-Core Processor");
        // more cores than any part: the largest one
        assert_eq!(brand(383, true), "AMD EPYC 9965 192-Core Processor");
    }

    #[test]
    fn every_brand_counts_the_cores_it_is_listed_for() {
        let mut previous = 0;
        for &(cores, brand) in TURIN_PARTS {
            assert!(cores > previous, "sorted, one part per count");
            assert!(brand.starts_with("AMD EPYC 9"), "{brand}");
            assert!(
                brand.ends_with(&format!(" {cores}-Core Processor")),
                "{brand}"
            );
            previous = cores;
        }
        assert_eq!(MAX_LOGICAL_PROCESSORS, 384);
    }
}
