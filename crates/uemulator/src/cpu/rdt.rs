//! Platform quality of service (AMD's name for Intel's Resource Director Technology): L3 cache
//! occupancy and memory-bandwidth monitoring, L3 cache allocation and L3 memory-bandwidth
//! enforcement (AMD "Platform Quality of Service Extensions" and the PQoS chapter of the family
//! 1Ah PPR). There is no cache and no memory bus to measure or partition, so this is the interface
//! with nothing behind it: every counter reads as valid and zero, and the allocation masks and
//! bandwidth limits are stored and read back. That is enough for Linux's resctrl to mount, accept
//! schemata writes and create monitoring groups.
//!
//! Real hardware shares the allocation MSRs between the threads of one L3; here each vCPU has
//! its own copy. Linux writes them on one CPU of each L3 domain and never reads them back, so the
//! difference does not show.

const MSR_QM_EVTSEL: u32 = 0xc8d;
const MSR_QM_CTR: u32 = 0xc8e;
const MSR_PQR_ASSOC: u32 = 0xc8f;
const MSR_L3_MASK_FIRST: u32 = 0xc90;
const MSR_L3_MASK_LAST: u32 = MSR_L3_MASK_FIRST + CLASSES_OF_SERVICE as u32 - 1;
const MSR_L3_BANDWIDTH_FIRST: u32 = 0xc000_0200;
const MSR_L3_BANDWIDTH_LAST: u32 = MSR_L3_BANDWIDTH_FIRST + CLASSES_OF_SERVICE as u32 - 1;

/// Resource monitoring ids, 0..=MAX_RMID, and classes of service, Turin's counts per L3.
const MAX_RMID: u64 = 255;
const CLASSES_OF_SERVICE: usize = 16;
/// L3 allocation masks are 16 bits, one per way of Zen 5's 16-way L3. AMD allows empty and
/// non-contiguous masks.
const L3_MASK_BITS: u32 = 16;
const L3_MASK_ALL: u64 = (1 << L3_MASK_BITS) - 1;
/// L3 bandwidth limits: an 11-bit limit in 1/8 GB/s plus U (bit 11), unlimited. Leaf
/// 0x8000_0020.1 EAX gives the width as 11, and Linux takes `1 << 11`, the U bit alone, as its
/// default "no limit".
const BANDWIDTH_WIDTH: u32 = 11;
const BANDWIDTH_UNLIMITED: u64 = 1 << BANDWIDTH_WIDTH;
const BANDWIDTH_WRITABLE: u64 = BANDWIDTH_UNLIMITED | (BANDWIDTH_UNLIMITED - 1);

/// PQR_ASSOC: the RMID the thread's accesses are counted under (9:0) and its class of service
/// (63:32).
const PQR_RMID_MASK: u64 = 0x3ff;
const PQR_CLOS_SHIFT: u32 = 32;
/// QM_EVTSEL: event id (7:0) and RMID (41:32) for the next QM_CTR read.
const EVTSEL_EVENT_MASK: u64 = 0xff;
const EVTSEL_RMID_SHIFT: u32 = 32;
const EVTSEL_RMID_MASK: u64 = 0x3ff << EVTSEL_RMID_SHIFT;
/// QM_CTR: Error (63) for an unsupported event or an RMID out of range, Unavailable (62) for no
/// data yet; the count is in 61:0.
const QM_CTR_ERROR: u64 = 1 << 63;

/// Monitoring events, also the bits of leaf 0xF.1 EDX (as `1 << (event - 1)`).
const EVENT_L3_OCCUPANCY: u64 = 1;
const EVENT_TOTAL_BANDWIDTH: u64 = 2;
const EVENT_LOCAL_BANDWIDTH: u64 = 3;

// CPUID.
pub(crate) const LEAF_MONITORING: u32 = 0xf;
pub(crate) const LEAF_ALLOCATION: u32 = 0x10;
pub(crate) const LEAF_EXTENDED_QOS: u32 = 0x8000_0020;
/// Leaf 7 EBX: PQM (12), leaves 0xF and the monitoring MSRs; PQE (15), leaf 0x10 and allocation.
pub(crate) const STRUCTURED_FEATURE_PQM: u32 = 1 << 12;
pub(crate) const STRUCTURED_FEATURE_PQE: u32 = 1 << 15;
/// Leaf 0x8000_0008 EBX bit 6, MBE: memory bandwidth enforcement, the leaf 0x8000_0020 limits.
/// Linux's X86_FEATURE_MBA on AMD.
pub(crate) const ADDRESS_SIZES_MBE: u32 = 1 << 6;
/// Leaf 0xF.0 EDX and leaf 0x10.0 EBX: L3 is the one monitored and the one allocated resource.
const RESOURCE_L3: u32 = 1 << 1;
/// Leaf 0xF.1 EBX: occupancy counts are in units of 64-byte lines.
const OCCUPANCY_SCALE_BYTES: u32 = 64;
const MONITORING_EVENTS: u32 = 1 << (EVENT_L3_OCCUPANCY - 1)
    | 1 << (EVENT_TOTAL_BANDWIDTH - 1)
    | 1 << (EVENT_LOCAL_BANDWIDTH - 1);
/// Leaf 0x8000_0020.0 EBX bit 1, L3MBE: L3 memory bandwidth enforcement (subleaf 1). Slow-memory
/// enforcement (2), the bandwidth event configuration (3) and assignable counters (5) are absent.
const EXTENDED_QOS_L3_BANDWIDTH: u32 = 1 << 1;

/// CPUID leaves 0xF, 0x10 and 0x8000_0020 for `subleaf`.
pub(crate) fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32, u32) {
    let highest_rmid = MAX_RMID as u32;
    let highest_class = CLASSES_OF_SERVICE as u32 - 1;
    match (leaf, subleaf) {
        // EBX: the highest RMID of any resource; EDX: the monitored resources
        (LEAF_MONITORING, 0) => (0, highest_rmid, 0, RESOURCE_L3),
        // EAX 7:0 is the counter width beyond 24 bits; zero makes Linux assume AMD's 44 bits
        (LEAF_MONITORING, 1) => (0, OCCUPANCY_SCALE_BYTES, highest_rmid, MONITORING_EVENTS),
        (LEAF_ALLOCATION, 0) => (0, RESOURCE_L3, 0, 0),
        // EAX: mask length minus one; EBX: bits shared with other agents; ECX: no CDP
        (LEAF_ALLOCATION, 1) => (L3_MASK_BITS - 1, 0, 0, highest_class),
        (LEAF_EXTENDED_QOS, 0) => (0, EXTENDED_QOS_L3_BANDWIDTH, 0, 0),
        (LEAF_EXTENDED_QOS, 1) => (BANDWIDTH_WIDTH, 0, 0, highest_class),
        _ => (0, 0, 0, 0),
    }
}

#[derive(Clone)]
pub(crate) struct Rdt {
    association: u64,
    event_select: u64,
    l3_mask: [u64; CLASSES_OF_SERVICE],
    bandwidth: [u64; CLASSES_OF_SERVICE],
}

impl Rdt {
    pub fn new() -> Self {
        Rdt {
            association: 0,
            event_select: 0,
            l3_mask: [L3_MASK_ALL; CLASSES_OF_SERVICE],
            bandwidth: [BANDWIDTH_UNLIMITED; CLASSES_OF_SERVICE],
        }
    }

    pub fn owns(msr: u32) -> bool {
        matches!(
            msr,
            MSR_QM_EVTSEL..=MSR_L3_MASK_LAST | MSR_L3_BANDWIDTH_FIRST..=MSR_L3_BANDWIDTH_LAST
        )
    }

    pub fn read(&self, msr: u32) -> Option<u64> {
        Some(match msr {
            MSR_QM_EVTSEL => self.event_select,
            MSR_QM_CTR => {
                let event = self.event_select & EVTSEL_EVENT_MASK;
                let rmid = self.event_select >> EVTSEL_RMID_SHIFT;
                let known = matches!(
                    event,
                    EVENT_L3_OCCUPANCY | EVENT_TOTAL_BANDWIDTH | EVENT_LOCAL_BANDWIDTH
                );
                if known && rmid <= MAX_RMID {
                    0
                } else {
                    QM_CTR_ERROR
                }
            }
            MSR_PQR_ASSOC => self.association,
            MSR_L3_MASK_FIRST..=MSR_L3_MASK_LAST => {
                self.l3_mask[(msr - MSR_L3_MASK_FIRST) as usize]
            }
            MSR_L3_BANDWIDTH_FIRST..=MSR_L3_BANDWIDTH_LAST => {
                self.bandwidth[(msr - MSR_L3_BANDWIDTH_FIRST) as usize]
            }
            _ => return None,
        })
    }

    /// `false` is #GP: QM_CTR is read-only, and reserved bits, an RMID past the highest or a
    /// class of service past the last fault.
    pub fn write(&mut self, msr: u32, value: u64) -> bool {
        match msr {
            MSR_QM_EVTSEL => {
                if value & !(EVTSEL_EVENT_MASK | EVTSEL_RMID_MASK) != 0 {
                    return false;
                }
                self.event_select = value;
            }
            MSR_PQR_ASSOC => {
                let rmid = value & PQR_RMID_MASK;
                let class = value >> PQR_CLOS_SHIFT;
                let reserved = value & !PQR_RMID_MASK & ((1 << PQR_CLOS_SHIFT) - 1);
                if reserved != 0 || rmid > MAX_RMID || class >= CLASSES_OF_SERVICE as u64 {
                    return false;
                }
                self.association = value;
            }
            MSR_L3_MASK_FIRST..=MSR_L3_MASK_LAST => {
                if value & !L3_MASK_ALL != 0 {
                    return false;
                }
                self.l3_mask[(msr - MSR_L3_MASK_FIRST) as usize] = value;
            }
            MSR_L3_BANDWIDTH_FIRST..=MSR_L3_BANDWIDTH_LAST => {
                if value & !BANDWIDTH_WRITABLE != 0 {
                    return false;
                }
                self.bandwidth[(msr - MSR_L3_BANDWIDTH_FIRST) as usize] = value;
            }
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_read_valid_zero_for_known_events_and_rmids() {
        let mut rdt = Rdt::new();
        for event in 1..=3u64 {
            for rmid in [0u64, 255] {
                assert!(rdt.write(0xc8d, rmid << 32 | event));
                assert_eq!(rdt.read(0xc8e), Some(0), "event {event} rmid {rmid}");
            }
        }
        assert!(rdt.write(0xc8d, 256 << 32 | 1));
        assert_eq!(rdt.read(0xc8e), Some(1 << 63));
        assert!(rdt.write(0xc8d, 4));
        assert_eq!(rdt.read(0xc8e), Some(1 << 63));
        assert!(!rdt.write(0xc8e, 0));
        assert!(!rdt.write(0xc8d, 1 << 8));
    }

    #[test]
    fn association_takes_an_rmid_and_a_class_of_service_in_range() {
        let mut rdt = Rdt::new();
        assert!(rdt.write(0xc8f, 15 << 32 | 255));
        assert_eq!(rdt.read(0xc8f), Some(15 << 32 | 255));
        assert!(!rdt.write(0xc8f, 16 << 32));
        assert!(!rdt.write(0xc8f, 256));
        assert!(!rdt.write(0xc8f, 1 << 10));
        assert_eq!(rdt.read(0xc8f), Some(15 << 32 | 255));
    }

    #[test]
    fn l3_masks_reset_to_every_way_and_take_any_16_bit_mask() {
        let mut rdt = Rdt::new();
        for class in 0..16 {
            assert_eq!(rdt.read(0xc90 + class), Some(0xffff));
        }
        // empty and non-contiguous masks are fine on AMD
        for mask in [0, 0x0f0f, 0x8001] {
            assert!(rdt.write(0xc9f, mask));
            assert_eq!(rdt.read(0xc9f), Some(mask));
        }
        assert!(!rdt.write(0xc90, 0x1_0000));
        assert!(!Rdt::owns(0xca0));
    }

    #[test]
    fn bandwidth_limits_reset_to_unlimited_and_take_12_bits() {
        let mut rdt = Rdt::new();
        assert_eq!(rdt.read(0xc000_0200), Some(1 << 11));
        assert!(rdt.write(0xc000_020f, 0x7ff));
        assert_eq!(rdt.read(0xc000_020f), Some(0x7ff));
        assert!(!rdt.write(0xc000_0200, 1 << 12));
        assert!(!Rdt::owns(0xc000_0210));
        assert!(!Rdt::owns(0xc8c));
    }

    #[test]
    fn the_leaves_agree_with_the_registers() {
        let (_, max_rmid, _, resources) = cpuid(0xf, 0);
        assert_eq!((max_rmid, resources), (255, 2));
        let (width, scale, l3_max_rmid, events) = cpuid(0xf, 1);
        assert_eq!((width, scale, l3_max_rmid, events), (0, 64, 255, 7));
        assert_eq!(cpuid(0x10, 0), (0, 2, 0, 0));
        let (mask_length, shared, cdp, highest_class) = cpuid(0x10, 1);
        assert_eq!(
            (mask_length + 1, shared, cdp, highest_class + 1),
            (16, 0, 0, 16)
        );
        assert_eq!(cpuid(0x8000_0020, 0), (0, 2, 0, 0));
        let (bandwidth_width, _, _, bandwidth_highest_class) = cpuid(0x8000_0020, 1);
        assert_eq!((bandwidth_width, bandwidth_highest_class), (11, 15));
        for (leaf, subleaf) in [
            (0xf, 2),
            (0x10, 2),
            (0x10, 3),
            (0x8000_0020, 2),
            (0x8000_0020, 3),
        ] {
            assert_eq!(cpuid(leaf, subleaf), (0, 0, 0, 0), "{leaf:#x}.{subleaf}");
        }
    }
}
