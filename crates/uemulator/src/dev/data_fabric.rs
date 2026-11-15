//! The AMD platform functions on bus 0 that are not devices of their own: the Turin root complex
//! at 00:00.0 and the data fabric of node 0 at 00:18.0-7. IDs are Turin's (pci.ids "Turin Root
//! Complex" 1022:153A, "Turin Data Fabric; Function 0-7" 1022:12C0-12C7; Linux's `pci_ids.h`
//! has `PCI_DEVICE_ID_AMD_1AH_M00H_DF_F3` = 0x12C3, and its amd_pmc names 0x153A as a family 1Ah
//! root).
//!
//! What Linux does with them (`arch/x86/kernel/amd_node.c`, `amd_nb.c`, 6.14 and later):
//! - amd_smn finds every AMD host bridge at devfn 0 as a root and reaches the System Management
//!   Network through its index/data pair at config 0x60/0x64. Reads of unknown SMN addresses
//!   must not be all ones, which `amd_smn_read` reports as an error.
//! - amd_nb takes function 3 of slot 0x18 + node as the node's "misc" device and function 4 as
//!   its "link"; one node per `CPUID 0x8000_001E ECX` (one here).
//! - k10temp binds to the function 3 id and reads the reported temperature at SMN 0x59800.
//! - amd64_edac reads the memory controllers' UMC registers over SMN; they read zero here (no
//!   channel initialized, no ECC), so it reports ECC disabled and does not load.
//!
//! Kernels before 6.14 matched the root (0x153A) and function 3/4 ids from tables instead; both
//! are the ones listed there for family 1Ah models 00h-0Fh.
pub(crate) const VENDOR_AMD: u16 = 0x1022;
pub(crate) const DEVICE_ROOT_COMPLEX: u16 = 0x153a;
/// Function `n` of the data fabric is this plus `n`.
pub(crate) const DEVICE_DATA_FABRIC_FUNCTION_0: u16 = 0x12c0;
/// Node 0's data fabric is device 0x18 (`AMD_NODE0_PCI_SLOT`), all eight functions.
pub(crate) const DATA_FABRIC_SLOT: u8 = 0x18;
pub(crate) const DATA_FABRIC_FUNCTIONS: u8 = 8;
const DEVFN_SLOT_SHIFT: u32 = 3;
pub(crate) const DATA_FABRIC_FIRST_DEVFN: u8 = DATA_FABRIC_SLOT << DEVFN_SLOT_SHIFT;

pub(crate) const CONFIG_SPACE_SIZE: usize = 256;
const CFG_VENDOR_ID: usize = 0x00;
const CFG_DEVICE_ID: usize = 0x02;
const CFG_SUBCLASS: usize = 0x0a;
const CFG_CLASS: usize = 0x0b;
const CFG_HEADER_TYPE: usize = 0x0e;
const CFG_SUBSYSTEM_VENDOR_ID: usize = 0x2c;
const HEADER_TYPE_MULTIFUNCTION: u8 = 0x80;
/// Both the root complex and every data fabric function are class 06h/00h (host bridge), as on
/// the hardware. amd_smn tells the root from the fabric by devfn 0.
const CLASS_BRIDGE: u8 = 0x06;
const SUBCLASS_HOST_BRIDGE: u8 = 0x00;

/// SMN index and data registers in the root complex's config space (`SMN_INDEX_OFFSET`,
/// `SMN_DATA_OFFSET`). The HSMP pair at 0xC4/0xC8 is not implemented: it is a mailbox to the
/// system management unit, which this machine has none of.
pub(crate) const SMN_INDEX: usize = 0x60;
pub(crate) const SMN_DATA: usize = 0x64;
const DWORD_BYTES: usize = 4;

/// THM_TCON_CUR_TMP (`ZEN_REPORTED_TEMP_CTRL_BASE` in k10temp): the current temperature in
/// eighths of a degree Celsius in bits 31:21. The range-select bit 19 (which would mean "minus
/// 49 degrees") stays clear.
pub(crate) const SMN_REPORTED_TEMPERATURE: u32 = 0x0005_9800;
const TEMPERATURE_SHIFT: u32 = 21;
const EIGHTHS_PER_DEGREE: u32 = 8;
/// A plausible idle temperature; nothing here heats up.
pub(crate) const REPORTED_DEGREES_CELSIUS: u32 = 40;

/// DF::FabricBlockInstanceCount at function 0 offset 0x40, whose bits 27:24 are the fabric's
/// major revision (`DF_MAJOR_REVISION` in amd_node.c). Zen 4 and 5 have DF4, where the
/// configuration address control register moves to 0xC04 in extended config space; that reads
/// zero here, which says the root of node 0 is on bus 0.
const DF_BLOCK_INSTANCE_COUNT: usize = 0x40;
const DF_MAJOR_REVISION_SHIFT: u32 = 24;
const DF_MAJOR_REVISION_4: u32 = 4;

/// The SMN registers, by address. Everything not listed reads zero and ignores writes.
fn smn_read(address: u32) -> u32 {
    match address {
        SMN_REPORTED_TEMPERATURE => {
            (REPORTED_DEGREES_CELSIUS * EIGHTHS_PER_DEGREE) << TEMPERATURE_SHIFT
        }
        _ => 0,
    }
}

fn put(config: &mut [u8; CONFIG_SPACE_SIZE], at: usize, bytes: &[u8]) {
    config[at..at + bytes.len()].copy_from_slice(bytes);
}

fn host_bridge_header(device: u16) -> [u8; CONFIG_SPACE_SIZE] {
    let mut config = [0u8; CONFIG_SPACE_SIZE];
    put(&mut config, CFG_VENDOR_ID, &VENDOR_AMD.to_le_bytes());
    put(&mut config, CFG_DEVICE_ID, &device.to_le_bytes());
    put(
        &mut config,
        CFG_SUBSYSTEM_VENDOR_ID,
        &VENDOR_AMD.to_le_bytes(),
    );
    config[CFG_SUBCLASS] = SUBCLASS_HOST_BRIDGE;
    config[CFG_CLASS] = CLASS_BRIDGE;
    config
}

/// 00:00.0. Only the SMN index is writable; everything else is read-only.
#[derive(Default)]
pub(crate) struct RootComplex {
    smn_index: u32,
}

impl RootComplex {
    /// `multifunction` when the IOMMU sits at 00:00.2: the OS scans the other functions only
    /// if function 0 says so.
    pub fn config(&self, multifunction: bool) -> [u8; CONFIG_SPACE_SIZE] {
        let mut config = host_bridge_header(DEVICE_ROOT_COMPLEX);
        if multifunction {
            config[CFG_HEADER_TYPE] = HEADER_TYPE_MULTIFUNCTION;
        }
        put(&mut config, SMN_INDEX, &self.smn_index.to_le_bytes());
        put(
            &mut config,
            SMN_DATA,
            &smn_read(self.smn_index).to_le_bytes(),
        );
        config
    }

    /// Byte-wise, so narrow writes to the index replace only their own bytes. SMN data writes
    /// are accepted and dropped: no SMN register here is writable.
    pub fn config_write(&mut self, register: usize, bytes: usize, value: u32) {
        for byte in 0..bytes {
            let at = register + byte;
            if (SMN_INDEX..SMN_INDEX + DWORD_BYTES).contains(&at) {
                let shift = 8 * (at - SMN_INDEX);
                let lane = ((value >> (8 * byte)) & 0xff) << shift;
                self.smn_index = (self.smn_index & !(0xff << shift)) | lane;
            }
        }
    }
}

pub(crate) fn is_data_fabric(devfn: u8) -> bool {
    (DATA_FABRIC_FIRST_DEVFN..DATA_FABRIC_FIRST_DEVFN + DATA_FABRIC_FUNCTIONS).contains(&devfn)
}

/// Config space of data fabric function `devfn & 7`; read-only.
pub(crate) fn data_fabric_config(devfn: u8) -> [u8; CONFIG_SPACE_SIZE] {
    let function = devfn - DATA_FABRIC_FIRST_DEVFN;
    let mut config = host_bridge_header(DEVICE_DATA_FABRIC_FUNCTION_0 + function as u16);
    if function == 0 {
        config[CFG_HEADER_TYPE] = HEADER_TYPE_MULTIFUNCTION;
        let block_instances = DF_MAJOR_REVISION_4 << DF_MAJOR_REVISION_SHIFT;
        put(
            &mut config,
            DF_BLOCK_INSTANCE_COUNT,
            &block_instances.to_le_bytes(),
        );
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dword(config: &[u8; CONFIG_SPACE_SIZE], at: usize) -> u32 {
        u32::from_le_bytes(config[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn smn_index_selects_what_the_data_register_reads() {
        let mut root = RootComplex::default();
        root.config_write(SMN_INDEX, 4, SMN_REPORTED_TEMPERATURE);
        let config = root.config(false);
        assert_eq!(dword(&config, SMN_INDEX), SMN_REPORTED_TEMPERATURE);
        let raw = dword(&config, SMN_DATA);
        // k10temp: (raw >> 21) * 125 millidegrees, minus 49 degrees only with bit 19 set.
        assert_eq!((raw >> 21) * 125, REPORTED_DEGREES_CELSIUS * 1000);
        assert_eq!(raw & 1 << 19, 0);
    }

    #[test]
    fn unknown_smn_registers_read_zero_not_the_pci_error_value() {
        let mut root = RootComplex::default();
        root.config_write(SMN_INDEX, 4, 0x0005_0000);
        assert_eq!(dword(&root.config(false), SMN_DATA), 0);
        root.config_write(SMN_DATA, 4, u32::MAX);
        assert_eq!(dword(&root.config(false), SMN_DATA), 0);
    }

    #[test]
    fn narrow_index_writes_replace_only_their_bytes() {
        let mut root = RootComplex::default();
        root.config_write(SMN_INDEX, 4, 0x1122_3344);
        root.config_write(SMN_INDEX + 2, 1, 0xaa);
        assert_eq!(dword(&root.config(false), SMN_INDEX), 0x11aa_3344);
    }

    #[test]
    fn data_fabric_functions_have_the_turin_ids_and_function_0_is_multifunction() {
        for function in 0..DATA_FABRIC_FUNCTIONS {
            let devfn = DATA_FABRIC_FIRST_DEVFN + function;
            assert!(is_data_fabric(devfn));
            let config = data_fabric_config(devfn);
            assert_eq!(dword(&config, 0), 0x1022 | (0x12c0 + function as u32) << 16);
            assert_eq!(dword(&config, 8) >> 16, 0x0600);
            assert_eq!(
                config[CFG_HEADER_TYPE] == HEADER_TYPE_MULTIFUNCTION,
                function == 0
            );
        }
        assert!(!is_data_fabric(DATA_FABRIC_FIRST_DEVFN - 1));
        assert!(!is_data_fabric(DATA_FABRIC_FIRST_DEVFN + 8));
        let f0 = data_fabric_config(DATA_FABRIC_FIRST_DEVFN);
        assert_eq!(dword(&f0, DF_BLOCK_INSTANCE_COUNT) >> 24 & 0xf, 4);
    }
}
