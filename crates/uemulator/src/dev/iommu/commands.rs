//! The command buffer (spec 2.4): 16-byte commands software appends at the tail, the IOMMU
//! consumes from the head. Opcode is bits 63:60 of the first qword.
use super::{
    CONTROL_COMMAND_BUFFER_ENABLE, CONTROL_COMPLETION_WAIT_INTERRUPT_ENABLE, Iommu, PhysWrite,
    RING_ENTRY_BYTES, STATUS_COMMAND_BUFFER_RUNNING, STATUS_COMPLETION_WAIT_INTERRUPT,
};

const OPCODE_SHIFT: u32 = 60;
const OPCODE_COMPLETION_WAIT: u64 = 0x1;
const OPCODE_INVALIDATE_DEVICE_TABLE_ENTRY: u64 = 0x2;
const OPCODE_INVALIDATE_IOMMU_PAGES: u64 = 0x3;
const OPCODE_INVALIDATE_IOTLB_PAGES: u64 = 0x4;
const OPCODE_INVALIDATE_INTERRUPT_TABLE: u64 = 0x5;
const OPCODE_PREFETCH_IOMMU_PAGES: u64 = 0x6;
const OPCODE_COMPLETE_PPR_REQUEST: u64 = 0x7;
const OPCODE_INVALIDATE_IOMMU_ALL: u64 = 0x8;

/// COMPLETION_WAIT: bit 0 (s) stores the second qword at the address in bits 51:3, bit 1 (i)
/// raises the completion-wait interrupt, bit 2 (f) orders it after earlier commands, which are
/// all complete by the time this one runs.
const COMPLETION_WAIT_STORE: u64 = 1 << 0;
const COMPLETION_WAIT_INTERRUPT: u64 = 1 << 1;
const COMPLETION_WAIT_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_fff8;
/// Device id in the low 16 bits of the first dword; domain id in bits 47:32 of the first qword.
const COMMAND_DEVICE_ID_MASK: u64 = 0xffff;
const COMMAND_DOMAIN_ID_SHIFT: u32 = 32;
const COMMAND_DOMAIN_ID_MASK: u64 = 0xffff;

impl Iommu {
    /// Executes every command between head and tail. Stops, with CmdBufRun clear, at one it does
    /// not understand (the head stays on it so software can find it) and resumes only after
    /// software toggles CmdBufEn.
    pub(super) fn run_commands(&mut self, mem: &mut dyn PhysWrite) {
        if self.control() & CONTROL_COMMAND_BUFFER_ENABLE == 0 || self.command_stalled {
            return;
        }
        let (base, bytes) = self.command_buffer();
        if bytes == 0 {
            return;
        }
        while self.command_head() % bytes != self.command_tail() % bytes {
            let head = self.command_head() % bytes;
            let address = base + head;
            let mut raw = [0u8; RING_ENTRY_BYTES as usize];
            if !mem.read_phys(address, &mut raw) {
                self.log_command_error(address);
                self.stall_commands();
                return;
            }
            let low = u64::from_le_bytes(raw[..8].try_into().unwrap());
            let high = u64::from_le_bytes(raw[8..].try_into().unwrap());
            if !self.execute_command(mem, address, low, high) {
                self.stall_commands();
                return;
            }
            self.set_command_head((head + RING_ENTRY_BYTES) % bytes);
        }
    }

    fn stall_commands(&mut self) {
        self.command_stalled = true;
        self.status_clear(STATUS_COMMAND_BUFFER_RUNNING);
    }

    /// False if the command was illegal or could not complete, after queueing the event.
    fn execute_command(
        &mut self,
        mem: &mut dyn PhysWrite,
        address: u64,
        low: u64,
        high: u64,
    ) -> bool {
        match low >> OPCODE_SHIFT {
            OPCODE_COMPLETION_WAIT => {
                if low & COMPLETION_WAIT_STORE != 0 {
                    let store_to = low & COMPLETION_WAIT_ADDRESS_MASK;
                    if !mem.write_phys(store_to, &high.to_le_bytes()) {
                        self.log_command_error(address);
                        return false;
                    }
                }
                if low & COMPLETION_WAIT_INTERRUPT != 0 {
                    self.status_set(STATUS_COMPLETION_WAIT_INTERRUPT);
                    if self.control() & CONTROL_COMPLETION_WAIT_INTERRUPT_ENABLE != 0 {
                        self.raise_msi();
                    }
                }
            }
            OPCODE_INVALIDATE_DEVICE_TABLE_ENTRY => {
                // The entry and the translations made through it.
                let device = (low & COMMAND_DEVICE_ID_MASK) as u16;
                self.forget_device_entry(device);
                self.iotlb_remove_device(device);
            }
            OPCODE_INVALIDATE_IOTLB_PAGES => {
                self.iotlb_remove_device((low & COMMAND_DEVICE_ID_MASK) as u16);
            }
            OPCODE_INVALIDATE_IOMMU_PAGES => {
                // The address and size bits narrow the invalidation; dropping every page of
                // the domain is always a legal over-approximation.
                let domain = (low >> COMMAND_DOMAIN_ID_SHIFT) & COMMAND_DOMAIN_ID_MASK;
                self.iotlb_remove_domain(domain as u16);
            }
            // Interrupt remapping table entries are read from RAM on every use.
            OPCODE_INVALIDATE_INTERRUPT_TABLE
            | OPCODE_PREFETCH_IOMMU_PAGES
            | OPCODE_COMPLETE_PPR_REQUEST => {}
            OPCODE_INVALIDATE_IOMMU_ALL => self.forget_device_table(),
            _ => {
                self.log_illegal_command(address);
                return false;
            }
        }
        true
    }
}
