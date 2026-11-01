#[cfg(freestanding)]
use crate::prelude::*;
pub(crate) mod acpi_pm;
pub(crate) mod apic;
pub(crate) mod bootfb;
pub(crate) mod guestmem;
pub(crate) mod handle;
pub(crate) mod hpet;
pub(crate) mod ioapic;
pub(crate) mod iommu;
pub(crate) mod kv;
pub(crate) mod pci;
mod pic;
mod pit;
mod serial;
mod vblk;
mod vcons;
mod vgpu;
mod vinput;
mod virtio;
mod vnet;
mod vrng;

use crate::console::ConsoleHandle;
use crate::disk::Disk;
use crate::display::{DisplayHandle, InputDevice, InputHandle};
use crate::netdev::NetHandle;
use crate::sync::Arc;
use acpi_pm::AcpiPm;
use apic::{Interrupt, Ipi, Lapic};
use bootfb::{BootFramebuffer, FramebufferRegion};
use guestmem::GuestMem;
use hpet::{Hpet, HpetOutput};
use ioapic::IoApic;
use iommu::{InterruptRequest, IommuInterrupt, PhysRead, ioapic_request_raw};
use kv::{KV_PORT_BASE, KV_PORT_COUNT, KvDevice, SharedKeyValueStore};
use pci::PciHost;
use pic::Pics;
use pit::Pit;
use serial::Uart;
use vblk::VirtioBlk;
use vcons::VirtioConsole;
use vgpu::VirtioGpu;
use vinput::VirtioInput;
use virtio::Transport;
use vnet::VirtioNet;
use vrng::VirtioRng;
pub(crate) use vrng::{EntropyFn, RngSource, host_entropy};

/// PIC wiring: IRQ lines used by the legacy devices we model.
const IRQ_PIT: u8 = 0;
const IRQ_UART: u8 = 4;
/// One 8N1 character (start, eight data, stop) at 115200 baud, in nanoseconds: how fast host
/// input reaches the UART.
const UART_BYTE_INTERVAL_NS: u64 = 10 * 1_000_000_000 / 115_200;

/// I/O ports. Anything not listed reads as all ones and ignores writes.
const PIC_MASTER_COMMAND: u16 = 0x20;
const PIC_MASTER_DATA: u16 = 0x21;
const PIC_SLAVE_COMMAND: u16 = 0xa0;
const PIC_SLAVE_DATA: u16 = 0xa1;
/// Edge/level control registers of the two PICs.
const PIC_ELCR_MASTER: u16 = 0x4d0;
const PIC_ELCR_SLAVE: u16 = 0x4d1;
const PIT_CHANNEL0: u16 = 0x40;
const PIT_CHANNEL2: u16 = 0x42;
const PIT_COMMAND: u16 = 0x43;
const SYSTEM_CONTROL_PORT_B: u16 = 0x61;
const CMOS_INDEX_PORT: u16 = 0x70;
const CMOS_DATA_PORT: u16 = 0x71;
/// i8042 command port; only the reset command is emulated.
const KBC_COMMAND_PORT: u16 = 0x64;
const FAST_A20_PORT: u16 = 0x92;
const RESET_CONTROL_PORT: u16 = 0xcf9;
/// PCI config mechanism #1 address (0xCF8..=0xCFB) and data (0xCFC..=0xCFF) windows. 0xCF9 sits
/// inside the address window but is the reset control register, so its match arm must come first.
const PCI_CONFIG_PORT_FIRST: u16 = 0xcf8;
const PCI_CONFIG_PORT_LAST: u16 = 0xcff;
const KV_PORT_FIRST: u16 = KV_PORT_BASE;
const KV_PORT_LAST: u16 = KV_PORT_BASE + KV_PORT_COUNT - 1;
const UART_COM1_BASE: u16 = 0x3f8;
const UART_COM1_LAST: u16 = 0x3ff;

/// i8042 command 0xFE pulses the CPU reset line.
const KBC_COMMAND_PULSE_RESET: u8 = 0xfe;
/// Port 0x92: bit 1 is the A20 gate (stored, not acted on), bit 0 resets the CPU.
const FAST_A20_GATE: u8 = 1 << 1;
const FAST_A20_RESET: u8 = 1 << 0;
/// Port 0xCF9 bit 2 (RST_CPU) triggers the reset; bit 1 selects a hard (1) or soft (0) reset
/// and is the only bit that reads back, as in QEMU's PIIX. Both kinds reset the machine here.
const RESET_CONTROL_RESET_CPU: u8 = 1 << 2;
const RESET_CONTROL_SYSTEM_RESET: u8 = 1 << 1;

/// Port 0x61 bits. Only the low two are writable (timer 2 gate, speaker data enable); the
/// others are synthesized on read.
const PORT_B_WRITABLE_MASK: u8 = 0x03;
const PORT_B_TIMER2_GATE: u8 = 1 << 0;
const PORT_B_REFRESH_TOGGLE: u8 = 1 << 4;
const PORT_B_TIMER2_OUT: u8 = 1 << 5;
/// The refresh toggle flips every 15.085 us (PIT channel 1 divisor 18 at 1.193182 MHz), so
/// guests busy-waiting on it see time pass. Derived from virtual time to stay deterministic.
const REFRESH_TOGGLE_PERIOD_NS: u64 = 15085;

/// xAPIC registers are 16 bytes apart; the offset within the 4 KiB page is at most 0xff0 + 0xf.
const APIC_REGISTER_MASK: u64 = 0xff0;

const NS_PER_SECOND: u64 = 1_000_000_000;
const SECONDS_PER_MINUTE: u64 = 60;
const SECONDS_PER_HOUR: u64 = 3600;
const SECONDS_PER_DAY: u64 = 86400;
/// 1970-01-01 was a Thursday; CMOS day-of-week counts Sunday as 1.
const EPOCH_DAYS_SINCE_SUNDAY: u64 = 4;
const DAYS_PER_WEEK: u64 = 7;

/// CMOS/RTC register indices. Alarm registers (1, 3, 5) and unlisted ones are plain RAM.
const CMOS_SIZE: usize = 128;
/// Bit 7 of the index port is the NMI mask, not part of the index.
const CMOS_INDEX_MASK: u8 = 0x7f;
const CMOS_SECONDS: usize = 0x00;
const CMOS_MINUTES: usize = 0x02;
const CMOS_HOURS: usize = 0x04;
const CMOS_DAY_OF_WEEK: usize = 0x06;
const CMOS_DAY_OF_MONTH: usize = 0x07;
const CMOS_MONTH: usize = 0x08;
const CMOS_YEAR: usize = 0x09;
const CMOS_STATUS_A: usize = 0x0a;
const CMOS_STATUS_B: usize = 0x0b;
const CMOS_STATUS_C: usize = 0x0c;
const CMOS_STATUS_D: usize = 0x0d;
const CMOS_CENTURY: usize = 0x32;
/// Status A: update-in-progress clear, 32.768 kHz time base, 1024 Hz periodic rate.
const CMOS_STATUS_A_VALUE: u8 = 0x26;
/// Status B: 24-hour mode, BCD, no interrupts enabled.
const CMOS_STATUS_B_VALUE: u8 = 0x02;
/// Status D: battery good (VRT), so the kernel trusts the time.
const CMOS_STATUS_D_VALUE: u8 = 0x80;

/// Requester ids the IOMMU sees for interrupts that do not come from a PCI function. Linux's
/// AMD IOMMU driver insists on finding an I/O APIC at the "southbridge" id 00:14.0 in the IVRS
/// (`IOAPIC_SB_DEVID`) and turns interrupt remapping off otherwise; the HPET gets the next
/// function. A virtio function in slot 0x14 (the 20th) would share the IOAPIC's table entry,
/// which still works because both only need the same interrupt table. Must match the IVRS
/// special-device entries (`acpi`).
pub(crate) const IOAPIC_REQUESTER: u16 = 0x00a0;
pub(crate) const HPET_REQUESTER: u16 = 0x00a1;

/// A device interrupt on its way to the LAPIC.
enum Source {
    /// An I/O APIC redirection entry's message and the pin that sent it.
    IoApic { pin: usize, interrupt: Interrupt },
    /// An MSI write from a PCI function, the HPET or the IOMMU itself.
    Message(InterruptRequest),
}

/// The screen's devices and the handle they share with the host.
#[derive(Clone)]
pub(crate) struct DisplayDevices {
    pub display: DisplayHandle,
    /// The UEFI firmware's framebuffer in RAM, shown while no virtio-gpu driver owns the screen.
    /// Only a machine booted through the firmware has one.
    pub boot_framebuffer: Option<FramebufferRegion>,
}

/// PCI functions `DevConfig::display` adds: the GPU.
pub(crate) const DISPLAY_FUNCTIONS: usize = 1;
/// PCI functions `DevConfig::input` adds: the keyboard and the tablet.
pub(crate) const INPUT_FUNCTIONS: usize = 2;

/// What to attach to the PCI bus; every device is optional and may repeat. Functions take slots
/// (and legacy IRQ lines) in this order: virtio consoles, disks, nets, the GPU, the keyboard and
/// tablet, then the entropy source, each in list order. A machine with both a display and input
/// has the GPU, keyboard and tablet in consecutive slots, as before they could be chosen apart:
/// installed guests and firmware boot entries remember slot numbers.
#[derive(Default)]
pub(crate) struct DevConfig {
    pub disks: Vec<Disk>,
    /// MAC address and the host side of the frame channel, one entry per NIC.
    pub nets: Vec<([u8; 6], NetHandle)>,
    /// One virtio-console function per handle. The UART stays on the `Devices::new` console
    /// handle; when that is one of these (the default), the UART gives way to the virtio driver.
    pub virtio_consoles: Vec<ConsoleHandle>,
    /// Adds the AMD IOMMU function (00:00.2) and translates every PCI function's DMA by it.
    pub iommu: bool,
    /// The store behind the firmware's variable device; its ports exist only when set.
    pub key_value_store: Option<SharedKeyValueStore>,
    /// How many vCPUs, so how many local APICs; 0 is taken as 1 (the default).
    pub cpus: usize,
    /// Adds a virtio-rng function, last in slot order, fed by this source.
    pub rng: Option<RngSource>,
    /// Adds the screen (`DISPLAY_FUNCTIONS` functions) after the NICs.
    pub display: Option<DisplayDevices>,
    /// Adds the keyboard and tablet (`INPUT_FUNCTIONS` functions) after the GPU.
    pub input: Option<InputHandle>,
}

/// Most virtio functions one machine can hold (PCI slots 1..=31).
pub(crate) const MAX_VIRTIO_FUNCTIONS: usize = pci::MAX_VIRTIO_FUNCTIONS;
/// The BARs the virtio functions are given, for the firmware's memory map.
pub(crate) const PCI_MMIO_BASE: u64 = pci::MMIO_BASE;
pub(crate) const PCI_MMIO_BYTES: u64 = pci::MAX_VIRTIO_FUNCTIONS as u64 * pci::BAR_SIZE;

/// The whole device bus: legacy port-I/O devices plus the virtio-pci host. Port I/O goes
/// through `io_in`/`io_out`, physical accesses above RAM through `mmio_read`/`mmio_write`.
/// Devices raise IRQs directly on `pic`; the CPU polls `pic.has_pending()`.
pub(crate) struct Devices {
    pub pic: Pics,
    pit: Pit,
    uart: Uart,
    cmos_index: u8,
    cmos: [u8; CMOS_SIZE],
    boot_unix: u64,
    system_control_port_b: u8,
    system_control_port_a: u8,
    reset_control: u8,
    pci: PciHost,
    kv: Option<KvDevice>,
    /// One local APIC per vCPU, indexed by vCPU number (which is also the APIC id it starts with).
    lapics: Vec<Lapic>,
    /// The vCPU that is executing; the CPU-facing APIC accesses (MMIO window, MSRs, CR8,
    /// interrupt acknowledge) go to its APIC. Set by the scheduler on every switch.
    current_cpu: usize,
    /// Where the previous lowest-priority interrupt went, so equal priorities take turns.
    lowest_priority_cursor: usize,
    ioapic: IoApic,
    pm: AcpiPm,
    hpet: Hpet,
    /// Level of the SCI line as last driven.
    sci_level: bool,
    /// Interrupts that need the IOMMU's interrupt table in guest RAM, waiting for a call that
    /// has RAM (`deliver`).
    deferred: Vec<Source>,
    /// Virtual time of the last tick or port access, for MMIO accesses that arrive without one.
    now_ns: u64,
    /// Position of the virtio console that shares the UART's console handle, if any: the UART
    /// stops taking host input once that console's driver is up.
    uart_peer_console: Option<usize>,
    /// Set by the reset ports (i8042, 0x92, 0xCF9); the CPU turns it into `ExitReason::Reset`.
    pub reset_requested: bool,
    /// The vCPU threads' shared state while the machine runs one thread per vCPU: what the device
    /// model tells them at every release of the device lock.
    links: Option<Arc<crate::cpu::vcpus::VcpuLinks>>,
    /// Scanned on every tick while the guest's GPU driver does not own the screen.
    boot_framebuffer: Option<BootFramebuffer>,
}

/// Mask of the low `size` bytes of a port value; sizes of 4 or more cover the whole dword.
fn size_mask(size: u8) -> u32 {
    match size {
        0 => 0,
        1..=3 => (1u32 << (8 * size as u32)) - 1,
        _ => u32::MAX,
    }
}

fn bcd(value: u64) -> u8 {
    (((value / 10) << 4) | (value % 10)) as u8
}

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian. Howard Hinnant's
/// civil_from_days: shifts the epoch to 0000-03-01 so the leap day is the last day of the
/// year, then works in 400-year eras.
fn civil(days_since_unix_epoch: i64) -> (i64, u64, u64) {
    const DAYS_FROM_0000_03_01_TO_UNIX_EPOCH: i64 = 719468;
    const DAYS_PER_ERA: i64 = 146097;
    const DAYS_PER_4_YEARS: i64 = 1460;
    const DAYS_PER_100_YEARS: i64 = 36524;
    const DAYS_PER_YEAR: i64 = 365;
    const YEARS_PER_ERA: i64 = 400;
    let shifted = days_since_unix_epoch + DAYS_FROM_0000_03_01_TO_UNIX_EPOCH;
    let era = shifted.div_euclid(DAYS_PER_ERA);
    let day_of_era = shifted.rem_euclid(DAYS_PER_ERA);
    let year_of_era = (day_of_era - day_of_era / DAYS_PER_4_YEARS
        + day_of_era / DAYS_PER_100_YEARS
        - day_of_era / (DAYS_PER_ERA - 1))
        / DAYS_PER_YEAR;
    let day_of_year =
        day_of_era - (DAYS_PER_YEAR * year_of_era + year_of_era / 4 - year_of_era / 100);
    // Month counted from March (0 = March); 153 is the day count of five consecutive months.
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_from_march + 2) / 5 + 1) as u64;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    } as u64;
    let year = year_of_era + era * YEARS_PER_ERA + (month <= 2) as i64;
    (year, month, day)
}

impl Devices {
    pub fn new(console: ConsoleHandle, cfg: DevConfig) -> Self {
        // Order decides PCI slot and legacy IRQ: consoles, disks, nets, GPU, input, rng.
        let mut virtio_devices = Vec::new();
        let mut uart_peer_console = None;
        for (position, handle) in cfg.virtio_consoles.into_iter().enumerate() {
            if uart_peer_console.is_none() && handle.is_same_console(&console) {
                uart_peer_console = Some(position);
            }
            virtio_devices.push(Transport::new(Box::new(VirtioConsole::new(handle))));
        }
        for disk in cfg.disks {
            virtio_devices.push(Transport::new(Box::new(VirtioBlk::new(disk))));
        }
        for (mac, handle) in cfg.nets {
            virtio_devices.push(Transport::new(Box::new(VirtioNet::new(mac, handle))));
        }
        let mut boot_framebuffer = None;
        if let Some(display) = cfg.display {
            virtio_devices.push(Transport::new(Box::new(VirtioGpu::new(
                display.display.clone(),
            ))));
            boot_framebuffer = display
                .boot_framebuffer
                .map(|region| BootFramebuffer::new(region, display.display));
        }
        if let Some(input) = cfg.input {
            for kind in [InputDevice::Keyboard, InputDevice::Tablet] {
                virtio_devices.push(Transport::new(Box::new(VirtioInput::new(
                    kind,
                    input.clone(),
                ))));
            }
        }
        if let Some(source) = cfg.rng {
            virtio_devices.push(Transport::new(Box::new(VirtioRng::new(source))));
        }
        #[cfg(not(freestanding))]
        let boot_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        #[cfg(freestanding)]
        let boot_unix = 0;
        Devices {
            pic: Pics::default(),
            pit: Pit::new(),
            uart: {
                let mut uart = Uart::new(console);
                uart.set_byte_interval(UART_BYTE_INTERVAL_NS);
                uart
            },
            cmos_index: 0,
            cmos: [0; CMOS_SIZE],
            boot_unix,
            system_control_port_b: 0,
            system_control_port_a: 0,
            reset_control: 0,
            pci: PciHost::new(virtio_devices).with_iommu(cfg.iommu),
            kv: cfg.key_value_store.map(KvDevice::new),
            lapics: (0..cfg.cpus.max(1) as u32).map(Lapic::new).collect(),
            current_cpu: 0,
            lowest_priority_cursor: 0,
            ioapic: IoApic::new(),
            pm: AcpiPm::new(),
            hpet: Hpet::new(),
            sci_level: false,
            deferred: Vec::new(),
            now_ns: 0,
            uart_peer_console,
            reset_requested: false,
            links: None,
            boot_framebuffer,
        }
    }

    /// Consumes the devices and returns the disks in attach order, for the machine to attach to
    /// the devices it builds for the next boot. Everything else (PCI config, virtqueues, UART,
    /// PIC, PIT) is dropped on purpose: a reboot must not leak state from the previous run.
    pub fn into_disks(self) -> Vec<Disk> {
        self.pci.into_disks()
    }

    /// Whether host input is still on its way to the 16550, for the BIOS keyboard.
    #[cfg(not(freestanding))]
    pub fn serial_input_pending(&self) -> bool {
        self.uart.input_pending()
    }

    /// How many disks the guest can see; the BIOS numbers them 0x80, 0x81, ... in this order.
    #[cfg(not(freestanding))]
    pub fn disk_count(&mut self) -> usize {
        (0..)
            .take_while(|index| self.pci.disk_backend(*index).is_some())
            .count()
    }

    /// The storage of the `index`-th disk.
    #[cfg(not(freestanding))]
    pub fn disk_backend(&mut self, index: usize) -> Option<&mut dyn crate::disk::BlockBackend> {
        self.pci.disk_backend(index)
    }

    pub fn state(&self, now: u64) -> String {
        format!("{} | {}", self.pic.state(), self.pit.state(now))
    }

    /// Pins the RTC to `unix` seconds at virtual time 0 (`--time`); guests must not depend on it.
    pub fn set_boot_time(&mut self, unix: u64) {
        self.boot_unix = unix;
    }

    /// Only the PIT schedules deadlines. The UART and virtio devices are polled from `tick`
    /// instead, which the CPU calls every `EVENT_INTERVAL` instructions and while idle, so
    /// host input and received frames wait at most one such interval.
    pub fn next_deadline(&self, now: u64) -> Option<u64> {
        [
            self.pit.next_deadline(),
            self.uart.next_deadline(),
            self.hpet.next_deadline(),
            self.pm.next_deadline(now),
        ]
        .into_iter()
        .chain(self.lapics.iter().map(Lapic::next_deadline))
        .flatten()
        .min()
    }

    /// The local APIC of the vCPU that is executing.
    pub fn lapic(&self) -> &Lapic {
        &self.lapics[self.current_cpu]
    }

    pub fn lapic_mut(&mut self) -> &mut Lapic {
        &mut self.lapics[self.current_cpu]
    }

    #[cfg(test)]
    pub fn cpu_count(&self) -> usize {
        self.lapics.len()
    }

    pub fn set_current_cpu(&mut self, cpu: usize) {
        self.current_cpu = cpu;
    }

    pub fn attach_links(&mut self, links: Arc<crate::cpu::vcpus::VcpuLinks>) {
        self.links = Some(links);
    }

    pub fn detach_links(&mut self) {
        self.links = None;
    }

    /// What vCPU `cpu` would be woken by, as `ATTENTION_*` device bits: an INIT or start-up IPI,
    /// an NMI, a deliverable interrupt (the CPU still checks IF and the interrupt shadow). The
    /// PIC's output goes to the bootstrap processor only.
    fn wake_bits(&self, cpu: usize) -> u32 {
        use crate::cpu::vcpus::{ATTENTION_IRQ, ATTENTION_NMI, ATTENTION_START};
        let lapic = &self.lapics[cpu];
        let mut bits = 0;
        if lapic.start_signal_pending() {
            bits |= ATTENTION_START;
        }
        if lapic.nmi_pending() {
            bits |= ATTENTION_NMI;
        }
        if lapic.pending_vector().is_some()
            || (cpu == 0 && lapic.extint_connected() && self.pic.has_pending())
        {
            bits |= ATTENTION_IRQ;
        }
        bits
    }

    /// Runs when the device lock is released with real locking on: tells every vCPU's thread what
    /// is now waiting for it (`cpu/vcpus.rs`), and the device thread that the guest may have
    /// handed the devices work. The bits are recomputed from scratch each time, so the vCPUs
    /// never need the lock just to ask.
    pub fn publish_wake_hints(&mut self) {
        let Some(links) = self.links.as_ref().map(Arc::as_ptr) else {
            return;
        };
        // `links` outlives this call: only `detach_links` drops it, and that runs after every
        // vCPU thread has been joined.
        let links = unsafe { &*links };
        self.route_line_events();
        for cpu in 0..self.lapics.len() {
            links.publish_device_bits(cpu, self.wake_bits(cpu));
        }
        links.poke_device_thread();
    }

    /// A vCPU's local APIC, for the scheduler's INIT and start-up handling.
    pub fn lapic_of(&mut self, cpu: usize) -> &mut Lapic {
        &mut self.lapics[cpu]
    }

    /// Whether a halted vCPU has something to wake it: an INIT or start-up IPI, an NMI, or (with
    /// interrupts enabled) a deliverable interrupt. The PIC's output goes to the bootstrap
    /// processor only.
    pub fn wake_pending(&mut self, cpu: usize, interrupts_enabled: bool) -> bool {
        self.route_line_events();
        let lapic = &self.lapics[cpu];
        lapic.start_signal_pending()
            || lapic.nmi_pending()
            || (interrupts_enabled
                && (lapic.pending_vector().is_some()
                    || (cpu == 0 && lapic.extint_connected() && self.pic.has_pending())))
    }

    /// Delivers an interrupt message to the local APICs it addresses (SDM 12.6): every APIC
    /// whose id or logical id matches, or for lowest-priority delivery the one of them that is
    /// least busy.
    fn route_interrupt(&mut self, interrupt: &Interrupt) {
        if let [only] = self.lapics.as_mut_slice() {
            only.accept(interrupt);
            return;
        }
        let mut targets = 0u64;
        for (index, lapic) in self.lapics.iter().enumerate() {
            if lapic.hardware_enabled() && lapic.matches_destination(interrupt) {
                targets |= 1 << index;
            }
        }
        if interrupt.delivery_mode == apic::DELIVERY_LOWEST_PRIORITY && targets.count_ones() > 1 {
            let count = self.lapics.len();
            let mut best: Option<(u8, usize)> = None;
            for step in 0..count {
                let index = (self.lowest_priority_cursor + step) % count;
                if targets >> index & 1 == 0 {
                    continue;
                }
                let priority = self.lapics[index].arbitration_priority();
                if best.is_none_or(|(lowest, _)| priority < lowest) {
                    best = Some((priority, index));
                }
            }
            if let Some((_, chosen)) = best {
                self.lowest_priority_cursor = chosen + 1;
                targets = 1 << chosen;
            }
        }
        for (index, lapic) in self.lapics.iter_mut().enumerate() {
            if targets >> index & 1 != 0 {
                lapic.accept_addressed(interrupt);
            }
        }
    }

    /// An IPI written to the ICR of the executing vCPU's APIC.
    fn send_ipi(&mut self, ipi: Ipi) {
        let sender = self.current_cpu;
        match ipi.shorthand {
            apic::SHORTHAND_NONE => self.route_interrupt(&ipi.interrupt),
            apic::SHORTHAND_SELF => {
                self.lapics[sender].accept_addressed(&ipi.interrupt);
            }
            apic::SHORTHAND_ALL_INCLUDING_SELF | apic::SHORTHAND_ALL_EXCLUDING_SELF => {
                let include_sender = ipi.shorthand == apic::SHORTHAND_ALL_INCLUDING_SELF;
                for (index, lapic) in self.lapics.iter_mut().enumerate() {
                    if include_sender || index != sender {
                        lapic.accept_addressed(&ipi.interrupt);
                    }
                }
            }
            // The field is two bits wide, so the four shorthands above are all there is.
            _ => {}
        }
    }

    /// The ISA line a device raised on the PIC, as an I/O APIC pin: the PIT's IRQ 0 is wired to
    /// pin 2 on PCs (the cascade has no device behind it), the rest map straight through.
    fn ioapic_pin(irq: u8) -> usize {
        if irq == IRQ_PIT { 2 } else { irq as usize }
    }

    /// Sends what the I/O APIC just produced towards the LAPIC. The pins come from
    /// `IoApic::take_sent_pins`, one per message in the same order.
    fn deliver_ioapic(&mut self, interrupts: Vec<Interrupt>, mem: Option<&dyn PhysRead>) {
        let pins = self.ioapic.take_sent_pins();
        debug_assert_eq!(pins.len(), interrupts.len());
        for (pin, interrupt) in pins.into_iter().zip(interrupts) {
            self.deliver(Source::IoApic { pin, interrupt }, mem);
        }
    }

    /// Hands a device interrupt to the LAPIC, through the IOMMU's interrupt remapping when the
    /// guest has the IOMMU on. Remapping reads the IRTE from guest RAM, which the callers that
    /// run on the CPU's interrupt-polling path do not have: `mem` is `None` there and the
    /// interrupt waits in `deferred` for the next `tick` (at most `EVENT_INTERVAL`
    /// instructions) or MMIO write.
    fn deliver(&mut self, source: Source, mem: Option<&dyn PhysRead>) {
        let remapping = self.pci.iommu().is_some_and(|iommu| iommu.enabled());
        if !remapping {
            let interrupt = match source {
                Source::IoApic { interrupt, .. } => Some(interrupt),
                Source::Message(request) => iommu::decode_message(&request)
                    .ok()
                    .map(|message| message.into_interrupt()),
            };
            if let Some(interrupt) = interrupt {
                self.route_interrupt(&interrupt);
            }
            return;
        }
        let Some(mem) = mem else {
            self.deferred.push(source);
            return;
        };
        let (request, pin) = match source {
            Source::IoApic { pin, interrupt } => {
                let request = ioapic_request_raw(
                    IOAPIC_REQUESTER,
                    interrupt.vector,
                    interrupt.delivery_mode,
                    interrupt.logical,
                    interrupt.level_triggered,
                    interrupt.destination as u8,
                );
                (request, Some(pin))
            }
            Source::Message(request) => (request, None),
        };
        let Some(iommu) = self.pci.iommu() else {
            return;
        };
        // A refused interrupt is dropped; the IOMMU has logged the fault.
        let Ok(remapped) = iommu.remap_interrupt(mem, request) else {
            return;
        };
        let interrupt = remapped.into_interrupt();
        if let Some(pin) = pin {
            self.ioapic.note_delivered(pin, interrupt.vector);
        }
        self.route_interrupt(&interrupt);
    }

    fn drain_deferred(&mut self, mem: &dyn PhysRead) {
        for source in core::mem::take(&mut self.deferred) {
            self.deliver(source, Some(mem));
        }
    }

    /// Forwards the device line changes the PIC logged to the I/O APIC, which turns them into
    /// messages for the local APIC. Cheap when nothing changed: the CPU calls it per instruction.
    fn route_line_events(&mut self) {
        self.route_line_events_with(None);
    }

    fn route_line_events_with(&mut self, mem: Option<&dyn PhysRead>) {
        if !self.pic.has_line_events() {
            return;
        }
        let mut messages = Vec::new();
        for (irq, level) in self.pic.take_line_events() {
            self.ioapic
                .set_line(Self::ioapic_pin(irq), level, &mut messages);
        }
        self.deliver_ioapic(messages, mem);
    }

    /// Queued HPET interrupts: I/O APIC pins (24 pins, so pins 20..=23 are not ISA lines and
    /// bypass the PIC) or MSI messages.
    fn route_hpet(&mut self, mem: Option<&dyn PhysRead>) {
        if !self.hpet.has_outputs() {
            return;
        }
        let mut messages = Vec::new();
        for output in self.hpet.take_outputs() {
            match output {
                HpetOutput::Pulse { pin } => {
                    self.ioapic.set_line(pin as usize, true, &mut messages);
                    self.ioapic.set_line(pin as usize, false, &mut messages);
                }
                HpetOutput::Level { pin, high } => {
                    self.ioapic.set_line(pin as usize, high, &mut messages);
                }
                HpetOutput::Message { address, data } => self.deliver(
                    Source::Message(InterruptRequest {
                        requester: HPET_REQUESTER,
                        address,
                        data,
                    }),
                    mem,
                ),
            }
        }
        self.deliver_ioapic(messages, mem);
    }

    /// The IOMMU's own event and completion-wait interrupt: its MSI (through interrupt
    /// remapping, like any device's) or, with IntCapXT, straight to the LAPIC.
    fn route_iommu_msi(&mut self, mem: &dyn PhysRead) {
        match self.pci.iommu().and_then(|iommu| iommu.take_interrupt()) {
            Some(IommuInterrupt::Message(message)) => self.deliver(
                Source::Message(InterruptRequest {
                    requester: iommu::REQUESTER_ID,
                    address: message.address,
                    data: message.data,
                }),
                Some(mem),
            ),
            Some(IommuInterrupt::Direct(interrupt)) => self.route_interrupt(&interrupt),
            None => {}
        }
    }

    /// Drives the SCI line to what the PM block wants.
    fn update_sci(&mut self, now: u64) {
        let level = self.pm.sci_asserted(now);
        if level == self.sci_level {
            return;
        }
        self.sci_level = level;
        if level {
            self.pic.raise(acpi_pm::SCI_IRQ);
        } else {
            self.pic.lower(acpi_pm::SCI_IRQ);
        }
    }

    /// The guest wrote SLP_EN with the S5 type: nothing wakes the machine from that.
    pub fn shutdown_requested(&self) -> bool {
        self.pm.shutdown_requested()
    }

    /// Whether an interrupt is waiting for the CPU (the CPU still checks IF and the interrupt
    /// shadow). Local APIC vectors win over the PIC's ExtINT.
    pub fn interrupt_pending(&mut self) -> bool {
        self.route_line_events();
        let lapic = &self.lapics[self.current_cpu];
        lapic.pending_vector().is_some()
            || (self.current_cpu == 0 && lapic.extint_connected() && self.pic.has_pending())
    }

    /// Interrupt acknowledge: the vector the CPU takes, if any.
    pub fn interrupt_ack(&mut self) -> Option<u8> {
        self.route_line_events();
        let lapic = &mut self.lapics[self.current_cpu];
        if let Some(vector) = lapic.ack() {
            return Some(vector);
        }
        if self.current_cpu == 0 && lapic.extint_connected() {
            return self.pic.ack();
        }
        None
    }

    /// An NMI the CPU should take at the next instruction boundary.
    pub fn take_nmi(&mut self) -> bool {
        self.lapic_mut().take_nmi()
    }

    pub fn nmi_pending(&self) -> bool {
        self.lapic().nmi_pending()
    }

    /// Whether any vCPU has an NMI waiting (the all-halted idle path asks).
    pub fn any_nmi_pending(&self) -> bool {
        self.lapics.iter().any(Lapic::nmi_pending)
    }

    /// With threads, the time a caller sampled before it got the device lock can be older than
    /// one another vCPU already used under it, and a counter read at the older time would run
    /// backwards for the guest (Linux watches for that). Time only moves forward here.
    fn clamp_time(&mut self, now: u64) -> u64 {
        if self.links.is_none() {
            return now;
        }
        self.now_ns = self.now_ns.max(now);
        self.now_ns
    }

    /// Reads an x2APIC register MSR; `None` is #GP (not in x2APIC mode, or a reserved register).
    pub fn apic_msr_read(&mut self, msr: u32, now: u64) -> Option<u64> {
        let now = self.clamp_time(now);
        if !self.lapic().x2apic() {
            return None;
        }
        let offset = (msr - apic::X2APIC_MSR_FIRST) << 4;
        if offset == 0x300 {
            // The ICR is one 64-bit register in x2APIC mode.
            return Some(self.lapic().read_icr_x2apic());
        }
        self.lapic_mut().read_register(offset, now).map(u64::from)
    }

    /// Writes an x2APIC register MSR; `false` is #GP.
    pub fn apic_msr_write(&mut self, msr: u32, value: u64, now: u64) -> bool {
        let now = self.clamp_time(now);
        if !self.lapic().x2apic() {
            return false;
        }
        let offset = (msr - apic::X2APIC_MSR_FIRST) << 4;
        if offset == 0x300 {
            if let Some(ipi) = self.lapic_mut().write_icr_x2apic(value) {
                self.send_ipi(ipi);
            }
            return true;
        }
        if value >> 32 != 0 {
            return false;
        }
        self.apic_register_write(offset, value as u32, now).is_ok()
    }

    fn apic_register_write(&mut self, offset: u32, value: u32, now: u64) -> Result<(), ()> {
        // Line changes the devices made before this access must be visible to the I/O APIC first:
        // an EOI re-sends a level interrupt only if the line is still high *now*.
        self.route_line_events();
        let effect = self.lapic_mut().write_register(offset, value, now)?;
        if let Some(vector) = effect.eoi_vector {
            let mut messages = Vec::new();
            self.ioapic.eoi(vector, &mut messages);
            self.deliver_ioapic(messages, None);
        }
        if let Some(ipi) = effect.ipi {
            self.send_ipi(ipi);
        }
        Ok(())
    }

    pub fn set_time(&mut self, now: u64) {
        self.now_ns = self.clamp_time(now);
    }

    /// The devices pulse `Pics::raise` on an interrupt event; a line the guest configured as
    /// level triggered (ELCR) also has to be released when the device stops asserting it, or it
    /// would re-request forever. Edge-triggered lines ignore `lower`.
    fn update_uart_irq(&mut self) {
        if self.uart.irq_edge() {
            self.pic.raise(IRQ_UART);
        }
        if !self.uart.irq_level() {
            self.pic.lower(IRQ_UART);
        }
    }

    fn update_pci_irqs(&mut self) {
        for irq in PciHost::interrupt_lines() {
            if !self.pci.line_asserted(irq) {
                self.pic.lower(irq);
            }
        }
    }

    /// `mem` collects code pages the devices wrote by DMA; the caller must pass them to
    /// `Cpu::code_written` afterwards (see `GuestMem`).
    pub fn tick(&mut self, now: u64, mem: &mut GuestMem) {
        let now = self.clamp_time(now);
        self.now_ns = now;
        for lapic in &mut self.lapics {
            lapic.tick(now);
        }
        if self.pit.tick(now) {
            self.pic.raise(IRQ_PIT);
        }
        self.hpet.tick(now);
        let mut raise = Vec::new();
        let mut messages = Vec::new();
        self.pci.poll_collecting(mem, &mut raise, &mut messages);
        if let Some(framebuffer) = &mut self.boot_framebuffer {
            framebuffer.tick(now, mem);
        }
        for irq in raise {
            self.pic.raise(irq);
        }
        self.update_pci_irqs();
        self.update_sci(now);
        // A UART that shares its console with a virtio console gives way once that driver is up:
        // host input belongs to hvc and the UART must not also take it. A UART with a console of
        // its own (`--console serial` next to a virtio one) always listens.
        let yielded = self
            .uart_peer_console
            .is_some_and(|position| self.pci.console_active(position));
        self.uart.set_rx_enabled(!yielded);
        self.uart.set_time(now);
        self.uart.poll();
        self.update_uart_irq();
        self.deliver_messages(messages, mem);
        self.route_line_events_with(Some(mem));
        self.route_hpet(Some(mem));
        self.route_iommu_msi(mem);
        self.drain_deferred(mem);
    }

    fn deliver_messages(&mut self, messages: Vec<InterruptRequest>, mem: &dyn PhysRead) {
        for request in messages {
            self.deliver(Source::Message(request), Some(mem));
        }
    }

    /// None means no device decodes `phys_addr` and the caller treats it as unmapped.
    pub fn mmio_read(&mut self, phys_addr: u64, size: u8) -> Option<u64> {
        if let Some(value) = self.pci.iommu_mmio_read(phys_addr, size) {
            return Some(value);
        }
        if self.lapic().mmio_decodes(phys_addr) {
            let offset = (phys_addr & APIC_REGISTER_MASK) as u32;
            let now = self.now_ns;
            return Some(self.lapic_mut().read_register(offset, now).unwrap_or(0) as u64);
        }
        if self.ioapic.decodes(phys_addr) {
            self.route_line_events();
            return Some(self.ioapic.mmio_read(phys_addr, size));
        }
        if Hpet::decodes(phys_addr) {
            return Some(self.hpet.mmio_read(phys_addr, size, self.now_ns));
        }
        if PciHost::ecam_decodes(phys_addr) {
            return Some(self.pci.ecam_read(phys_addr, size) as u64);
        }
        let value = self.pci.mmio_read(phys_addr, size);
        // Reading the ISR register acknowledges the interrupt.
        self.update_pci_irqs();
        value.map(|value| value as u64)
    }

    /// Returns false if no device decodes `phys_addr`.
    pub fn mmio_write(&mut self, phys_addr: u64, size: u8, val: u64, mem: &mut GuestMem) -> bool {
        if self.pci.iommu_mmio_write(phys_addr, size, val, mem) {
            self.route_iommu_msi(mem);
            self.drain_deferred(mem);
            return true;
        }
        if self.lapic().mmio_decodes(phys_addr) {
            let offset = (phys_addr & APIC_REGISTER_MASK) as u32;
            if self
                .apic_register_write(offset, val as u32, self.now_ns)
                .is_err()
            {
                self.lapic_mut().record_illegal_register();
            }
            return true;
        }
        if self.ioapic.decodes(phys_addr) {
            self.route_line_events_with(Some(mem));
            let mut messages = Vec::new();
            self.ioapic.mmio_write(phys_addr, val, &mut messages);
            self.deliver_ioapic(messages, Some(mem));
            // Reprogramming an entry can release interrupts that were waiting on the IOMMU.
            self.drain_deferred(mem);
            return true;
        }
        if Hpet::decodes(phys_addr) {
            self.hpet.mmio_write(phys_addr, size, val, self.now_ns);
            self.route_hpet(Some(mem));
            return true;
        }
        if PciHost::ecam_decodes(phys_addr) {
            self.pci.ecam_write(phys_addr, size, val as u32);
            return true;
        }
        let mut messages = Vec::new();
        let result =
            self.pci
                .mmio_write_collecting(phys_addr, size, val as u32, mem, &mut messages);
        let handled = match result {
            Some(irq) => {
                // IRQ 0 is the PIT and never a PCI line: it means "no interrupt".
                if irq != 0 {
                    self.pic.raise(irq);
                }
                self.update_pci_irqs();
                true
            }
            None => false,
        };
        if handled {
            self.deliver_messages(messages, mem);
            self.route_iommu_msi(mem);
        }
        handled
    }

    /// `now` is virtual nanoseconds. Time registers are computed on read from `boot_unix`
    /// plus elapsed virtual time; writes to them are stored but never read back.
    fn cmos_read(&self, now: u64) -> u8 {
        let index = (self.cmos_index & CMOS_INDEX_MASK) as usize;
        let unix_seconds = self.boot_unix.saturating_add(now / NS_PER_SECOND);
        let days = unix_seconds / SECONDS_PER_DAY;
        let (year, month, day) = civil(days as i64);
        let seconds_of_day = unix_seconds % SECONDS_PER_DAY;
        let day_of_week = (days + EPOCH_DAYS_SINCE_SUNDAY) % DAYS_PER_WEEK + 1;
        match index {
            CMOS_SECONDS => bcd(seconds_of_day % SECONDS_PER_MINUTE),
            CMOS_MINUTES => bcd((seconds_of_day / SECONDS_PER_MINUTE) % SECONDS_PER_MINUTE),
            CMOS_HOURS => bcd(seconds_of_day / SECONDS_PER_HOUR),
            CMOS_DAY_OF_WEEK => day_of_week as u8,
            CMOS_DAY_OF_MONTH => bcd(day),
            CMOS_MONTH => bcd(month),
            CMOS_YEAR => bcd(year as u64 % 100),
            CMOS_CENTURY => bcd(year as u64 / 100),
            CMOS_STATUS_A => CMOS_STATUS_A_VALUE,
            CMOS_STATUS_B => CMOS_STATUS_B_VALUE,
            CMOS_STATUS_C => 0,
            CMOS_STATUS_D => CMOS_STATUS_D_VALUE,
            _ => self.cmos[index],
        }
    }

    pub fn io_in(&mut self, port: u16, size: u8, now: u64) -> u32 {
        let now = self.clamp_time(now);
        self.now_ns = now;
        self.uart.set_time(now);
        let ones = size_mask(size);
        let value: u8 = match port {
            PIC_MASTER_COMMAND | PIC_MASTER_DATA | PIC_SLAVE_COMMAND | PIC_SLAVE_DATA
            | PIC_ELCR_MASTER | PIC_ELCR_SLAVE => self.pic.io_read(port),
            PIT_CHANNEL0..=PIT_CHANNEL2 => self.pit.read(port, now),
            SYSTEM_CONTROL_PORT_B => {
                let mut value = self.system_control_port_b & PORT_B_WRITABLE_MASK;
                if (now / REFRESH_TOGGLE_PERIOD_NS) & 1 != 0 {
                    value |= PORT_B_REFRESH_TOGGLE;
                }
                if self.pit.out2(now) {
                    value |= PORT_B_TIMER2_OUT;
                }
                value
            }
            CMOS_DATA_PORT => self.cmos_read(now),
            FAST_A20_PORT => self.system_control_port_a,
            RESET_CONTROL_PORT => self.reset_control,
            port if AcpiPm::decodes(port) => return self.pm.read(port, size, now),
            UART_COM1_BASE..=UART_COM1_LAST => self.uart.read(port - UART_COM1_BASE),
            PCI_CONFIG_PORT_FIRST..=PCI_CONFIG_PORT_LAST if !self.pci.is_empty() => {
                return self.pci.io_in(port, size);
            }
            KV_PORT_FIRST..=KV_PORT_LAST if self.kv.is_some() => {
                return self.kv.as_mut().unwrap().io_in(port, size);
            }
            _ => return ones,
        };
        if matches!(port, UART_COM1_BASE..=UART_COM1_LAST) {
            self.update_uart_irq();
        }
        value as u32 & ones
    }

    /// Port devices are byte-wide: only the low byte of `value` is used except for PCI config.
    pub fn io_out(&mut self, port: u16, size: u8, value: u32, now: u64) {
        let now = self.clamp_time(now);
        self.now_ns = now;
        self.uart.set_time(now);
        let byte = value as u8;
        match port {
            PIC_MASTER_COMMAND | PIC_MASTER_DATA | PIC_SLAVE_COMMAND | PIC_SLAVE_DATA
            | PIC_ELCR_MASTER | PIC_ELCR_SLAVE => self.pic.io_write(port, byte),
            PIT_CHANNEL0..=PIT_COMMAND => self.pit.write(port, byte, now),
            SYSTEM_CONTROL_PORT_B => {
                self.system_control_port_b = byte & PORT_B_WRITABLE_MASK;
                self.pit.set_gate2(byte & PORT_B_TIMER2_GATE != 0, now);
            }
            CMOS_INDEX_PORT => self.cmos_index = byte,
            CMOS_DATA_PORT => self.cmos[(self.cmos_index & CMOS_INDEX_MASK) as usize] = byte,
            KBC_COMMAND_PORT if byte == KBC_COMMAND_PULSE_RESET => self.reset_requested = true,
            FAST_A20_PORT => {
                self.system_control_port_a = byte & FAST_A20_GATE;
                if byte & FAST_A20_RESET != 0 {
                    self.reset_requested = true;
                }
            }
            RESET_CONTROL_PORT => {
                if byte & RESET_CONTROL_RESET_CPU != 0 {
                    self.reset_requested = true;
                } else {
                    self.reset_control = byte & RESET_CONTROL_SYSTEM_RESET;
                }
            }
            PCI_CONFIG_PORT_FIRST..=PCI_CONFIG_PORT_LAST if !self.pci.is_empty() => {
                self.pci.io_out(port, size, value)
            }
            port if AcpiPm::decodes(port) => {
                self.pm.write(port, size, value, now);
                self.update_sci(now);
            }
            KV_PORT_FIRST..=KV_PORT_LAST if self.kv.is_some() => {
                self.kv.as_mut().unwrap().io_out(port, size, value)
            }
            UART_COM1_BASE..=UART_COM1_LAST => {
                self.uart.write(port - UART_COM1_BASE, byte);
                self.update_uart_irq();
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod smp_tests;
#[cfg(test)]
mod tsc_calibration_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::guestmem::test_mem;
    use crate::dev::pci::{CONFIG_ADDRESS_PORT, CONFIG_DATA_PORT, MMIO_BASE};
    use crate::netdev::NetHandle;

    const NOW_ZERO: u64 = 0;
    const ONE_SECOND_NS: u64 = 1_000_000_000;
    const RAM_BYTES: usize = 0x2000;
    /// The BAR of the only function is 32 KiB: common, ISR, device, notify, MSI-X table and PBA.
    const FIRST_BYTE_PAST_THE_BAR: u64 = 0x8000;
    const PCI_ENABLE: u32 = 1 << 31;
    const SLOT_SHIFT: u32 = 11;
    const COMMAND_REGISTER: u32 = 0x04;
    const MEMORY_SPACE_ENABLE: u32 = 1 << 1;
    const VIRTIO_VENDOR: u32 = 0x1af4;
    const COMMON_DEVICE_STATUS_OFFSET: u64 = 0x14;
    const STATUS_DRIVER_OK: u32 = 4;
    const UART_SCRATCH_PORT: u16 = 0x3ff;
    const UART_DATA_PORT: u16 = 0x3f8;
    const UART_IER_PORT: u16 = 0x3f9;
    const UART_LSR_PORT: u16 = 0x3fd;
    const LSR_THR_EMPTY: u32 = 0x20;
    const IER_RX_AVAILABLE: u32 = 1;
    const PIT_CONTROL_PORT: u16 = 0x43;
    const PIT_CHANNEL0_PORT: u16 = 0x40;
    const PIT_CHANNEL2_PORT: u16 = 0x42;
    const PIT_CH0_MODE2_LOHI: u32 = 0x34;
    const PIT_CH2_MODE0_LOHI: u32 = 0xb0;
    const PIC_VECTOR_BASE_MASTER: u32 = 0x20;
    const PIC_VECTOR_BASE_SLAVE: u32 = 0x28;

    fn bus() -> Devices {
        Devices::new(ConsoleHandle::default(), DevConfig::default())
    }

    fn bus_with_console(console: &ConsoleHandle) -> Devices {
        Devices::new(
            console.clone(),
            DevConfig {
                virtio_consoles: vec![console.clone()],
                ..Default::default()
            },
        )
    }

    fn bus_with_pci() -> Devices {
        bus_with_console(&ConsoleHandle::default())
    }

    fn out(dev: &mut Devices, port: u16, size: u8, value: u32) {
        dev.io_out(port, size, value, NOW_ZERO);
    }

    fn inp(dev: &mut Devices, port: u16, size: u8) -> u32 {
        dev.io_in(port, size, NOW_ZERO)
    }

    fn cmos_read(dev: &mut Devices, index: u8, now: u64) -> u8 {
        dev.io_out(CMOS_INDEX_PORT, 1, index as u32, now);
        dev.io_in(CMOS_DATA_PORT, 1, now) as u8
    }

    fn cmos_write(dev: &mut Devices, index: u8, value: u8) {
        out(dev, CMOS_INDEX_PORT, 1, index as u32);
        out(dev, CMOS_DATA_PORT, 1, value as u32);
    }

    fn pci_address(slot: u32, reg: u32) -> u32 {
        PCI_ENABLE | slot << SLOT_SHIFT | reg
    }

    /// Programs both PICs like the kernel does so vectors are 0x20.. and 0x28..
    fn init_pics(dev: &mut Devices) {
        for (command, data, base, cascade) in [
            (
                PIC_MASTER_COMMAND,
                PIC_MASTER_DATA,
                PIC_VECTOR_BASE_MASTER,
                0x04,
            ),
            (
                PIC_SLAVE_COMMAND,
                PIC_SLAVE_DATA,
                PIC_VECTOR_BASE_SLAVE,
                0x02,
            ),
        ] {
            out(dev, command, 1, 0x11);
            out(dev, data, 1, base);
            out(dev, data, 1, cascade);
            out(dev, data, 1, 0x01);
        }
    }

    fn enable_console_driver(dev: &mut Devices) {
        out(
            dev,
            PCI_CONFIG_PORT_FIRST,
            4,
            pci_address(1, COMMAND_REGISTER),
        );
        out(dev, 0xcfc, 2, MEMORY_SPACE_ENABLE);
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert!(dev.mmio_write(
            MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET,
            1,
            STATUS_DRIVER_OK as u64,
            &mut mem
        ));
    }

    fn tick(dev: &mut Devices, now: u64) {
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        dev.tick(now, &mut mem);
    }

    fn bcd_of(value: u32) -> u8 {
        (((value / 10) << 4) | (value % 10)) as u8
    }

    #[test]
    fn size_mask_covers_the_low_bytes_and_is_total() {
        assert_eq!(size_mask(0), 0);
        assert_eq!(size_mask(1), 0xff);
        assert_eq!(size_mask(2), 0xffff);
        assert_eq!(size_mask(3), 0x00ff_ffff);
        assert_eq!(size_mask(4), u32::MAX);
        assert_eq!(size_mask(8), u32::MAX);
        assert_eq!(size_mask(u8::MAX), u32::MAX);
    }

    #[test]
    fn bcd_encodes_two_decimal_digits() {
        assert_eq!(bcd(0), 0x00);
        assert_eq!(bcd(9), 0x09);
        assert_eq!(bcd(10), 0x10);
        assert_eq!(bcd(59), 0x59);
        assert_eq!(bcd(99), 0x99);
    }

    #[test]
    fn civil_converts_days_since_the_epoch_to_dates() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(-1), (1969, 12, 31));
        assert_eq!(civil(59), (1970, 3, 1));
        assert_eq!(civil(11016), (2000, 2, 29));
        assert_eq!(civil(11017), (2000, 3, 1));
        assert_eq!(civil(47541), (2100, 3, 1), "2100 is not a leap year");
        assert_eq!(civil(47540), (2100, 2, 28));
        assert_eq!(civil(-719468), (0, 3, 1), "the algorithm's own epoch");
    }

    #[test]
    fn unknown_ports_read_all_ones_at_every_size_and_ignore_writes() {
        let mut dev = bus();
        for port in [
            0x0, 0x1f, 0x60, 0x80, 0x1234, 0x3f0, 0x400, 0xcf7, 0xd00, 0xffff,
        ] {
            assert_eq!(inp(&mut dev, port, 1), 0xff, "port {port:#x}");
            assert_eq!(inp(&mut dev, port, 2), 0xffff, "port {port:#x}");
            assert_eq!(inp(&mut dev, port, 4), u32::MAX, "port {port:#x}");
            out(&mut dev, port, 4, u32::MAX);
        }
        assert!(!dev.reset_requested);
    }

    #[test]
    fn io_in_of_size_zero_is_total_and_returns_nothing() {
        let mut dev = bus_with_pci();
        assert_eq!(inp(&mut dev, 0x1234, 0), 0);
        assert_eq!(inp(&mut dev, CMOS_DATA_PORT, 0), 0);
        assert_eq!(inp(&mut dev, 0xcfc, 0), 0);
    }

    #[test]
    fn cmos_index_port_is_write_only() {
        let mut dev = bus();
        out(&mut dev, CMOS_INDEX_PORT, 1, 0x0a);
        assert_eq!(inp(&mut dev, CMOS_INDEX_PORT, 1), 0xff);
    }

    #[test]
    fn cmos_status_registers_have_fixed_values() {
        let mut dev = bus();
        assert_eq!(cmos_read(&mut dev, CMOS_STATUS_A as u8, 0), 0x26);
        assert_eq!(cmos_read(&mut dev, CMOS_STATUS_B as u8, 0), 0x02);
        assert_eq!(cmos_read(&mut dev, CMOS_STATUS_C as u8, 0), 0x00);
        assert_eq!(cmos_read(&mut dev, CMOS_STATUS_D as u8, 0), 0x80);
    }

    #[test]
    fn cmos_status_a_never_reports_update_in_progress() {
        let mut dev = bus();
        for now in [
            0,
            1,
            ONE_SECOND_NS - 1,
            ONE_SECOND_NS,
            123 * ONE_SECOND_NS + 7,
        ] {
            assert_eq!(cmos_read(&mut dev, CMOS_STATUS_A as u8, now) & 0x80, 0);
        }
    }

    #[test]
    fn cmos_status_b_stays_bcd_and_24_hour_whatever_the_guest_writes() {
        let mut dev = bus();
        dev.set_boot_time(1_709_210_096);
        cmos_write(&mut dev, CMOS_STATUS_B as u8, 0x06);
        let status_b = cmos_read(&mut dev, CMOS_STATUS_B as u8, 0);
        assert_eq!(status_b & 0x04, 0, "binary mode bit stays clear");
        assert_ne!(status_b & 0x02, 0, "24-hour mode bit stays set");
        assert_eq!(
            cmos_read(&mut dev, CMOS_HOURS as u8, 0),
            0x12,
            "hours are still BCD"
        );
    }

    #[test]
    fn cmos_status_c_and_d_ignore_writes() {
        let mut dev = bus();
        cmos_write(&mut dev, CMOS_STATUS_C as u8, 0xff);
        cmos_write(&mut dev, CMOS_STATUS_D as u8, 0x00);
        assert_eq!(cmos_read(&mut dev, CMOS_STATUS_C as u8, 0), 0);
        assert_eq!(cmos_read(&mut dev, CMOS_STATUS_D as u8, 0), 0x80);
    }

    #[test]
    fn cmos_nmi_mask_bit_of_the_index_is_not_part_of_the_index() {
        let mut dev = bus();
        assert_eq!(cmos_read(&mut dev, 0x80 | CMOS_STATUS_A as u8, 0), 0x26);
        cmos_write(&mut dev, 0x80 | 0x20, 0xab);
        assert_eq!(cmos_read(&mut dev, 0x20, 0), 0xab);
        assert_eq!(cmos_read(&mut dev, 0xa0, 0), 0xab);
    }

    #[test]
    fn cmos_ram_registers_are_independent_and_start_zero() {
        let mut dev = bus();
        let ram_indices: Vec<u8> = (0x0eu8..0x80)
            .filter(|&index| index as usize != CMOS_CENTURY)
            .collect();
        for &index in &ram_indices {
            assert_eq!(cmos_read(&mut dev, index, 0), 0, "index {index:#x}");
        }
        for &index in &ram_indices {
            cmos_write(&mut dev, index, index ^ 0x5a);
        }
        for &index in &ram_indices {
            assert_eq!(
                cmos_read(&mut dev, index, 0),
                index ^ 0x5a,
                "index {index:#x}"
            );
        }
    }

    #[test]
    fn cmos_alarm_registers_are_plain_ram() {
        let mut dev = bus();
        for (index, value) in [(0x01u8, 0x11u8), (0x03, 0x22), (0x05, 0x33)] {
            cmos_write(&mut dev, index, value);
            assert_eq!(cmos_read(&mut dev, index, 0), value);
        }
    }

    #[test]
    fn cmos_time_registers_ignore_writes() {
        let mut dev = bus();
        dev.set_boot_time(0);
        cmos_write(&mut dev, CMOS_SECONDS as u8, 0x45);
        cmos_write(&mut dev, CMOS_YEAR as u8, 0x99);
        cmos_write(&mut dev, CMOS_CENTURY as u8, 0x30);
        assert_eq!(cmos_read(&mut dev, CMOS_SECONDS as u8, 0), 0x00);
        assert_eq!(cmos_read(&mut dev, CMOS_YEAR as u8, 0), 0x70);
        assert_eq!(cmos_read(&mut dev, CMOS_CENTURY as u8, 0), 0x19);
    }

    #[test]
    fn cmos_data_read_is_masked_to_the_access_size() {
        let mut dev = bus();
        dev.set_boot_time(1_709_210_096);
        out(&mut dev, CMOS_INDEX_PORT, 1, CMOS_MINUTES as u32);
        assert_eq!(inp(&mut dev, CMOS_DATA_PORT, 4), 0x34);
        assert_eq!(inp(&mut dev, CMOS_DATA_PORT, 2), 0x34);
    }

    struct DateCase {
        unix: u64,
        year: u32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
        /// Sunday is 1, as the RTC counts.
        weekday: u8,
    }

    /// Expected values come from an independent calendar (Python's datetime), not from this code.
    const DATE_CASES: [DateCase; 17] = [
        DateCase {
            unix: 0,
            year: 1970,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 5,
        },
        DateCase {
            unix: 946_684_799,
            year: 1999,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            weekday: 6,
        },
        DateCase {
            unix: 946_684_800,
            year: 2000,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 7,
        },
        DateCase {
            unix: 951_696_000,
            year: 2000,
            month: 2,
            day: 28,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 2,
        },
        DateCase {
            unix: 951_782_400,
            year: 2000,
            month: 2,
            day: 29,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 3,
        },
        DateCase {
            unix: 951_868_800,
            year: 2000,
            month: 3,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 4,
        },
        DateCase {
            unix: 1_675_123_200,
            year: 2023,
            month: 1,
            day: 31,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 3,
        },
        DateCase {
            unix: 1_675_209_600,
            year: 2023,
            month: 2,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 4,
        },
        DateCase {
            unix: 1_682_812_800,
            year: 2023,
            month: 4,
            day: 30,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 1,
        },
        DateCase {
            unix: 1_704_067_199,
            year: 2023,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            weekday: 1,
        },
        DateCase {
            unix: 1_709_210_096,
            year: 2024,
            month: 2,
            day: 29,
            hour: 12,
            minute: 34,
            second: 56,
            weekday: 5,
        },
        DateCase {
            unix: 2_147_483_647,
            year: 2038,
            month: 1,
            day: 19,
            hour: 3,
            minute: 14,
            second: 7,
            weekday: 3,
        },
        DateCase {
            unix: 2_147_483_648,
            year: 2038,
            month: 1,
            day: 19,
            hour: 3,
            minute: 14,
            second: 8,
            weekday: 3,
        },
        DateCase {
            unix: 4_107_456_000,
            year: 2100,
            month: 2,
            day: 28,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 1,
        },
        DateCase {
            unix: 4_107_542_400,
            year: 2100,
            month: 3,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 2,
        },
        DateCase {
            unix: 13_574_563_200,
            year: 2400,
            month: 2,
            day: 29,
            hour: 0,
            minute: 0,
            second: 0,
            weekday: 3,
        },
        DateCase {
            unix: 253_402_300_799,
            year: 9999,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 59,
            weekday: 6,
        },
    ];

    #[test]
    fn set_boot_time_converts_tricky_dates_to_every_rtc_register() {
        for case in &DATE_CASES {
            let mut dev = bus();
            dev.set_boot_time(case.unix);
            let read = |dev: &mut Devices, index: usize| cmos_read(dev, index as u8, 0);
            let label = format!("unix {}", case.unix);
            assert_eq!(
                read(&mut dev, CMOS_SECONDS),
                bcd_of(case.second),
                "{label} seconds"
            );
            assert_eq!(
                read(&mut dev, CMOS_MINUTES),
                bcd_of(case.minute),
                "{label} minutes"
            );
            assert_eq!(
                read(&mut dev, CMOS_HOURS),
                bcd_of(case.hour),
                "{label} hours"
            );
            assert_eq!(
                read(&mut dev, CMOS_DAY_OF_WEEK),
                case.weekday,
                "{label} weekday"
            );
            assert_eq!(
                read(&mut dev, CMOS_DAY_OF_MONTH),
                bcd_of(case.day),
                "{label} day"
            );
            assert_eq!(
                read(&mut dev, CMOS_MONTH),
                bcd_of(case.month),
                "{label} month"
            );
            assert_eq!(
                read(&mut dev, CMOS_YEAR),
                bcd_of(case.year % 100),
                "{label} year"
            );
            assert_eq!(
                read(&mut dev, CMOS_CENTURY),
                bcd_of(case.year / 100),
                "{label} century"
            );
        }
    }

    #[test]
    fn rtc_follows_virtual_time_not_the_host_clock() {
        let mut dev = bus();
        dev.set_boot_time(1_000_000_000);
        let seconds = |dev: &mut Devices, now| cmos_read(dev, CMOS_SECONDS as u8, now);
        let base = seconds(&mut dev, 0);
        assert_eq!(base, 0x40, "1_000_000_000 is 01:46:40 UTC");
        assert_eq!(
            seconds(&mut dev, ONE_SECOND_NS - 1),
            0x40,
            "sub-second time does not tick"
        );
        assert_eq!(seconds(&mut dev, ONE_SECOND_NS), 0x41);
        assert_eq!(seconds(&mut dev, 20 * ONE_SECOND_NS), 0x00);
        assert_eq!(
            cmos_read(&mut dev, CMOS_MINUTES as u8, 20 * ONE_SECOND_NS),
            0x47
        );
    }

    #[test]
    fn rtc_rolls_over_midnight_month_year_and_leap_day() {
        let mut dev = bus();
        // 2023-12-31 23:59:59 plus one second.
        dev.set_boot_time(1_704_067_199);
        let at = |dev: &mut Devices, index: usize| cmos_read(dev, index as u8, ONE_SECOND_NS);
        assert_eq!(at(&mut dev, CMOS_SECONDS), 0x00);
        assert_eq!(at(&mut dev, CMOS_MINUTES), 0x00);
        assert_eq!(at(&mut dev, CMOS_HOURS), 0x00);
        assert_eq!(at(&mut dev, CMOS_DAY_OF_MONTH), 0x01);
        assert_eq!(at(&mut dev, CMOS_MONTH), 0x01);
        assert_eq!(at(&mut dev, CMOS_YEAR), 0x24);
        assert_eq!(at(&mut dev, CMOS_DAY_OF_WEEK), 2, "Monday");
        // 2000-02-28 23:59:59 plus one second is the leap day.
        dev.set_boot_time(951_782_399);
        assert_eq!(at(&mut dev, CMOS_DAY_OF_MONTH), 0x29);
        assert_eq!(at(&mut dev, CMOS_MONTH), 0x02);
    }

    #[test]
    fn rtc_with_extreme_boot_time_and_clock_never_panics() {
        let mut dev = bus();
        dev.set_boot_time(u64::MAX);
        for index in 0..CMOS_SIZE as u8 {
            let _ = cmos_read(&mut dev, index, u64::MAX);
        }
    }

    #[test]
    fn port_b_returns_only_the_two_writable_bits_plus_synthesized_ones() {
        let mut dev = bus();
        out(&mut dev, SYSTEM_CONTROL_PORT_B, 1, 0xff);
        let value = dev.io_in(SYSTEM_CONTROL_PORT_B, 1, 0);
        assert_eq!(value & 0x03, 0x03);
        assert_eq!(
            value & !0x03 & !(PORT_B_REFRESH_TOGGLE as u32) & !(PORT_B_TIMER2_OUT as u32),
            0
        );
        out(&mut dev, SYSTEM_CONTROL_PORT_B, 1, 0x00);
        assert_eq!(dev.io_in(SYSTEM_CONTROL_PORT_B, 1, 0) & 0x0f, 0);
        out(&mut dev, SYSTEM_CONTROL_PORT_B, 1, 0x02);
        assert_eq!(dev.io_in(SYSTEM_CONTROL_PORT_B, 1, 0) & 0x0f, 0x02);
    }

    #[test]
    fn port_b_refresh_bit_toggles_every_period_of_virtual_time() {
        let mut dev = bus();
        let refresh = |dev: &mut Devices, now: u64| {
            dev.io_in(SYSTEM_CONTROL_PORT_B, 1, now) & PORT_B_REFRESH_TOGGLE as u32 != 0
        };
        let period = REFRESH_TOGGLE_PERIOD_NS;
        assert!(!refresh(&mut dev, 0));
        assert!(!refresh(&mut dev, period - 1));
        assert!(refresh(&mut dev, period));
        assert!(refresh(&mut dev, 2 * period - 1));
        assert!(!refresh(&mut dev, 2 * period));
        assert!(refresh(&mut dev, 3 * period));
        assert!(!refresh(&mut dev, 4 * period));
    }

    #[test]
    fn port_b_timer2_output_reflects_the_pit_channel() {
        let mut dev = bus();
        out(
            &mut dev,
            SYSTEM_CONTROL_PORT_B,
            1,
            PORT_B_TIMER2_GATE as u32,
        );
        out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH2_MODE0_LOHI);
        out(&mut dev, PIT_CHANNEL2_PORT, 1, 0x10);
        out(&mut dev, PIT_CHANNEL2_PORT, 1, 0x00);
        let out2 = |dev: &mut Devices, now: u64| {
            dev.io_in(SYSTEM_CONTROL_PORT_B, 1, now) & PORT_B_TIMER2_OUT as u32 != 0
        };
        assert!(!out2(&mut dev, 1000), "16 PIT ticks have not elapsed");
        assert!(
            out2(&mut dev, ONE_SECOND_NS / 1000),
            "1 ms is far past 16 ticks"
        );
    }

    #[test]
    fn fast_a20_gate_bit_is_stored_and_other_bits_dropped() {
        let mut dev = bus();
        assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0);
        out(&mut dev, FAST_A20_PORT, 1, 0x02);
        assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0x02);
        assert!(!dev.reset_requested);
        out(&mut dev, FAST_A20_PORT, 1, 0xfc);
        assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0);
        assert!(!dev.reset_requested);
    }

    #[test]
    fn fast_a20_bit_zero_requests_a_reset() {
        let mut dev = bus();
        out(&mut dev, FAST_A20_PORT, 1, 0x01);
        assert!(dev.reset_requested);
    }

    #[test]
    fn fast_a20_reset_with_gate_set_keeps_the_gate() {
        let mut dev = bus();
        out(&mut dev, FAST_A20_PORT, 1, 0x03);
        assert!(dev.reset_requested);
        assert_eq!(inp(&mut dev, FAST_A20_PORT, 1), 0x02);
    }

    #[test]
    fn keyboard_controller_pulse_reset_command_requests_a_reset() {
        let mut dev = bus();
        out(&mut dev, KBC_COMMAND_PORT, 1, 0xfe);
        assert!(dev.reset_requested);
    }

    #[test]
    fn other_keyboard_controller_commands_do_not_reset() {
        let mut dev = bus();
        for command in [0x00, 0x20, 0x60, 0xaa, 0xad, 0xae, 0xd1, 0xfc, 0xfd, 0xff] {
            out(&mut dev, KBC_COMMAND_PORT, 1, command);
        }
        assert!(!dev.reset_requested);
    }

    #[test]
    fn keyboard_data_port_and_status_port_are_not_modelled() {
        let mut dev = bus();
        out(&mut dev, 0x60, 1, 0xfe);
        assert!(!dev.reset_requested);
        assert_eq!(inp(&mut dev, 0x60, 1), 0xff);
        assert_eq!(
            inp(&mut dev, KBC_COMMAND_PORT, 1),
            0xff,
            "status reads as no controller"
        );
    }

    #[test]
    fn keyboard_reset_uses_the_low_byte_of_wider_writes() {
        let mut dev = bus();
        out(&mut dev, KBC_COMMAND_PORT, 4, 0xffff_ff00);
        assert!(!dev.reset_requested);
        out(&mut dev, KBC_COMMAND_PORT, 2, 0x1200 | 0xfe);
        assert!(dev.reset_requested);
    }

    #[test]
    fn reset_control_without_the_cpu_reset_bit_does_not_reset() {
        let mut dev = bus();
        for value in [0x00, 0x01, 0x02, 0x03, 0xfb, 0xf3] {
            out(&mut dev, RESET_CONTROL_PORT, 1, value);
        }
        assert!(!dev.reset_requested);
    }

    #[test]
    fn reset_control_cpu_reset_bit_requests_a_reset_for_hard_and_soft_variants() {
        for value in [0x04, 0x06, 0x0e, 0x05, 0xff] {
            let mut dev = bus();
            out(&mut dev, RESET_CONTROL_PORT, 1, value);
            assert!(dev.reset_requested, "value {value:#x}");
        }
    }

    #[test]
    fn linux_reboot_sequence_resets_on_the_second_write() {
        let mut dev = bus_with_pci();
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
        assert!(!dev.reset_requested);
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x06);
        assert!(dev.reset_requested);
    }

    #[test]
    fn reset_control_reads_back_only_the_system_reset_type_bit() {
        let mut dev = bus();
        assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0);
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
        assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0x02);
        out(&mut dev, RESET_CONTROL_PORT, 1, 0xfa);
        assert_eq!(
            inp(&mut dev, RESET_CONTROL_PORT, 1),
            0x02,
            "bit 3 and above are not kept"
        );
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x00);
        assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0);
    }

    #[test]
    fn reset_control_works_with_and_without_pci_devices() {
        for with_pci in [false, true] {
            let mut dev = if with_pci { bus_with_pci() } else { bus() };
            out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
            assert_eq!(inp(&mut dev, RESET_CONTROL_PORT, 1), 0x02, "pci {with_pci}");
            out(&mut dev, RESET_CONTROL_PORT, 1, 0x04);
            assert!(dev.reset_requested, "pci {with_pci}");
        }
    }

    #[test]
    fn reset_control_write_does_not_touch_the_pci_address_register() {
        let mut dev = bus_with_pci();
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
        assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), pci_address(1, 0));
        assert_eq!(inp(&mut dev, 0xcfc, 2), VIRTIO_VENDOR);
    }

    #[test]
    fn reset_control_write_is_not_a_config_data_write() {
        let mut dev = bus_with_pci();
        out(
            &mut dev,
            PCI_CONFIG_PORT_FIRST,
            4,
            pci_address(1, COMMAND_REGISTER),
        );
        out(&mut dev, RESET_CONTROL_PORT, 1, 0x02);
        assert_eq!(
            inp(&mut dev, 0xcfc, 2),
            0,
            "the command register must not see byte 0x02"
        );
        assert_eq!(inp(&mut dev, 0xcfd, 1), 0);
    }

    #[test]
    fn dword_write_to_the_pci_address_port_does_not_reset_even_if_the_cf9_lane_has_bit_two() {
        let mut dev = bus_with_pci();
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, 0x8000_0400);
        assert!(!dev.reset_requested);
        assert_eq!(
            inp(&mut dev, RESET_CONTROL_PORT, 1),
            0,
            "reset control register unchanged"
        );
        assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), 0x8000_0400);
    }

    #[test]
    fn the_entropy_device_appears_on_the_pci_bus_as_virtio_device_1044() {
        const PCI_ID_REGISTER: u32 = 0;
        const ENTROPY_PCI_DEVICE_ID: u32 = 0x1044;
        let mut dev = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                rng: Some(RngSource::Seeded(1)),
                ..Default::default()
            },
        );
        out(
            &mut dev,
            PCI_CONFIG_PORT_FIRST,
            4,
            PCI_ENABLE | 1 << SLOT_SHIFT | PCI_ID_REGISTER,
        );
        let id = inp(&mut dev, PCI_CONFIG_PORT_FIRST + 4, 4);
        assert_eq!(id, ENTROPY_PCI_DEVICE_ID << 16 | VIRTIO_VENDOR);
    }

    #[test]
    fn pci_address_register_accepts_byte_word_and_dword_accesses_through_the_bus() {
        let mut dev = bus_with_pci();
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, 0x8000_0000);
        out(&mut dev, PCI_CONFIG_PORT_FIRST + 2, 1, 0x01);
        out(&mut dev, PCI_CONFIG_PORT_FIRST + 3, 1, 0x80);
        assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), 0x8001_0000);
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 2, 0x0800);
        assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4), 0x8001_0800);
        assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST + 2, 2), 0x8001);
        assert_eq!(inp(&mut dev, PCI_CONFIG_PORT_FIRST + 3, 1), 0x80);
        assert_eq!(inp(&mut dev, 0xcfa, 1), 0x01);
    }

    #[test]
    fn pci_config_space_is_reachable_through_the_ports() {
        let mut dev = bus_with_pci();
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(0, 0));
        assert_eq!(inp(&mut dev, 0xcfc, 4), 0x1237_8086);
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
        assert_eq!(inp(&mut dev, CONFIG_DATA_PORT, 2), VIRTIO_VENDOR);
        assert_eq!(inp(&mut dev, CONFIG_DATA_PORT + 2, 2), 0x1043);
        assert_eq!(inp(&mut dev, CONFIG_DATA_PORT + 3, 1), 0x10);
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(2, 0));
        assert_eq!(inp(&mut dev, 0xcfc, 4), u32::MAX);
    }

    #[test]
    fn pci_ports_are_dead_when_no_virtio_device_is_attached() {
        let mut dev = bus();
        out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(0, 0));
        assert_eq!(
            inp(&mut dev, PCI_CONFIG_PORT_FIRST, 4),
            u32::MAX,
            "address register does not echo"
        );
        assert_eq!(inp(&mut dev, 0xcfc, 4), u32::MAX);
        assert_eq!(inp(&mut dev, 0xcfe, 2), 0xffff);
    }

    #[test]
    fn pci_windows_do_not_extend_past_cff_or_below_cf8() {
        let mut dev = bus_with_pci();
        assert_eq!(
            inp(&mut dev, PCI_CONFIG_PORT_LAST, 1),
            0xff,
            "cff in the data window, no device selected"
        );
        assert_eq!(inp(&mut dev, 0xd00, 1), 0xff);
        assert_eq!(inp(&mut dev, 0xcf7, 1), 0xff);
        assert_eq!(CONFIG_ADDRESS_PORT, PCI_CONFIG_PORT_FIRST);
    }

    #[test]
    fn devices_attach_in_console_disk_net_order_to_consecutive_slots() {
        let mut dev = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                disks: vec![Disk::memory(vec![0; 4096])],
                nets: vec![([2, 0, 0, 0, 0, 1], NetHandle::default())],
                virtio_consoles: vec![ConsoleHandle::default()],
                ..Default::default()
            },
        );
        let mut device_ids = Vec::new();
        for slot in 1..=4 {
            out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(slot, 0));
            device_ids.push(inp(&mut dev, 0xcfc, 4) >> 16);
        }
        assert_eq!(device_ids, vec![0x1043, 0x1042, 0x1041, 0xffff]);
    }

    #[test]
    fn a_lone_disk_or_net_device_takes_slot_one() {
        let mut disk_only = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                disks: vec![Disk::memory(vec![0; 4096])],
                ..Default::default()
            },
        );
        out(&mut disk_only, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
        assert_eq!(inp(&mut disk_only, 0xcfc, 4) >> 16, 0x1042);
        let mut net_only = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                nets: vec![([2, 0, 0, 0, 0, 1], NetHandle::default())],
                ..Default::default()
            },
        );
        out(&mut net_only, PCI_CONFIG_PORT_FIRST, 4, pci_address(1, 0));
        assert_eq!(inp(&mut net_only, 0xcfc, 4) >> 16, 0x1041);
    }

    #[test]
    fn mmio_with_no_devices_is_unmapped() {
        let mut dev = bus();
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert_eq!(dev.mmio_read(MMIO_BASE, 4), None);
        assert_eq!(dev.mmio_read(0, 1), None);
        assert!(!dev.mmio_write(MMIO_BASE, 4, 0, &mut mem));
        assert!(!dev.mmio_write(u64::MAX, 1, 0, &mut mem));
    }

    #[test]
    fn mmio_is_unmapped_until_the_guest_enables_memory_space() {
        let mut dev = bus_with_pci();
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert_eq!(dev.mmio_read(MMIO_BASE + 0x12, 2), None);
        assert!(!dev.mmio_write(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1, 4, &mut mem));
        out(
            &mut dev,
            PCI_CONFIG_PORT_FIRST,
            4,
            pci_address(1, COMMAND_REGISTER),
        );
        out(&mut dev, 0xcfc, 2, MEMORY_SPACE_ENABLE);
        assert_eq!(dev.mmio_read(MMIO_BASE + 0x12, 2), Some(2));
        assert!(dev.mmio_write(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1, 4, &mut mem));
        assert_eq!(
            dev.mmio_read(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1),
            Some(4)
        );
    }

    #[test]
    fn mmio_outside_every_bar_stays_unmapped_after_enabling() {
        let mut dev = bus_with_pci();
        enable_console_driver(&mut dev);
        let mut ram = vec![0u8; RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert_eq!(dev.mmio_read(MMIO_BASE - 1, 1), None);
        assert_eq!(dev.mmio_read(MMIO_BASE + FIRST_BYTE_PAST_THE_BAR, 1), None);
        assert_eq!(dev.mmio_read(0xffff_f000, 1), None);
        assert!(!dev.mmio_write(MMIO_BASE + FIRST_BYTE_PAST_THE_BAR, 4, 0, &mut mem));
    }

    #[test]
    fn mmio_write_that_raises_nothing_leaves_the_pic_quiet() {
        let mut dev = bus_with_pci();
        init_pics(&mut dev);
        enable_console_driver(&mut dev);
        assert!(!dev.pic.has_pending());
        assert_eq!(dev.pic.ack(), None);
    }

    /// Console transmit queue programmed through BAR0, one chain posted, ready to kick.
    fn console_with_transmit_chain(dev: &mut Devices, ram: &mut [u8]) {
        const TRANSMIT_QUEUE: u32 = 1;
        const DESC: u64 = 0x100;
        const AVAIL: u64 = 0x200;
        const USED: u64 = 0x300;
        const BUFFER: u64 = 0x400;
        ram[BUFFER as usize] = b'!';
        ram[DESC as usize..DESC as usize + 8].copy_from_slice(&BUFFER.to_le_bytes());
        ram[DESC as usize + 8..DESC as usize + 12].copy_from_slice(&1u32.to_le_bytes());
        ram[AVAIL as usize + 2] = 1;
        out(
            dev,
            PCI_CONFIG_PORT_FIRST,
            4,
            pci_address(1, COMMAND_REGISTER),
        );
        out(dev, 0xcfc, 2, MEMORY_SPACE_ENABLE);
        let mut mem = test_mem(ram);
        let mut write = |offset: u64, size: u8, value: u64| {
            assert!(dev.mmio_write(MMIO_BASE + offset, size, value, &mut mem));
        };
        write(0x16, 2, TRANSMIT_QUEUE as u64);
        write(0x18, 2, 8);
        write(0x20, 4, DESC);
        write(0x28, 4, AVAIL);
        write(0x30, 4, USED);
        write(0x1c, 2, 1);
        write(COMMON_DEVICE_STATUS_OFFSET, 1, STATUS_DRIVER_OK as u64);
        write(0x3000 + 4 * TRANSMIT_QUEUE as u64, 2, TRANSMIT_QUEUE as u64);
    }

    #[test]
    fn mmio_write_that_completes_a_buffer_raises_the_function_irq_on_the_pic() {
        let console = ConsoleHandle::default();
        let mut dev = bus_with_console(&console);
        init_pics(&mut dev);
        let mut ram = vec![0u8; RAM_BYTES];
        console_with_transmit_chain(&mut dev, &mut ram);
        let mut buf = [0u8; 4];
        assert_eq!(console.read(&mut buf, None), 1);
        assert_eq!(buf[0], b'!');
        assert!(dev.pic.has_pending());
        assert_eq!(
            dev.pic.ack(),
            Some(PIC_VECTOR_BASE_SLAVE as u8 + 1),
            "IRQ 9 is slave line 1"
        );
    }

    #[test]
    fn tick_polls_virtio_devices_and_raises_their_irq() {
        let console = ConsoleHandle::default();
        let mut dev = bus_with_console(&console);
        init_pics(&mut dev);
        enable_console_driver(&mut dev);
        tick(&mut dev, 0);
        assert!(!dev.pic.has_pending());
        console.resize(120, 40);
        tick(&mut dev, 1);
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_SLAVE as u8 + 1));
    }

    #[test]
    fn tick_raises_irq0_when_the_pit_deadline_passes() {
        let mut dev = bus();
        init_pics(&mut dev);
        out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0xa9);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x04);
        let deadline = dev.next_deadline(0).expect("channel 0 is armed");
        tick(&mut dev, deadline - 1);
        assert!(!dev.pic.has_pending(), "not due yet");
        tick(&mut dev, deadline);
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8));
    }

    #[test]
    fn next_deadline_is_the_pit_deadline_and_ignores_the_now_argument() {
        let mut dev = bus();
        assert_eq!(dev.next_deadline(0), None);
        assert_eq!(dev.next_deadline(u64::MAX), None);
        out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0xa9);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x04);
        let deadline = dev.next_deadline(0).unwrap();
        assert_eq!(dev.next_deadline(deadline / 2), Some(deadline));
        assert_eq!(
            dev.next_deadline(deadline * 10),
            Some(deadline),
            "a past deadline is still reported"
        );
        assert_eq!(dev.next_deadline(u64::MAX), Some(deadline));
    }

    #[test]
    fn uart_and_virtio_devices_do_not_schedule_deadlines() {
        let console = ConsoleHandle::default();
        let mut dev = bus_with_console(&console);
        console.write(b"pending input");
        console.resize(100, 50);
        out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
        assert_eq!(dev.next_deadline(0), None);
    }

    #[test]
    fn state_dump_combines_pic_and_pit_state_with_the_given_time() {
        let mut dev = bus();
        let first = dev.state(111);
        assert!(first.starts_with("m(irr "), "{first}");
        assert!(first.contains(" | pit0 mode "), "{first}");
        assert!(first.ends_with("now 111"), "{first}");
        assert!(dev.state(222).ends_with("now 222"));
        out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x00);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x10);
        assert!(dev.state(0).contains("reload 4096"), "{}", dev.state(0));
    }

    #[test]
    fn pic_ports_are_routed_to_the_interrupt_controllers() {
        let mut dev = bus();
        for (port, mask) in [(PIC_MASTER_DATA, 0xa5), (PIC_SLAVE_DATA, 0x5a)] {
            out(&mut dev, port, 1, mask);
            assert_eq!(inp(&mut dev, port, 1), mask, "port {port:#x}");
        }
        out(&mut dev, PIC_MASTER_DATA, 1, 0x00);
        assert_eq!(inp(&mut dev, PIC_MASTER_DATA, 1), 0);
        assert_eq!(
            inp(&mut dev, PIC_SLAVE_DATA, 1),
            0x5a,
            "the slave mask is separate"
        );
    }

    #[test]
    fn pic_command_ports_and_elcr_ports_are_claimed_by_the_pics() {
        let mut dev = bus();
        for port in [
            PIC_MASTER_COMMAND,
            PIC_SLAVE_COMMAND,
            PIC_ELCR_MASTER,
            PIC_ELCR_SLAVE,
        ] {
            assert_ne!(
                inp(&mut dev, port, 4),
                u32::MAX,
                "port {port:#x} must not read as unmapped"
            );
        }
        out(&mut dev, PIC_ELCR_MASTER, 1, 0);
        assert_eq!(inp(&mut dev, PIC_ELCR_MASTER, 1), 0);
    }

    #[test]
    fn pic_initialization_through_the_ports_sets_the_vector_bases() {
        let mut dev = bus();
        init_pics(&mut dev);
        dev.pic.raise(IRQ_UART);
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
        dev.pic.raise(12);
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_SLAVE as u8 + 4));
    }

    #[test]
    fn pit_ports_are_routed_to_the_timer() {
        let mut dev = bus();
        out(&mut dev, PIT_CONTROL_PORT, 1, PIT_CH0_MODE2_LOHI);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x34);
        out(&mut dev, PIT_CHANNEL0_PORT, 1, 0x12);
        assert!(dev.state(0).contains("reload 4660"), "{}", dev.state(0));
        assert_eq!(inp(&mut dev, 0x44, 1), 0xff, "0x44 is past the PIT");
    }

    #[test]
    fn uart_ports_are_routed_and_transmit_to_the_console() {
        let console = ConsoleHandle::default();
        let mut dev = Devices::new(console.clone(), DevConfig::default());
        out(&mut dev, UART_SCRATCH_PORT, 1, 0x5a);
        assert_eq!(inp(&mut dev, UART_SCRATCH_PORT, 1), 0x5a);
        assert_ne!(inp(&mut dev, UART_LSR_PORT, 1) & LSR_THR_EMPTY, 0);
        out(&mut dev, UART_DATA_PORT, 1, b'A' as u32);
        out(&mut dev, UART_DATA_PORT, 1, b'B' as u32);
        let mut buf = [0u8; 8];
        assert_eq!(console.read(&mut buf, None), 2);
        assert_eq!(&buf[..2], b"AB");
    }

    #[test]
    fn uart_receive_interrupt_reaches_the_pic_once_per_edge() {
        let console = ConsoleHandle::default();
        let mut dev = Devices::new(console.clone(), DevConfig::default());
        init_pics(&mut dev);
        out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
        assert!(!dev.pic.has_pending());
        console.write(b"x");
        tick(&mut dev, 0);
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
        tick(&mut dev, 1);
        assert_eq!(
            dev.pic.ack(),
            None,
            "the line is still high: no second edge"
        );
        assert_eq!(inp(&mut dev, UART_DATA_PORT, 1), b'x' as u32);
    }

    #[test]
    fn uart_stops_taking_host_input_once_the_virtio_console_driver_is_up() {
        let console = ConsoleHandle::default();
        let mut dev = bus_with_console(&console);
        init_pics(&mut dev);
        out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
        console.write(b"y");
        enable_console_driver(&mut dev);
        tick(&mut dev, 0);
        assert!(console.has_input(), "hvc0 owns the input now");
        assert!(!dev.pic.has_pending());
    }

    #[test]
    fn uart_keeps_host_input_while_the_virtio_console_driver_is_not_up() {
        let console = ConsoleHandle::default();
        let mut dev = bus_with_console(&console);
        init_pics(&mut dev);
        out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
        console.write(b"y");
        tick(&mut dev, 0);
        assert!(!console.has_input());
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
    }

    #[test]
    fn reset_request_is_sticky() {
        let mut dev = bus();
        out(&mut dev, KBC_COMMAND_PORT, 1, 0xfe);
        out(&mut dev, FAST_A20_PORT, 1, 0);
        out(&mut dev, RESET_CONTROL_PORT, 1, 0);
        tick(&mut dev, 0);
        assert!(dev.reset_requested);
    }

    #[test]
    fn boot_time_defaults_to_a_value_the_guest_can_read_without_panicking() {
        let mut dev = bus();
        for index in 0..CMOS_SIZE as u8 {
            let _ = cmos_read(&mut dev, index, 0);
        }
        let seconds = cmos_read(&mut dev, CMOS_SECONDS as u8, 0);
        assert!(
            seconds <= 0x59 && seconds & 0x0f <= 9,
            "valid BCD seconds: {seconds:#x}"
        );
    }

    #[test]
    fn level_triggered_uart_line_is_released_once_the_driver_reads_the_data() {
        const MASTER_ELCR_PORT: u16 = 0x4d0;
        const MASTER_COMMAND_PORT: u16 = 0x20;
        const NON_SPECIFIC_EOI: u32 = 0x20;
        let console = ConsoleHandle::default();
        let mut dev = Devices::new(console.clone(), DevConfig::default());
        init_pics(&mut dev);
        out(&mut dev, MASTER_ELCR_PORT, 1, 1 << IRQ_UART);
        out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
        console.write(b"x");
        tick(&mut dev, 0);
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
        out(&mut dev, MASTER_COMMAND_PORT, 1, NON_SPECIFIC_EOI);
        assert!(
            dev.pic.has_pending(),
            "level line: still asserted while the byte is unread"
        );
        assert_eq!(inp(&mut dev, UART_DATA_PORT, 1), b'x' as u32);
        assert!(
            !dev.pic.has_pending(),
            "reading the data cleared the cause, so the line must drop without waiting for a tick"
        );
        assert_eq!(dev.pic.ack(), None);
    }

    #[test]
    fn edge_triggered_uart_line_is_unaffected_by_lowering() {
        let console = ConsoleHandle::default();
        let mut dev = Devices::new(console.clone(), DevConfig::default());
        init_pics(&mut dev);
        out(&mut dev, UART_IER_PORT, 1, IER_RX_AVAILABLE);
        console.write(b"x");
        tick(&mut dev, 0);
        assert!(dev.pic.has_pending());
        // Servicing the cause does not retract an edge request that is already latched.
        assert_eq!(inp(&mut dev, UART_DATA_PORT, 1), b'x' as u32);
        assert_eq!(dev.pic.ack(), Some(PIC_VECTOR_BASE_MASTER as u8 + IRQ_UART));
    }

    #[test]
    fn into_disk_hands_back_the_attached_disk_with_its_contents() {
        const MARKER: &[u8] = b"survives a reboot";
        let dev = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                disks: vec![Disk::memory(vec![0; 4096])],
                nets: vec![([2, 0, 0, 0, 0, 1], NetHandle::default())],
                virtio_consoles: vec![ConsoleHandle::default()],
                ..Default::default()
            },
        );
        let mut disks = dev.into_disks();
        assert_eq!(disks.len(), 1);
        disks[0].backend.write_at(512, MARKER).unwrap();
        // The next boot's devices are built around the returned disk and give it back again.
        let rebuilt = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                disks,
                ..Default::default()
            },
        );
        let mut disks = rebuilt.into_disks();
        let mut read_back = [0u8; MARKER.len()];
        disks[0].backend.read_at(512, &mut read_back).unwrap();
        assert_eq!(read_back, MARKER);
    }

    #[test]
    fn into_disks_without_a_disk_is_empty() {
        assert!(bus().into_disks().is_empty());
        assert!(bus_with_pci().into_disks().is_empty());
    }

    #[test]
    fn into_disks_returns_every_disk_in_attach_order() {
        let dev = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                disks: (0..3u8).map(|n| Disk::memory(vec![n; 4096])).collect(),
                nets: vec![([2, 0, 0, 0, 0, 1], NetHandle::default())],
                virtio_consoles: vec![ConsoleHandle::default()],
                ..Default::default()
            },
        );
        let firsts: Vec<u8> = dev
            .into_disks()
            .into_iter()
            .map(|mut disk| {
                let mut byte = [0u8; 1];
                disk.backend.read_at(0, &mut byte).unwrap();
                byte[0]
            })
            .collect();
        assert_eq!(firsts, vec![0, 1, 2]);
    }

    #[test]
    fn many_devices_take_consecutive_slots_in_console_disk_net_order() {
        let mut dev = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                disks: vec![Disk::memory(vec![0; 4096]), Disk::memory(vec![0; 4096])],
                nets: vec![
                    ([2, 0, 0, 0, 0, 1], NetHandle::default()),
                    ([2, 0, 0, 0, 0, 2], NetHandle::default()),
                ],
                virtio_consoles: vec![ConsoleHandle::default(), ConsoleHandle::default()],
                ..Default::default()
            },
        );
        let mut device_ids = Vec::new();
        for slot in 1..=7 {
            out(&mut dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(slot, 0));
            device_ids.push(inp(&mut dev, 0xcfc, 4) >> 16);
        }
        assert_eq!(
            device_ids,
            vec![0x1043, 0x1043, 0x1042, 0x1042, 0x1041, 0x1041, 0xffff]
        );
    }

    #[test]
    fn the_uart_yields_only_to_a_virtio_console_that_shares_its_handle() {
        let uart_console = ConsoleHandle::default();
        let separate = Devices::new(
            uart_console.clone(),
            DevConfig {
                virtio_consoles: vec![ConsoleHandle::default()],
                ..Default::default()
            },
        );
        assert_eq!(separate.uart_peer_console, None);
        let shared = Devices::new(
            uart_console.clone(),
            DevConfig {
                virtio_consoles: vec![ConsoleHandle::default(), uart_console],
                ..Default::default()
            },
        );
        assert_eq!(shared.uart_peer_console, Some(1));
    }

    const LAPIC_SPURIOUS_VECTOR_REGISTER: u64 = 0xfee0_00f0;
    const LAPIC_SOFTWARE_ENABLE: u64 = 0x1ff;
    const IOAPIC_SELECT: u64 = 0xfec0_0000;
    const IOAPIC_WINDOW: u64 = 0xfec0_0010;
    const IOAPIC_ID_REGISTER: u64 = 0;
    const IOAPIC_FIRST_REDIRECTION: u64 = 0x10;
    const HPET_CONFIG: u64 = 0xfed0_0010;
    const HPET_TIMER0_CONFIG: u64 = 0xfed0_0100;
    const HPET_TIMER0_COMPARATOR: u64 = 0xfed0_0108;
    const HPET_TIMER0_FSB_ROUTE: u64 = 0xfed0_0110;
    const HPET_ENABLE: u64 = 1;
    const HPET_TIMER_INTERRUPT_ENABLE: u64 = 1 << 2;
    const HPET_TIMER_LEVEL: u64 = 1 << 1;
    const HPET_TIMER_FSB_ENABLE: u64 = 1 << 14;
    const HPET_ROUTE_SHIFT: u32 = 9;
    const HPET_PIN: u64 = 22;
    const TEST_VECTOR: u8 = 0x55;
    const FSB_VECTOR: u32 = 0x66;
    const MSI_ADDRESS_BASE: u64 = 0xfee0_0000;
    const PM_TIMER_ENABLE_PORT: u16 = 0x602;
    const PM_CONTROL_PORT: u16 = 0x604;
    const PM_TIMER_PORT: u16 = 0x608;
    const PM_SLEEP_S5: u32 = 0x3401;
    const PM_TIMER_STATUS_PORT: u16 = 0x600;
    const SCI_VECTOR: u8 = 0x28 + 4;

    /// A machine with the local APIC software-enabled and pin `pin` of the I/O APIC routed to
    /// `vector` (edge, fixed, destination 0).
    fn with_routed_pin(pin: u64, vector: u8, level: bool) -> (Devices, Vec<u8>) {
        let mut dev = bus();
        let mut ram = vec![0u8; RAM_BYTES];
        {
            let mut mem = test_mem(&mut ram);
            assert!(dev.mmio_write(
                LAPIC_SPURIOUS_VECTOR_REGISTER,
                4,
                LAPIC_SOFTWARE_ENABLE,
                &mut mem
            ));
            let register = IOAPIC_FIRST_REDIRECTION + 2 * pin;
            let low = vector as u64 | (level as u64) << 15;
            for (select, value) in [(register + 1, 0), (register, low)] {
                assert!(dev.mmio_write(IOAPIC_SELECT, 4, select, &mut mem));
                assert!(dev.mmio_write(IOAPIC_WINDOW, 4, value, &mut mem));
            }
        }
        (dev, ram)
    }

    fn mmio_write(dev: &mut Devices, ram: &mut [u8], address: u64, size: u8, value: u64) {
        let mut mem = test_mem(ram);
        assert!(
            dev.mmio_write(address, size, value, &mut mem),
            "{address:#x}"
        );
    }

    #[test]
    fn the_s5_sleep_write_requests_shutdown_through_the_pm_control_port() {
        let mut dev = bus();
        assert!(!dev.shutdown_requested());
        out(&mut dev, PM_CONTROL_PORT, 2, 0x3001);
        assert!(!dev.shutdown_requested(), "sleep type 4 is not offered");
        out(&mut dev, PM_CONTROL_PORT, 2, PM_SLEEP_S5);
        assert!(dev.shutdown_requested());
        assert!(!dev.reset_requested);
    }

    #[test]
    fn the_pm_timer_reads_through_its_port_and_follows_virtual_time() {
        let mut dev = bus();
        assert_eq!(dev.io_in(PM_TIMER_PORT, 4, 0), 0);
        let one_second = dev.io_in(PM_TIMER_PORT, 4, ONE_SECOND_NS);
        assert_eq!(one_second, 3_579_545);
    }

    #[test]
    fn the_pm_timer_overflow_interrupts_on_the_sci_line_and_clearing_the_status_drops_it() {
        let mut dev = bus();
        init_pics(&mut dev);
        out(&mut dev, PIC_MASTER_DATA, 1, 0);
        out(&mut dev, PIC_SLAVE_DATA, 1, 0);
        out(&mut dev, PM_TIMER_ENABLE_PORT, 2, 1);
        let deadline = dev
            .next_deadline(0)
            .expect("the timer overflow is scheduled");
        tick(&mut dev, deadline - 1_000);
        assert!(!dev.pic.has_pending());
        tick(&mut dev, deadline);
        assert!(dev.pic.has_pending());
        assert_eq!(dev.pic.ack(), Some(SCI_VECTOR));
        dev.io_out(PM_TIMER_STATUS_PORT, 2, 1, deadline);
        tick(&mut dev, deadline + 1);
        assert!(!dev.sci_level);
    }

    #[test]
    fn ecam_reads_the_same_config_space_as_the_cf8_ports() {
        let mut dev = bus_with_pci();
        let via_ports = |dev: &mut Devices, slot: u32, register: u32| {
            out(dev, PCI_CONFIG_PORT_FIRST, 4, pci_address(slot, register));
            inp(dev, 0xcfc, 4)
        };
        for (slot, register) in [(0, 0), (1, 0), (1, 8), (1, 0x2c), (1, 0x40), (2, 0)] {
            let address = pci::ECAM_BASE + (slot << 15) + register as u64;
            assert_eq!(
                dev.mmio_read(address, 4),
                Some(via_ports(&mut dev, slot as u32, register) as u64),
                "slot {slot} register {register:#x}"
            );
        }
        assert_eq!(dev.mmio_read(pci::ECAM_BASE, 2), Some(0x8086));
        assert_eq!(
            dev.mmio_read(pci::ECAM_BASE + (1 << 15), 2),
            Some(VIRTIO_VENDOR as u64)
        );
    }

    #[test]
    fn ecam_reports_absent_functions_as_all_ones_and_extended_space_as_zero() {
        let mut dev = bus_with_pci();
        assert_eq!(
            dev.mmio_read(pci::ECAM_BASE + (9 << 15), 4),
            Some(0xffff_ffff)
        );
        assert_eq!(
            dev.mmio_read(pci::ECAM_BASE + (9 << 15) + 0x100, 4),
            Some(0xffff_ffff)
        );
        assert_eq!(
            dev.mmio_read(pci::ECAM_BASE + (1 << 15) + 0x100, 4),
            Some(0)
        );
        assert_eq!(
            dev.mmio_read(pci::ECAM_BASE + (1 << 15) + 0xffc, 4),
            Some(0)
        );
    }

    #[test]
    fn ecam_writes_reach_the_functions_config_space() {
        let mut dev = bus_with_pci();
        let mut ram = vec![0u8; RAM_BYTES];
        let command = pci::ECAM_BASE + (1 << 15) + COMMAND_REGISTER as u64;
        mmio_write(&mut dev, &mut ram, command, 2, MEMORY_SPACE_ENABLE as u64);
        assert_eq!(dev.mmio_read(command, 2), Some(MEMORY_SPACE_ENABLE as u64));
        // Memory decoding is now on: BAR0 answers.
        assert!(
            dev.mmio_read(MMIO_BASE + COMMON_DEVICE_STATUS_OFFSET, 1)
                .is_some()
        );
        // Extended space is read-only nothing.
        mmio_write(&mut dev, &mut ram, command + 0x100, 4, 0xdead_beef);
        assert_eq!(dev.mmio_read(command + 0x100, 4), Some(0));
    }

    #[test]
    fn the_ioapic_id_is_the_one_the_madt_names() {
        let mut dev = bus();
        let mut ram = vec![0u8; RAM_BYTES];
        mmio_write(&mut dev, &mut ram, IOAPIC_SELECT, 4, IOAPIC_ID_REGISTER);
        assert_eq!(
            dev.mmio_read(IOAPIC_WINDOW, 4),
            Some((ioapic::IOAPIC_ID as u64) << 24)
        );
    }

    #[test]
    fn an_hpet_timer_interrupts_through_the_ioapic_pin_the_guest_chose() {
        let (mut dev, mut ram) = with_routed_pin(HPET_PIN, TEST_VECTOR, false);
        mmio_write(&mut dev, &mut ram, HPET_CONFIG, 8, HPET_ENABLE);
        let config = HPET_TIMER_INTERRUPT_ENABLE | HPET_PIN << HPET_ROUTE_SHIFT;
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_CONFIG, 8, config);
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_COMPARATOR, 8, 1000);
        let deadline = dev.next_deadline(0).expect("armed");
        assert!(!dev.interrupt_pending());
        tick(&mut dev, deadline);
        assert!(dev.interrupt_pending());
        assert_eq!(dev.interrupt_ack(), Some(TEST_VECTOR));
        assert_eq!(dev.next_deadline(deadline), None, "one-shot");
    }

    #[test]
    fn a_level_hpet_timer_re_interrupts_after_the_eoi_until_its_status_is_cleared() {
        let (mut dev, mut ram) = with_routed_pin(HPET_PIN, TEST_VECTOR, true);
        mmio_write(&mut dev, &mut ram, HPET_CONFIG, 8, HPET_ENABLE);
        let config = HPET_TIMER_INTERRUPT_ENABLE | HPET_TIMER_LEVEL | HPET_PIN << HPET_ROUTE_SHIFT;
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_CONFIG, 8, config);
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_COMPARATOR, 8, 1000);
        let deadline = dev.next_deadline(0).unwrap();
        tick(&mut dev, deadline);
        assert_eq!(dev.interrupt_ack(), Some(TEST_VECTOR));
        // The handler EOIs without clearing the status: the line is still high.
        mmio_write(&mut dev, &mut ram, 0xfee0_00b0, 4, 0);
        assert_eq!(dev.interrupt_ack(), Some(TEST_VECTOR));
        mmio_write(&mut dev, &mut ram, 0xfed0_0020, 8, 1);
        mmio_write(&mut dev, &mut ram, 0xfee0_00b0, 4, 0);
        assert_eq!(dev.interrupt_ack(), None);
    }

    #[test]
    fn an_hpet_timer_with_fsb_delivery_sends_its_message_to_the_lapic() {
        let (mut dev, mut ram) = with_routed_pin(HPET_PIN, TEST_VECTOR, false);
        mmio_write(&mut dev, &mut ram, HPET_CONFIG, 8, HPET_ENABLE);
        mmio_write(
            &mut dev,
            &mut ram,
            HPET_TIMER0_FSB_ROUTE,
            8,
            MSI_ADDRESS_BASE << 32 | FSB_VECTOR as u64,
        );
        let config = HPET_TIMER_INTERRUPT_ENABLE | HPET_TIMER_FSB_ENABLE;
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_CONFIG, 8, config);
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_COMPARATOR, 8, 500);
        let deadline = dev.next_deadline(0).unwrap();
        tick(&mut dev, deadline);
        assert_eq!(dev.interrupt_ack(), Some(FSB_VECTOR as u8));
    }

    const IOMMU_BASE: u64 = crate::dev::iommu::MMIO_BASE;
    const IOMMU_DEVICE_TABLE_REGISTER: u64 = 0x00;
    const IOMMU_CONTROL_REGISTER: u64 = 0x18;
    const IOMMU_ENABLE: u64 = 1;
    const IOMMU_RAM_BYTES: usize = 1 << 20;
    const DEVICE_TABLE_AT: usize = 0x1_0000;
    const DEVICE_TABLE_TWO_PAGES: u64 = 1;
    const INTERRUPT_TABLE_AT: usize = 0x2_0000;
    const DEVICE_TABLE_ENTRY_BYTES: usize = 32;
    /// The I/O APIC's entry has V clear, as Linux leaves it: only IV matters for interrupts.
    const DTE_INTERRUPT_QUAD: usize = 16;
    /// IV, IntTabLen 9 (512 entries), IntCtl 2 (remap).
    const DTE_REMAP_9: u64 = 1 | 9 << 1 | 2 << 60;
    const IRTE_REMAP_ENABLE: u32 = 1;
    const IRTE_VECTOR_SHIFT: u32 = 16;
    const ISA_IRQ_1_PIN: u64 = 1;
    const REMAPPED_VECTOR: u8 = 0x77;
    const IRTE_INDEX: u8 = 5;
    const MSI_ADDRESS_LOW_OFFSET: u64 = iommu::MSI_ADDRESS_LOW as u64;
    const MSI_DATA_OFFSET: u64 = iommu::MSI_DATA as u64;
    const MSI_CONTROL_OFFSET: u64 = iommu::MSI_CONTROL as u64;
    const IOMMU_FUNCTION_ECAM: u64 = pci::ECAM_BASE + (2 << 12);

    /// A machine with the IOMMU and a console, the local APIC on and the I/O APIC pin `pin`
    /// routed to the IRTE index `IRTE_INDEX`; a megabyte of RAM for the guest's tables.
    fn iommu_machine(pin: u64, level: bool) -> (Devices, Vec<u8>) {
        let mut dev = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                iommu: true,
                virtio_consoles: vec![ConsoleHandle::default()],
                ..Default::default()
            },
        );
        let mut ram = vec![0u8; IOMMU_RAM_BYTES];
        let mut mem = test_mem(&mut ram);
        assert!(dev.mmio_write(
            LAPIC_SPURIOUS_VECTOR_REGISTER,
            4,
            LAPIC_SOFTWARE_ENABLE,
            &mut mem
        ));
        let register = IOAPIC_FIRST_REDIRECTION + 2 * pin;
        let low = IRTE_INDEX as u64 | (level as u64) << 15;
        for (select, value) in [(register + 1, 0), (register, low)] {
            assert!(dev.mmio_write(IOAPIC_SELECT, 4, select, &mut mem));
            assert!(dev.mmio_write(IOAPIC_WINDOW, 4, value, &mut mem));
        }
        (dev, ram)
    }

    fn write_u64(ram: &mut [u8], at: usize, value: u64) {
        ram[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }

    /// Gives `requester` a remapping device table entry over the one interrupt table, and IRTE
    /// `IRTE_INDEX` the vector `vector` (or a disabled entry for `None`).
    fn remap_requester(ram: &mut [u8], requester: u16, vector: Option<u8>) {
        let entry = DEVICE_TABLE_AT + requester as usize * DEVICE_TABLE_ENTRY_BYTES;
        write_u64(
            ram,
            entry + DTE_INTERRUPT_QUAD,
            DTE_REMAP_9 | INTERRUPT_TABLE_AT as u64,
        );
        let irte = vector.map_or(0, |vector| {
            IRTE_REMAP_ENABLE | (vector as u32) << IRTE_VECTOR_SHIFT
        });
        let at = INTERRUPT_TABLE_AT + 4 * IRTE_INDEX as usize;
        ram[at..at + 4].copy_from_slice(&irte.to_le_bytes());
    }

    fn enable_iommu(dev: &mut Devices, ram: &mut [u8]) {
        // Two pages of device table (the size field is pages - 1): entries up to device id 255.
        let table = DEVICE_TABLE_AT as u64 | DEVICE_TABLE_TWO_PAGES;
        mmio_write(dev, ram, IOMMU_BASE + IOMMU_DEVICE_TABLE_REGISTER, 8, table);
        mmio_write(
            dev,
            ram,
            IOMMU_BASE + IOMMU_CONTROL_REGISTER,
            8,
            IOMMU_ENABLE,
        );
    }

    fn fire_hpet_timer(dev: &mut Devices, ram: &mut [u8], pin: u64, level: bool) {
        mmio_write(dev, ram, HPET_CONFIG, 8, HPET_ENABLE);
        let mut config = HPET_TIMER_INTERRUPT_ENABLE | pin << HPET_ROUTE_SHIFT;
        if level {
            config |= HPET_TIMER_LEVEL;
        }
        mmio_write(dev, ram, HPET_TIMER0_CONFIG, 8, config);
        mmio_write(dev, ram, HPET_TIMER0_COMPARATOR, 8, 1000);
        let deadline = dev.next_deadline(0).unwrap();
        let mut mem = test_mem(ram);
        dev.tick(deadline, &mut mem);
    }

    #[test]
    fn ioapic_interrupts_are_remapped_through_the_irte_once_the_iommu_is_on() {
        let (mut dev, mut ram) = iommu_machine(HPET_PIN, false);
        remap_requester(&mut ram, IOAPIC_REQUESTER, Some(REMAPPED_VECTOR));
        enable_iommu(&mut dev, &mut ram);
        fire_hpet_timer(&mut dev, &mut ram, HPET_PIN, false);
        assert!(dev.interrupt_pending());
        assert_eq!(dev.interrupt_ack(), Some(REMAPPED_VECTOR));
        assert_eq!(
            dev.interrupt_ack(),
            None,
            "the index itself is never delivered"
        );
    }

    #[test]
    fn with_the_iommu_off_the_ioapics_vector_field_is_the_vector() {
        let (mut dev, mut ram) = iommu_machine(HPET_PIN, false);
        remap_requester(&mut ram, IOAPIC_REQUESTER, Some(REMAPPED_VECTOR));
        // The table is programmed but IOMMUEn is clear.
        let table = DEVICE_TABLE_AT as u64 | DEVICE_TABLE_TWO_PAGES;
        mmio_write(
            &mut dev,
            &mut ram,
            IOMMU_BASE + IOMMU_DEVICE_TABLE_REGISTER,
            8,
            table,
        );
        // Vector 5 is below the first valid one: give the entry a real vector instead.
        let register = IOAPIC_FIRST_REDIRECTION + 2 * HPET_PIN;
        mmio_write(&mut dev, &mut ram, IOAPIC_SELECT, 4, register);
        mmio_write(&mut dev, &mut ram, IOAPIC_WINDOW, 4, TEST_VECTOR as u64);
        fire_hpet_timer(&mut dev, &mut ram, HPET_PIN, false);
        assert_eq!(dev.interrupt_ack(), Some(TEST_VECTOR));
    }

    #[test]
    fn an_interrupt_the_irte_does_not_enable_is_dropped() {
        let (mut dev, mut ram) = iommu_machine(HPET_PIN, false);
        remap_requester(&mut ram, IOAPIC_REQUESTER, None);
        enable_iommu(&mut dev, &mut ram);
        fire_hpet_timer(&mut dev, &mut ram, HPET_PIN, false);
        assert!(!dev.interrupt_pending());
        assert_eq!(dev.interrupt_ack(), None);
    }

    #[test]
    fn an_isa_line_raised_on_the_cpu_path_waits_for_the_next_tick_to_be_remapped() {
        let (mut dev, mut ram) = iommu_machine(ISA_IRQ_1_PIN, false);
        remap_requester(&mut ram, IOAPIC_REQUESTER, Some(REMAPPED_VECTOR));
        enable_iommu(&mut dev, &mut ram);
        // As a kernel running on the APIC leaves it: every PIC input masked.
        out(&mut dev, PIC_MASTER_DATA, 1, 0xff);
        dev.pic.raise(1);
        // `interrupt_pending` has no RAM to read the IRTE from.
        assert!(!dev.interrupt_pending());
        let mut mem = test_mem(&mut ram);
        dev.tick(1, &mut mem);
        assert!(dev.interrupt_pending());
        assert_eq!(dev.interrupt_ack(), Some(REMAPPED_VECTOR));
    }

    #[test]
    fn a_level_interrupt_is_re_sent_after_the_eoi_of_the_remapped_vector() {
        let (mut dev, mut ram) = iommu_machine(HPET_PIN, true);
        remap_requester(&mut ram, IOAPIC_REQUESTER, Some(REMAPPED_VECTOR));
        enable_iommu(&mut dev, &mut ram);
        fire_hpet_timer(&mut dev, &mut ram, HPET_PIN, true);
        assert_eq!(dev.interrupt_ack(), Some(REMAPPED_VECTOR));
        // The EOI carries the delivered vector, not the index in the redirection entry.
        mmio_write(&mut dev, &mut ram, 0xfee0_00b0, 4, 0);
        let mut mem = test_mem(&mut ram);
        dev.tick(2_000_000, &mut mem);
        assert_eq!(
            dev.interrupt_ack(),
            Some(REMAPPED_VECTOR),
            "the line is still high"
        );
    }

    #[test]
    fn hpet_fsb_messages_go_through_the_iommu_as_the_hpets_requester() {
        let (mut dev, mut ram) = iommu_machine(HPET_PIN, false);
        remap_requester(&mut ram, HPET_REQUESTER, Some(REMAPPED_VECTOR));
        enable_iommu(&mut dev, &mut ram);
        mmio_write(&mut dev, &mut ram, HPET_CONFIG, 8, HPET_ENABLE);
        mmio_write(
            &mut dev,
            &mut ram,
            HPET_TIMER0_FSB_ROUTE,
            8,
            MSI_ADDRESS_BASE << 32 | IRTE_INDEX as u64,
        );
        let config = HPET_TIMER_INTERRUPT_ENABLE | HPET_TIMER_FSB_ENABLE;
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_CONFIG, 8, config);
        mmio_write(&mut dev, &mut ram, HPET_TIMER0_COMPARATOR, 8, 500);
        let deadline = dev.next_deadline(0).unwrap();
        let mut mem = test_mem(&mut ram);
        dev.tick(deadline, &mut mem);
        assert_eq!(dev.interrupt_ack(), Some(REMAPPED_VECTOR));
    }

    /// Brings the console's MSI-X up by hand: memory space on, MSI-X on, driver OK, table entry
    /// 0 pointing at the local APIC with data `data`, the configuration vector assigned to it.
    fn console_with_msix_config_vector(dev: &mut Devices, ram: &mut [u8], data: u64) {
        const CONSOLE_DEVFN: u64 = 8;
        const MSIX_CONTROL_HIGH_BYTE: u64 = 0x87;
        const MSIX_ENABLE_BIT_IN_HIGH_BYTE: u64 = 0x80;
        const COMMON_MSIX_CONFIG_OFFSET: u64 = 0x10;
        const MSIX_TABLE: u64 = 0x4000;
        let function = pci::ECAM_BASE + (CONSOLE_DEVFN << 12);
        let command = function + COMMAND_REGISTER as u64;
        mmio_write(dev, ram, command, 2, MEMORY_SPACE_ENABLE as u64);
        mmio_write(
            dev,
            ram,
            function + MSIX_CONTROL_HIGH_BYTE,
            1,
            MSIX_ENABLE_BIT_IN_HIGH_BYTE,
        );
        let bar = |offset: u64| MMIO_BASE + offset;
        let status = bar(COMMON_DEVICE_STATUS_OFFSET);
        mmio_write(dev, ram, status, 1, STATUS_DRIVER_OK as u64);
        mmio_write(dev, ram, bar(MSIX_TABLE), 4, MSI_ADDRESS_BASE);
        mmio_write(dev, ram, bar(MSIX_TABLE + 4), 4, 0);
        mmio_write(dev, ram, bar(MSIX_TABLE + 8), 4, data);
        mmio_write(dev, ram, bar(MSIX_TABLE + 12), 4, 0);
        mmio_write(dev, ram, bar(COMMON_MSIX_CONFIG_OFFSET), 2, 0);
    }

    #[test]
    fn a_virtio_msix_message_reaches_the_lapic_without_the_iommu() {
        let console = ConsoleHandle::default();
        let mut dev = bus_with_console(&console);
        let mut ram = vec![0u8; IOMMU_RAM_BYTES];
        mmio_write(
            &mut dev,
            &mut ram,
            LAPIC_SPURIOUS_VECTOR_REGISTER,
            4,
            LAPIC_SOFTWARE_ENABLE,
        );
        console_with_msix_config_vector(&mut dev, &mut ram, TEST_VECTOR as u64);
        console.resize(100, 30);
        let mut mem = test_mem(&mut ram);
        dev.tick(1, &mut mem);
        assert_eq!(dev.interrupt_ack(), Some(TEST_VECTOR));
    }

    #[test]
    fn a_virtio_msix_message_is_remapped_with_the_functions_requester_id() {
        const CONSOLE_REQUESTER: u16 = 8;
        let console = ConsoleHandle::default();
        let mut dev = Devices::new(
            ConsoleHandle::default(),
            DevConfig {
                iommu: true,
                virtio_consoles: vec![console.clone()],
                ..Default::default()
            },
        );
        let mut ram = vec![0u8; IOMMU_RAM_BYTES];
        mmio_write(
            &mut dev,
            &mut ram,
            LAPIC_SPURIOUS_VECTOR_REGISTER,
            4,
            LAPIC_SOFTWARE_ENABLE,
        );
        remap_requester(&mut ram, CONSOLE_REQUESTER, Some(REMAPPED_VECTOR));
        enable_iommu(&mut dev, &mut ram);
        console_with_msix_config_vector(&mut dev, &mut ram, IRTE_INDEX as u64);
        console.resize(100, 30);
        let mut mem = test_mem(&mut ram);
        dev.tick(1, &mut mem);
        assert_eq!(dev.interrupt_ack(), Some(REMAPPED_VECTOR));
    }

    #[test]
    fn the_iommus_own_msi_is_delivered_to_the_lapic_once() {
        let (mut dev, mut ram) = iommu_machine(HPET_PIN, false);
        let function = IOMMU_FUNCTION_ECAM;
        mmio_write(
            &mut dev,
            &mut ram,
            function + MSI_ADDRESS_LOW_OFFSET,
            4,
            MSI_ADDRESS_BASE,
        );
        mmio_write(
            &mut dev,
            &mut ram,
            function + MSI_DATA_OFFSET,
            2,
            TEST_VECTOR as u64,
        );
        mmio_write(&mut dev, &mut ram, function + MSI_CONTROL_OFFSET, 2, 1);
        dev.pci.iommu().unwrap().raise_msi();
        let mut mem = test_mem(&mut ram);
        dev.tick(1, &mut mem);
        assert_eq!(dev.interrupt_ack(), Some(TEST_VECTOR));
        dev.tick(2, &mut mem);
        assert_eq!(dev.interrupt_ack(), None);
    }

    #[test]
    fn with_intcapxt_the_iommus_interrupt_uses_the_xt_register_and_skips_remapping() {
        const XT_GENERAL_REGISTER: u64 = 0x170;
        const CONTROL_XT: u64 = 1 << 50 | 1 << 51;
        const XT_VECTOR_SHIFT: u32 = 32;
        let (mut dev, mut ram) = iommu_machine(HPET_PIN, false);
        let control = IOMMU_BASE + IOMMU_CONTROL_REGISTER;
        mmio_write(&mut dev, &mut ram, control, 8, CONTROL_XT);
        let register = IOMMU_BASE + XT_GENERAL_REGISTER;
        mmio_write(
            &mut dev,
            &mut ram,
            register,
            8,
            (TEST_VECTOR as u64) << XT_VECTOR_SHIFT,
        );
        dev.pci.iommu().unwrap().raise_msi();
        let mut mem = test_mem(&mut ram);
        dev.tick(1, &mut mem);
        assert_eq!(dev.interrupt_ack(), Some(TEST_VECTOR));
    }
}
