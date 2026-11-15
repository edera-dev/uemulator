use super::guestmem::GuestMem;
use super::virtio::{DEVICE_TYPE_CONSOLE, Queue, VirtioDev};
use crate::console::ConsoleHandle;

/// VIRTIO_CONSOLE_F_SIZE: config space carries cols/rows and changes raise a config interrupt.
/// The multiport feature is not offered, so there are no control queues: just port 0's pair.
const F_SIZE: u64 = 1;

/// Host-to-guest queue (guest posts empty buffers).
const RECEIVEQ: usize = 0;
/// Guest-to-host queue.
const TRANSMITQ: usize = 1;
const QUEUE_COUNT: usize = 2;

/// Device config layout: le16 cols, le16 rows, le32 max_nr_ports, le32 emerg_wr.
const CONFIG_LEN: usize = 12;
const CONFIG_COLS_OFFSET: usize = 0;
const CONFIG_ROWS_OFFSET: usize = 2;
const CONFIG_MAX_NR_PORTS_OFFSET: usize = 4;
const MAX_NR_PORTS: u32 = 1;

/// Transmit data is handed to the console in pieces of this size. A chain longer than one piece
/// is forwarded whole, never truncated: dropping console output silently is worse than briefly
/// holding one more piece of a buffer the guest already proved is in RAM.
const TX_CHUNK_BYTES: usize = 1 << 16;

pub(crate) struct VirtioConsole {
    console: ConsoleHandle,
    cols: u16,
    rows: u16,
    /// Size changed since the guest was last told; consumed by `config_event`.
    changed: bool,
}

impl VirtioConsole {
    pub fn new(console: ConsoleHandle) -> Self {
        let (cols, rows) = console.size();
        VirtioConsole {
            console,
            cols,
            rows,
            changed: false,
        }
    }
}

impl VirtioDev for VirtioConsole {
    fn device_type(&self) -> u16 {
        DEVICE_TYPE_CONSOLE
    }

    fn features(&self) -> u64 {
        F_SIZE
    }

    fn num_queues(&self) -> usize {
        QUEUE_COUNT
    }

    fn config_read(&self, off: usize) -> u8 {
        let mut cfg = [0u8; CONFIG_LEN];
        cfg[CONFIG_COLS_OFFSET..CONFIG_COLS_OFFSET + 2].copy_from_slice(&self.cols.to_le_bytes());
        cfg[CONFIG_ROWS_OFFSET..CONFIG_ROWS_OFFSET + 2].copy_from_slice(&self.rows.to_le_bytes());
        cfg[CONFIG_MAX_NR_PORTS_OFFSET..CONFIG_MAX_NR_PORTS_OFFSET + 4]
            .copy_from_slice(&MAX_NR_PORTS.to_le_bytes());
        cfg.get(off).copied().unwrap_or(0)
    }

    fn kick(&mut self, queue: usize, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if queue != TRANSMITQ {
            return false;
        }
        let mut any = false;
        while let Some(chain) = queues[TRANSMITQ].pop(mem) {
            chain.read_chunks(mem, TX_CHUNK_BYTES, |piece| {
                self.console.guest_tx_bytes(piece)
            });
            queues[TRANSMITQ].push_used(mem, chain.head, 0);
            any = true;
        }
        any
    }

    fn poll(&mut self, queues: &mut [Queue], mem: &mut GuestMem) -> bool {
        if let Some((cols, rows)) = self.console.take_size_change() {
            (self.cols, self.rows) = (cols, rows);
            self.changed = true;
        }
        let mut any = false;
        while self.console.has_input() {
            let Some(chain) = queues[RECEIVEQ].pop(mem) else {
                break;
            };
            // Take only what the buffer can hold: bytes pulled from the console and not stored
            // would be lost. A buffer with no writable space is returned empty.
            let data = self.console.guest_rx_bulk(chain.writable());
            let written = chain.write_at(mem, 0, &data);
            if written < data.len() {
                self.console.guest_unrx_bulk(&data[written..]);
            }
            queues[RECEIVEQ].push_used(mem, chain.head, written as u32);
            any = true;
        }
        any
    }

    fn config_event(&mut self) -> bool {
        core::mem::take(&mut self.changed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::virtio::testutil::*;
    use super::super::virtio::{DESC_F_NEXT, DESC_F_WRITE};
    use super::*;
    use crate::dev::guestmem::test_mem;

    const QUEUE_SIZE: u16 = 8;
    const RX_DESC: u64 = 0x1000;
    const RX_AVAIL: u64 = 0x2000;
    const RX_USED: u64 = 0x3000;
    const TX_DESC: u64 = 0x8000;
    const TX_AVAIL: u64 = 0x9000;
    const TX_USED: u64 = 0xa000;
    const BUF: u64 = 0xc000;
    const RX_BUF: u64 = 0xd000;
    const RAM_BYTES: usize = 0x2_0000;

    struct Rig {
        console: ConsoleHandle,
        dev: VirtioConsole,
        queues: Vec<Queue>,
        ram: Vec<u8>,
    }

    impl Rig {
        fn new() -> Rig {
            let console = ConsoleHandle::default();
            let mut ram = vec![0u8; RAM_BYTES];
            let queues = vec![
                setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
                setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
            ];
            Rig {
                dev: VirtioConsole::new(console.clone()),
                console,
                queues,
                ram,
            }
        }

        fn kick(&mut self, queue: usize) -> bool {
            let mut mem = test_mem(&mut self.ram);
            self.dev.kick(queue, &mut self.queues, &mut mem)
        }

        fn poll(&mut self) -> bool {
            let mut mem = test_mem(&mut self.ram);
            self.dev.poll(&mut self.queues, &mut mem)
        }

        fn transmit(&mut self, data: &[u8]) {
            self.ram[BUF as usize..BUF as usize + data.len()].copy_from_slice(data);
            put_desc(&mut self.ram, TX_DESC, 0, BUF, data.len() as u32, 0, 0);
            post(&mut self.ram, TX_AVAIL, QUEUE_SIZE, 0);
        }

        fn post_receive(&mut self, head: u16, addr: u64, len: u32) {
            put_desc(&mut self.ram, RX_DESC, head, addr, len, DESC_F_WRITE, 0);
            post(&mut self.ram, RX_AVAIL, QUEUE_SIZE, head);
        }

        fn drain_console(&self) -> Vec<u8> {
            let mut out = vec![0u8; 1 << 20];
            let count = self.console.read(&mut out, None);
            out.truncate(count);
            out
        }
    }

    #[test]
    fn transmit_does_not_forward_bytes_from_a_write_only_dma_page() {
        let mut rig = Rig::new();
        rig.transmit(b"secret");
        let mut iommu = iommu_deny_read(&mut rig.ram, BUF);
        let mut mem = test_mem(&mut rig.ram);
        assert!(mem.with_dma(&mut iommu, 0x18, |mem| rig.dev.kick(
            TRANSMITQ,
            &mut rig.queues,
            mem
        )));
        assert_eq!(used_idx(&rig.ram, TX_USED), 1);
        assert!(rig.drain_console().is_empty());
    }

    #[test]
    fn receive_chain_without_writable_bytes_leaves_input_queued() {
        let mut rig = Rig::new();
        rig.console.write(b"x");
        put_desc(&mut rig.ram, RX_DESC, 0, RX_BUF, 8, 0, 0);
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        rig.poll();
        assert!(
            rig.console.has_input(),
            "the byte must not be consumed and lost"
        );
    }

    #[test]
    fn device_identity_features_and_queues() {
        let rig = Rig::new();
        assert_eq!(rig.dev.device_type(), 3);
        assert_eq!(rig.dev.features(), 1, "only VIRTIO_CONSOLE_F_SIZE");
        assert_eq!(rig.dev.num_queues(), 2);
        assert_eq!((RECEIVEQ, TRANSMITQ), (0, 1));
    }

    #[test]
    fn config_space_reports_size_and_a_single_port() {
        let rig = Rig::new();
        let config: Vec<u8> = (0..16).map(|off| rig.dev.config_read(off)).collect();
        assert_eq!(u16::from_le_bytes([config[0], config[1]]), 80);
        assert_eq!(u16::from_le_bytes([config[2], config[3]]), 25);
        assert_eq!(
            u32::from_le_bytes(config[4..8].try_into().unwrap()),
            1,
            "max_nr_ports"
        );
        assert_eq!(&config[8..12], &[0; 4], "emerg_wr");
        assert_eq!(&config[12..], &[0; 4]);
    }

    #[test]
    fn config_space_picks_up_the_size_set_before_the_device_was_created() {
        let console = ConsoleHandle::default();
        console.resize(200, 60);
        let dev = VirtioConsole::new(console);
        assert_eq!(dev.config_read(0), 200);
        assert_eq!(dev.config_read(2), 60);
    }

    #[test]
    fn config_reads_past_the_structure_return_zero_and_writes_are_ignored() {
        let mut rig = Rig::new();
        for off in [12, 13, 100, usize::MAX] {
            assert_eq!(rig.dev.config_read(off), 0);
        }
        rig.dev.config_write(0, 0xff);
        assert_eq!(rig.dev.config_read(0), 80);
    }

    #[test]
    fn transmit_chain_is_written_to_the_console() {
        let mut rig = Rig::new();
        rig.transmit(b"hey");
        assert!(rig.kick(TRANSMITQ));
        assert_eq!(rig.drain_console(), b"hey");
        assert_eq!(used_idx(&rig.ram, TX_USED), 1);
        assert_eq!(used_entry(&rig.ram, TX_USED, 0), (0, 0));
    }

    #[test]
    fn several_transmit_chains_keep_their_order() {
        let mut rig = Rig::new();
        for (head, text) in [(0u16, &b"one "[..]), (1, b"two "), (2, b"three")] {
            let addr = BUF + 0x100 * head as u64;
            rig.ram[addr as usize..addr as usize + text.len()].copy_from_slice(text);
            put_desc(&mut rig.ram, TX_DESC, head, addr, text.len() as u32, 0, 0);
            post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, head);
        }
        assert!(rig.kick(TRANSMITQ));
        assert_eq!(rig.drain_console(), b"one two three");
        assert_eq!(used_idx(&rig.ram, TX_USED), 3);
    }

    #[test]
    fn transmit_chain_with_several_segments_is_concatenated_and_skips_writable_ones() {
        let mut rig = Rig::new();
        rig.ram[BUF as usize..BUF as usize + 3].copy_from_slice(b"abc");
        rig.ram[BUF as usize + 0x100..BUF as usize + 0x102].copy_from_slice(b"de");
        rig.ram[BUF as usize + 0x200..BUF as usize + 0x204].copy_from_slice(b"JUNK");
        put_desc(&mut rig.ram, TX_DESC, 0, BUF, 3, DESC_F_NEXT, 1);
        put_desc(
            &mut rig.ram,
            TX_DESC,
            1,
            BUF + 0x200,
            4,
            DESC_F_NEXT | DESC_F_WRITE,
            2,
        );
        put_desc(&mut rig.ram, TX_DESC, 2, BUF + 0x100, 2, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        rig.kick(TRANSMITQ);
        assert_eq!(rig.drain_console(), b"abcde");
    }

    #[test]
    fn empty_transmit_chain_is_completed_without_output() {
        let mut rig = Rig::new();
        put_desc(&mut rig.ram, TX_DESC, 0, BUF, 0, 0, 0);
        post(&mut rig.ram, TX_AVAIL, QUEUE_SIZE, 0);
        assert!(rig.kick(TRANSMITQ));
        assert!(rig.drain_console().is_empty());
        assert_eq!(used_idx(&rig.ram, TX_USED), 1);
    }

    #[test]
    fn kick_on_the_receive_queue_does_nothing() {
        let mut rig = Rig::new();
        rig.transmit(b"nope");
        assert!(!rig.kick(RECEIVEQ));
        assert!(rig.drain_console().is_empty());
        assert_eq!(used_idx(&rig.ram, TX_USED), 0);
    }

    #[test]
    fn transmit_kick_with_nothing_posted_reports_no_work() {
        let mut rig = Rig::new();
        assert!(!rig.kick(TRANSMITQ));
    }

    #[test]
    fn transmit_longer_than_one_chunk_is_forwarded_whole() {
        let mut ram = vec![0u8; 1 << 20];
        let console = ConsoleHandle::default();
        let mut dev = VirtioConsole::new(console.clone());
        let mut queues = vec![
            setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
            setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
        ];
        let data: Vec<u8> = (0..3 * TX_CHUNK_BYTES + 10)
            .map(|i| (i % 251) as u8)
            .collect();
        let data_at = 0x1_0000usize;
        ram[data_at..data_at + data.len()].copy_from_slice(&data);
        put_desc(
            &mut ram,
            TX_DESC,
            0,
            data_at as u64,
            data.len() as u32,
            0,
            0,
        );
        post(&mut ram, TX_AVAIL, QUEUE_SIZE, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(TRANSMITQ, &mut queues, &mut mem));
        let mut out = vec![0u8; data.len() + 100];
        let count = console.read(&mut out, None);
        assert_eq!(
            count,
            data.len(),
            "no byte beyond the chunk size is dropped"
        );
        assert_eq!(&out[..count], &data[..]);
    }

    #[test]
    fn transmit_chunks_arrive_at_an_output_callback_in_order() {
        let console = ConsoleHandle::default();
        let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = received.clone();
        console.on_output(move |bytes| sink.lock().unwrap().extend_from_slice(bytes));
        let mut ram = vec![0u8; 1 << 20];
        let mut dev = VirtioConsole::new(console);
        let mut queues = vec![
            setup_queue(&mut ram, RX_DESC, RX_AVAIL, RX_USED, QUEUE_SIZE),
            setup_queue(&mut ram, TX_DESC, TX_AVAIL, TX_USED, QUEUE_SIZE),
        ];
        let data: Vec<u8> = (0..TX_CHUNK_BYTES + 5).map(|i| (i % 13) as u8).collect();
        ram[0x1_0000..0x1_0000 + data.len()].copy_from_slice(&data);
        put_desc(&mut ram, TX_DESC, 0, 0x1_0000, data.len() as u32, 0, 0);
        post(&mut ram, TX_AVAIL, QUEUE_SIZE, 0);
        let mut mem = test_mem(&mut ram);
        dev.kick(TRANSMITQ, &mut queues, &mut mem);
        assert_eq!(*received.lock().unwrap(), data);
    }

    #[test]
    fn poll_without_input_leaves_posted_buffers_alone() {
        let mut rig = Rig::new();
        rig.post_receive(0, RX_BUF, 16);
        assert!(!rig.poll());
        assert_eq!(
            rig.queues[RECEIVEQ].last_avail, 0,
            "the buffer was not consumed"
        );
        assert_eq!(used_idx(&rig.ram, RX_USED), 0);
    }

    #[test]
    fn input_without_a_posted_buffer_stays_queued() {
        let mut rig = Rig::new();
        rig.console.write(b"wait");
        assert!(!rig.poll());
        assert!(rig.console.has_input());
        rig.post_receive(0, RX_BUF, 16);
        assert!(rig.poll());
        assert_eq!(&rig.ram[RX_BUF as usize..RX_BUF as usize + 4], b"wait");
        assert!(!rig.console.has_input());
    }

    #[test]
    fn input_is_delivered_into_the_buffer_and_reported_with_its_length() {
        let mut rig = Rig::new();
        rig.console.write(b"ok");
        rig.post_receive(0, RX_BUF, 64);
        assert!(rig.poll());
        assert_eq!(&rig.ram[RX_BUF as usize..RX_BUF as usize + 2], b"ok");
        assert_eq!(
            rig.ram[RX_BUF as usize + 2],
            0,
            "nothing beyond the data is written"
        );
        assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, 2));
    }

    #[test]
    fn only_what_fits_the_buffer_is_taken_and_the_rest_waits_for_the_next_buffer() {
        let mut rig = Rig::new();
        rig.console.write(b"0123456789");
        rig.post_receive(0, RX_BUF, 4);
        assert!(rig.poll());
        assert_eq!(&rig.ram[RX_BUF as usize..RX_BUF as usize + 4], b"0123");
        assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, 4));
        assert!(rig.console.has_input(), "six bytes are still queued");
        rig.post_receive(1, RX_BUF + 0x100, 16);
        assert!(rig.poll());
        assert_eq!(
            &rig.ram[RX_BUF as usize + 0x100..RX_BUF as usize + 0x106],
            b"456789"
        );
        assert_eq!(used_entry(&rig.ram, RX_USED, 1), (1, 6));
        assert!(!rig.console.has_input());
    }

    #[test]
    fn one_poll_fills_several_buffers_in_order() {
        let mut rig = Rig::new();
        rig.console.write(b"ABCDEFGHIJ");
        rig.post_receive(0, RX_BUF, 4);
        rig.post_receive(1, RX_BUF + 0x100, 8);
        assert!(rig.poll());
        assert_eq!(&rig.ram[RX_BUF as usize..RX_BUF as usize + 4], b"ABCD");
        assert_eq!(
            &rig.ram[RX_BUF as usize + 0x100..RX_BUF as usize + 0x106],
            b"EFGHIJ"
        );
        assert_eq!(used_idx(&rig.ram, RX_USED), 2);
        assert_eq!(used_entry(&rig.ram, RX_USED, 1), (1, 6));
    }

    #[test]
    fn input_spills_across_the_segments_of_one_receive_chain() {
        let mut rig = Rig::new();
        rig.console.write(b"abcde");
        put_desc(
            &mut rig.ram,
            RX_DESC,
            0,
            RX_BUF,
            3,
            DESC_F_WRITE | DESC_F_NEXT,
            1,
        );
        put_desc(&mut rig.ram, RX_DESC, 1, RX_BUF + 0x100, 3, DESC_F_WRITE, 0);
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        assert!(rig.poll());
        assert_eq!(&rig.ram[RX_BUF as usize..RX_BUF as usize + 3], b"abc");
        assert_eq!(
            &rig.ram[RX_BUF as usize + 0x100..RX_BUF as usize + 0x102],
            b"de"
        );
        assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, 5));
    }

    #[test]
    fn readable_segments_of_a_receive_chain_are_not_written() {
        let mut rig = Rig::new();
        rig.console.write(b"xyz");
        rig.ram[RX_BUF as usize..RX_BUF as usize + 4].copy_from_slice(b"KEEP");
        put_desc(&mut rig.ram, RX_DESC, 0, RX_BUF, 4, DESC_F_NEXT, 1);
        put_desc(&mut rig.ram, RX_DESC, 1, RX_BUF + 0x100, 8, DESC_F_WRITE, 0);
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        rig.poll();
        assert_eq!(&rig.ram[RX_BUF as usize..RX_BUF as usize + 4], b"KEEP");
        assert_eq!(
            &rig.ram[RX_BUF as usize + 0x100..RX_BUF as usize + 0x103],
            b"xyz"
        );
        assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, 3));
    }

    #[test]
    fn receive_buffer_with_a_single_writable_byte_gets_exactly_one_byte() {
        let mut rig = Rig::new();
        rig.console.write(b"pq");
        rig.post_receive(0, RX_BUF, 1);
        rig.poll();
        assert_eq!(rig.ram[RX_BUF as usize], b'p');
        assert_eq!(rig.ram[RX_BUF as usize + 1], 0);
        assert_eq!(used_entry(&rig.ram, RX_USED, 0), (0, 1));
        assert!(rig.console.has_input());
    }

    #[test]
    fn receive_chain_without_writable_bytes_is_returned_empty_and_the_next_buffer_gets_the_data() {
        let mut rig = Rig::new();
        rig.console.write(b"data");
        put_desc(&mut rig.ram, RX_DESC, 0, RX_BUF, 8, 0, 0);
        post(&mut rig.ram, RX_AVAIL, QUEUE_SIZE, 0);
        rig.post_receive(1, RX_BUF + 0x100, 16);
        assert!(rig.poll());
        assert_eq!(
            used_entry(&rig.ram, RX_USED, 0),
            (0, 0),
            "the useless buffer goes back empty"
        );
        assert_eq!(used_entry(&rig.ram, RX_USED, 1), (1, 4));
        assert_eq!(
            &rig.ram[RX_BUF as usize + 0x100..RX_BUF as usize + 0x104],
            b"data"
        );
    }

    #[test]
    fn zero_length_receive_buffer_does_not_lose_input() {
        let mut rig = Rig::new();
        rig.console.write(b"keep me");
        rig.post_receive(0, RX_BUF, 0);
        rig.poll();
        assert_eq!(rig.console.guest_rx_bulk(16), b"keep me");
    }

    #[test]
    fn resize_raises_one_config_event_with_the_new_size() {
        let mut rig = Rig::new();
        assert!(!rig.dev.config_event());
        rig.console.resize(132, 43);
        assert!(!rig.poll(), "a size change completes no buffers");
        assert!(rig.dev.config_event());
        assert!(!rig.dev.config_event(), "reported once");
        assert_eq!((rig.dev.config_read(0), rig.dev.config_read(2)), (132, 43));
    }

    #[test]
    fn size_change_is_only_noticed_by_poll() {
        let mut rig = Rig::new();
        rig.console.resize(100, 30);
        assert_eq!(rig.dev.config_read(0), 80, "not yet visible to the guest");
        assert!(!rig.dev.config_event());
        rig.poll();
        assert_eq!(rig.dev.config_read(0), 100);
    }

    #[test]
    fn resizing_to_the_same_size_raises_no_event() {
        let mut rig = Rig::new();
        rig.console.resize(80, 25);
        rig.poll();
        assert!(!rig.dev.config_event());
    }

    #[test]
    fn consecutive_resizes_between_polls_report_the_latest_size_once() {
        let mut rig = Rig::new();
        rig.console.resize(90, 30);
        rig.console.resize(120, 40);
        rig.poll();
        assert!(rig.dev.config_event());
        assert!(!rig.dev.config_event());
        assert_eq!((rig.dev.config_read(0), rig.dev.config_read(2)), (120, 40));
    }

    #[test]
    fn size_is_encoded_little_endian_in_the_config_space() {
        let mut rig = Rig::new();
        rig.console.resize(0x0102, 0x0304);
        rig.poll();
        let bytes: Vec<u8> = (0..4).map(|off| rig.dev.config_read(off)).collect();
        assert_eq!(bytes, vec![0x02, 0x01, 0x04, 0x03]);
    }

    #[test]
    fn input_and_resize_in_one_poll_are_both_handled() {
        let mut rig = Rig::new();
        rig.console.write(b"z");
        rig.console.resize(70, 20);
        rig.post_receive(0, RX_BUF, 8);
        assert!(rig.poll());
        assert!(rig.dev.config_event());
        assert_eq!(rig.ram[RX_BUF as usize], b'z');
    }

    #[test]
    fn receive_buffer_outside_ram_breaks_the_queue_and_keeps_the_input() {
        let mut rig = Rig::new();
        rig.console.write(b"safe");
        rig.post_receive(0, RAM_BYTES as u64 - 2, 16);
        assert!(!rig.poll());
        assert!(rig.queues[RECEIVEQ].broken);
        assert!(rig.console.has_input());
    }

    #[test]
    fn io_and_resize() {
        let c = ConsoleHandle::default();
        let mut dev = VirtioConsole::new(c.clone());
        assert_eq!(dev.config_read(CONFIG_COLS_OFFSET), 80);
        let mut ram = vec![0u8; 0x20000];
        let mut qs = vec![
            setup_queue(&mut ram, 0x1000, 0x2000, 0x3000, 8),
            setup_queue(&mut ram, 0x8000, 0x9000, 0xa000, 8),
        ];
        ram[0xc000..0xc003].copy_from_slice(b"hey");
        put_desc(&mut ram, 0x8000, 0, 0xc000, 3, 0, 0);
        post(&mut ram, 0x9000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.kick(TRANSMITQ, &mut qs, &mut mem));
        let mut b = [0u8; 8];
        assert_eq!(c.read(&mut b, None), 3);
        c.write(b"ok");
        put_desc(&mut ram, 0x1000, 0, 0xd000, 64, 2, 0);
        post(&mut ram, 0x2000, 8, 0);
        let mut mem = test_mem(&mut ram);
        assert!(dev.poll(&mut qs, &mut mem));
        assert_eq!(&ram[0xd000..0xd002], b"ok");
        c.resize(132, 43);
        let mut mem = test_mem(&mut ram);
        dev.poll(&mut qs, &mut mem);
        assert!(dev.config_event());
        assert_eq!(
            (
                dev.config_read(CONFIG_COLS_OFFSET),
                dev.config_read(CONFIG_ROWS_OFFSET)
            ),
            (132, 43)
        );
    }
}
