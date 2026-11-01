//! `EFI_RNG_PROTOCOL`, which the Linux EFI stub uses to seed KASLR. The bytes come from a
//! xoshiro256** generator seeded by the handoff's random seed and stirred with the time stamp
//! counter on each request: as good as the entropy the embedder supplied, no better.

use core::ffi::c_void;

use crate::cpu::rdtsc;
use crate::handoff::Handoff;
use crate::protocols;
use crate::sync::Global;
use crate::uefi::*;

#[repr(C)]
struct Rng {
    get_info: extern "efiapi" fn(*mut Rng, *mut usize, *mut Guid) -> Status,
    get_rng: extern "efiapi" fn(*mut Rng, *const Guid, usize, *mut u8) -> Status,
}

static PROTOCOL: Global<Rng> = Global::new(Rng { get_info, get_rng });
static STATE: Global<[u64; 4]> = Global::new([0; 4]);

fn next(state: &mut [u64; 4]) -> u64 {
    let result = state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
    let shifted = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= shifted;
    state[3] = state[3].rotate_left(45);
    result
}

pub fn init(handoff: &Handoff) {
    let state = STATE.get();
    for (word, chunk) in state.iter_mut().zip(handoff.random_seed.as_chunks::<8>().0) {
        *word = u64::from_le_bytes(*chunk);
    }
    // An all-zero state is the one fixed point of the generator.
    state[0] |= 1;
    let _ = protocols::install(
        core::ptr::null_mut(),
        guids::RNG,
        PROTOCOL.get() as *mut Rng as *mut c_void,
    );
}

extern "efiapi" fn get_info(_: *mut Rng, size: *mut usize, list: *mut Guid) -> Status {
    if size.is_null() {
        return INVALID_PARAMETER;
    }
    let needed = core::mem::size_of::<Guid>();
    unsafe {
        let capacity = size.read_unaligned();
        size.write_unaligned(needed);
        if capacity < needed {
            return BUFFER_TOO_SMALL;
        }
        if list.is_null() {
            return INVALID_PARAMETER;
        }
        list.write_unaligned(guids::RNG_ALGORITHM_RAW);
    }
    SUCCESS
}

extern "efiapi" fn get_rng(
    _: *mut Rng,
    algorithm: *const Guid,
    length: usize,
    value: *mut u8,
) -> Status {
    if value.is_null() {
        return INVALID_PARAMETER;
    }
    if !algorithm.is_null() {
        let requested = unsafe { algorithm.read_unaligned() };
        if requested != guids::RNG_ALGORITHM_RAW {
            return UNSUPPORTED;
        }
    }
    let state = STATE.get();
    state[2] ^= rdtsc();
    let output = unsafe { core::slice::from_raw_parts_mut(value, length) };
    for chunk in output.chunks_mut(8) {
        let word = next(state).to_le_bytes();
        chunk.copy_from_slice(&word[..chunk.len()]);
    }
    SUCCESS
}
