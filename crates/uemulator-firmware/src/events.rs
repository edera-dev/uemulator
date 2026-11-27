//! Events, timers and task priority levels. There are no interrupts while applications run, so
//! timers are checked whenever the application calls into the firmware (`poll`) and while it
//! waits; notification functions run at the points a real firmware's timer tick would reach.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;

use crate::cpu;
use crate::sync::Global;
use crate::time;
use crate::uefi::*;

struct EventData {
    kind: u32,
    tpl: usize,
    notify: Option<NotifyFunction>,
    context: *mut c_void,
    signaled: bool,
    /// A signal-type notification that has not run because the TPL was too high.
    pending: bool,
    group: Option<Guid>,
    timer: Timer,
}

#[derive(Clone, Copy)]
enum Timer {
    Off,
    Once { deadline: u64 },
    Periodic { deadline: u64, period: u64 },
}

struct State {
    events: Vec<*mut EventData>,
    tpl: usize,
}

static STATE: Global<State> = Global::new(State {
    events: Vec::new(),
    tpl: TPL_APPLICATION,
});

const TIMER_UNIT_NS: u64 = 100;
/// How long `WaitForEvent` sleeps between looks at its events.
const WAIT_SLICE_NS: u64 = 500_000;

fn valid(event: Event) -> bool {
    STATE.get().events.contains(&event.cast())
}

fn data(event: Event) -> &'static mut EventData {
    unsafe { &mut *event.cast::<EventData>() }
}

pub fn create(
    kind: u32,
    tpl: usize,
    notify: Option<NotifyFunction>,
    context: *mut c_void,
    group: Option<Guid>,
) -> Result<Event, Status> {
    let signal = kind & EVT_NOTIFY_SIGNAL != 0;
    let wait = kind & EVT_NOTIFY_WAIT != 0;
    if signal && wait
        || (signal || wait)
            && (notify.is_none() || !(TPL_APPLICATION..=TPL_HIGH_LEVEL).contains(&tpl))
    {
        return Err(INVALID_PARAMETER);
    }
    let group = match kind {
        EVT_SIGNAL_EXIT_BOOT_SERVICES => Some(guids::EXIT_BOOT_SERVICES_EVENT),
        EVT_SIGNAL_VIRTUAL_ADDRESS_CHANGE => Some(guids::VIRTUAL_ADDRESS_CHANGE_EVENT),
        _ => group,
    };
    let event = Box::into_raw(Box::new(EventData {
        kind,
        tpl,
        notify,
        context,
        signaled: false,
        pending: false,
        group,
        timer: Timer::Off,
    }));
    STATE.get().events.push(event);
    Ok(event.cast())
}

pub fn signal(event: Event) {
    let target = data(event);
    target.signaled = true;
    if target.kind & EVT_NOTIFY_SIGNAL != 0 {
        target.pending = true;
        run_pending();
    }
}

pub fn signal_group(group: &Guid) {
    let members: Vec<Event> = STATE
        .get()
        .events
        .iter()
        .filter(|&&event| unsafe { (*event).group } == Some(*group))
        .map(|&event| event.cast())
        .collect();
    for event in members {
        if valid(event) {
            signal(event);
        }
    }
}

/// Runs the notifications whose priority is above the current one, highest first.
fn run_pending() {
    loop {
        let state = STATE.get();
        let next = state
            .events
            .iter()
            .copied()
            .filter(|&event| unsafe { (*event).pending && (*event).tpl > state.tpl })
            .max_by_key(|&event| unsafe { (*event).tpl });
        let Some(event) = next else { return };
        let (notify, context, tpl) = unsafe {
            (*event).pending = false;
            ((*event).notify, (*event).context, (*event).tpl)
        };
        let outer = state.tpl;
        state.tpl = tpl;
        if let Some(notify) = notify {
            notify(event.cast(), context);
        }
        STATE.get().tpl = outer;
    }
}

/// Fires every timer whose deadline has passed.
pub fn poll() {
    let now = time::now_ns();
    let due: Vec<Event> = STATE
        .get()
        .events
        .iter()
        .copied()
        .filter(|&event| match unsafe { (*event).timer } {
            Timer::Off => false,
            Timer::Once { deadline } | Timer::Periodic { deadline, .. } => deadline <= now,
        })
        .map(|event| event.cast())
        .collect();
    for event in due {
        if !valid(event) {
            continue;
        }
        let target = data(event);
        target.timer = match target.timer {
            Timer::Periodic { deadline, period } if period > 0 => Timer::Periodic {
                deadline: deadline + period * ((now - deadline) / period + 1),
                period,
            },
            Timer::Periodic { .. } => Timer::Periodic {
                deadline: now,
                period: 0,
            },
            _ => Timer::Off,
        };
        signal(event);
    }
}

extern "efiapi" fn create_event(
    kind: u32,
    tpl: usize,
    notify: Option<NotifyFunction>,
    context: *mut c_void,
    event: *mut Event,
) -> Status {
    create_event_ex(kind, tpl, notify, context, ptr::null(), event)
}

extern "efiapi" fn create_event_ex(
    kind: u32,
    tpl: usize,
    notify: Option<NotifyFunction>,
    context: *const c_void,
    group: *const Guid,
    event: *mut Event,
) -> Status {
    if event.is_null() {
        return INVALID_PARAMETER;
    }
    let group = (!group.is_null()).then(|| unsafe { group.read_unaligned() });
    match create(kind, tpl, notify, context.cast_mut(), group) {
        Ok(created) => {
            unsafe { event.write(created) };
            SUCCESS
        }
        Err(status) => status,
    }
}

extern "efiapi" fn set_timer(event: Event, kind: u32, trigger: u64) -> Status {
    if !valid(event) || data(event).kind & EVT_TIMER == 0 {
        return INVALID_PARAMETER;
    }
    let now = time::now_ns();
    let delay = trigger.saturating_mul(TIMER_UNIT_NS);
    data(event).timer = match kind {
        TIMER_CANCEL => Timer::Off,
        TIMER_RELATIVE => Timer::Once {
            deadline: now.saturating_add(delay),
        },
        TIMER_PERIODIC => Timer::Periodic {
            deadline: now.saturating_add(delay),
            period: delay,
        },
        _ => return INVALID_PARAMETER,
    };
    SUCCESS
}

extern "efiapi" fn signal_event(event: Event) -> Status {
    if !valid(event) {
        return INVALID_PARAMETER;
    }
    signal(event);
    SUCCESS
}

extern "efiapi" fn close_event(event: Event) -> Status {
    let events = &mut STATE.get().events;
    let Some(position) = events
        .iter()
        .position(|&candidate| candidate == event.cast())
    else {
        return INVALID_PARAMETER;
    };
    events.swap_remove(position);
    drop(unsafe { Box::from_raw(event.cast::<EventData>()) });
    SUCCESS
}

extern "efiapi" fn check_event(event: Event) -> Status {
    if !valid(event) {
        return INVALID_PARAMETER;
    }
    if data(event).kind & EVT_NOTIFY_SIGNAL != 0 {
        return INVALID_PARAMETER;
    }
    poll();
    if !data(event).signaled && data(event).kind & EVT_NOTIFY_WAIT != 0 {
        let (notify, context) = (data(event).notify, data(event).context);
        if let Some(notify) = notify {
            notify(event, context);
        }
    }
    if valid(event) && data(event).signaled {
        data(event).signaled = false;
        return SUCCESS;
    }
    NOT_READY
}

extern "efiapi" fn wait_for_event(count: usize, events: *mut Event, index: *mut usize) -> Status {
    if count == 0 || events.is_null() || index.is_null() {
        return INVALID_PARAMETER;
    }
    if STATE.get().tpl != TPL_APPLICATION {
        return UNSUPPORTED;
    }
    let list = unsafe { core::slice::from_raw_parts(events, count) };
    if list
        .iter()
        .any(|&event| !valid(event) || data(event).kind & EVT_NOTIFY_SIGNAL != 0)
    {
        return INVALID_PARAMETER;
    }
    loop {
        for (position, &event) in list.iter().enumerate() {
            match check_event(event) {
                NOT_READY => {}
                status => {
                    unsafe { index.write(position) };
                    return status;
                }
            }
        }
        cpu::idle_wait(WAIT_SLICE_NS);
    }
}

extern "efiapi" fn raise_tpl(new: usize) -> usize {
    let state = STATE.get();
    let old = state.tpl;
    if new >= old {
        state.tpl = new;
    }
    if new >= TPL_HIGH_LEVEL {
        cpu::disable_interrupts();
    }
    old
}

/// Whether `restore_tpl` is already firing timers further up the stack: their notification
/// functions raise and restore the TPL themselves.
static POLLING: Global<bool> = Global::new(false);

extern "efiapi" fn restore_tpl(old: usize) {
    STATE.get().tpl = old;
    if old < TPL_HIGH_LEVEL {
        cpu::enable_interrupts();
    }
    // Where a real firmware's tick would deliver timer notifications: an application that drops
    // its TPL to let time pass (iPXE's `currticks`) must see its periodic timer fire.
    if old < TPL_CALLBACK && !*POLLING.get() {
        *POLLING.get() = true;
        poll();
        *POLLING.get() = false;
    }
    run_pending();
}

pub fn install(services: &mut BootServices) {
    services.raise_tpl = raise_tpl;
    services.restore_tpl = restore_tpl;
    services.create_event = create_event;
    services.create_event_ex = create_event_ex;
    services.set_timer = set_timer;
    services.signal_event = signal_event;
    services.close_event = close_event;
    services.check_event = check_event;
    services.wait_for_event = wait_for_event;
}
