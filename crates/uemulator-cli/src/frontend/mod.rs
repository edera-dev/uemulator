//! Frontends show a machine's screen and feed it keyboard and mouse input. Each one only sees a
//! `Screen`, the library's public display and input handles, so adding one (a native window
//! next to the web viewer) touches neither the library nor `run.rs` beyond choosing it.

pub mod web;

use uemulator::{DisplayHandle, InputHandle};

/// What a frontend gets of the machine: its screen, its keyboard and tablet, or both (a machine
/// with neither has no frontend).
#[derive(Clone)]
pub struct Screen {
    pub display: Option<DisplayHandle>,
    pub input: Option<InputHandle>,
}

/// A way to show the screen. `start` returns once the frontend is up and runs it on threads of
/// its own until the process ends, so the caller goes on to run the machine. A frontend that
/// must own the main thread (a native window on macOS) would get a `run` of its own instead;
/// the `Screen` stays the whole interface.
pub trait Frontend {
    /// Starts showing `screen`; the message says where to find it (the viewer's URL).
    fn start(self: Box<Self>, screen: Screen) -> Result<String, String>;
}
