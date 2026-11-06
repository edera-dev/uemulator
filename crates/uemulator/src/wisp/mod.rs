//! Wisp v1 (MercuryWorkshop/wisp-protocol): a WebSocket relay that carries the gateway's TCP and
//! UDP streams to the internet from a browser.

// The Wisp client (not written yet) is the only user of the codec outside its tests, and the tests do not
// touch every close reason.
#[allow(dead_code)]
mod packet;
#[cfg(test)]
mod tests;
