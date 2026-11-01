//! The built-in web viewer: a small HTTP server that serves one page (`web/viewer.html`) and a
//! WebSocket at `/ws`. Over the socket the server sends screen updates as PNG rectangles and the
//! page sends keyboard and mouse events as short text commands. One update is in flight per
//! viewer at a time (the page acknowledges each after drawing it), so a slow browser or link
//! gets fewer, larger updates instead of a growing backlog. Protocol in
//! `docs/plans/24-display-input.md`.

mod png;
mod websocket;

use std::io::{self, BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uemulator::{BYTES_PER_PIXEL, DisplayHandle, FrameUpdate, InputHandle, PointerButton, Rect};

use super::{Frontend, Screen};
use websocket::{
    MessageReader, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG, OPCODE_TEXT, write_frame,
};

const VIEWER_PAGE: &str = include_str!("web/viewer.html");
const SOCKET_PATH: &str = "/ws";
/// The longest request head accepted; the page's requests are a few hundred bytes.
const MAX_REQUEST_HEAD: usize = 16 * 1024;
/// A client that does not finish its request head in this time is dropped.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the update loop waits for the screen to change before it checks that the viewer is
/// still there.
const UPDATE_WAIT: Duration = Duration::from_millis(500);
/// At most this many updates per second per viewer: what a browser shows anyway.
const MIN_UPDATE_INTERVAL: Duration = Duration::from_millis(16);
/// Update message: kind u8, then screen width, screen height, x, y, width, height as u16 little
/// endian, then the PNG of the rectangle.
const UPDATE_KIND_RECT: u8 = 1;
const UPDATE_HEADER_BYTES: usize = 13;
/// WebSocket close status "going away", sent when the server ends a session.
const CLOSE_GOING_AWAY: u16 = 1001;
const WEBSOCKET_VERSION: &str = "13";

pub struct WebViewer {
    listener: TcpListener,
    address: SocketAddr,
}

impl WebViewer {
    /// Binds `address` (`HOST:PORT`; port 0 picks a free one).
    pub fn bind(address: &str) -> Result<WebViewer, String> {
        let listener =
            TcpListener::bind(address).map_err(|e| format!("viewer: bind {address}: {e}"))?;
        let address = listener
            .local_addr()
            .map_err(|e| format!("viewer: {address}: {e}"))?;
        Ok(WebViewer { listener, address })
    }

    pub fn url(&self) -> String {
        format!("http://{}/", self.address)
    }
}

impl Frontend for WebViewer {
    fn start(self: Box<Self>, screen: Screen) -> Result<String, String> {
        let mut message = format!("viewer: {}", self.url());
        if !self.address.ip().is_loopback() {
            message.push_str(
                " (reachable from the network: anyone who can open it controls the machine)",
            );
        }
        let (listener, bound) = (self.listener, self.address);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let screen = screen.clone();
                std::thread::spawn(move || {
                    // A failed connection only ends that viewer.
                    let _ = serve_connection(stream, bound, &screen);
                });
            }
        });
        Ok(message)
    }
}

/// A parsed HTTP request head.
#[derive(Debug, Default, PartialEq, Eq)]
struct Request {
    method: String,
    path: String,
    /// Names lowercased, values trimmed.
    headers: Vec<(String, String)>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// Whether a comma-separated header lists `token` (case-insensitive), as `Connection:
    /// keep-alive, Upgrade` does.
    fn header_lists(&self, name: &str, token: &str) -> bool {
        self.header(name).is_some_and(|value| {
            value
                .split(',')
                .any(|item| item.trim().eq_ignore_ascii_case(token))
        })
    }
}

/// Reads a request head (up to the empty line). `None` when the client closed the connection
/// first; an error when it is malformed or too long.
fn read_request(reader: &mut impl BufRead) -> io::Result<Option<Request>> {
    let mut lines = Vec::new();
    let mut total = 0;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return if lines.is_empty() {
                Ok(None)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "request head cut short",
                ))
            };
        }
        total += read;
        if total > MAX_REQUEST_HEAD {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request head too long",
            ));
        }
        let line = line.trim_end_matches(['\r', '\n']).to_string();
        if line.is_empty() {
            break;
        }
        lines.push(line);
    }
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "malformed request line");
    let mut request_line = lines.first().ok_or_else(bad)?.split_whitespace();
    let method = request_line.next().ok_or_else(bad)?.to_string();
    let target = request_line.next().ok_or_else(bad)?;
    // The query string does not choose anything here.
    let path = target.split('?').next().unwrap_or(target).to_string();
    let headers = lines[1..]
        .iter()
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    Ok(Some(Request {
        method,
        path,
        headers,
    }))
}

/// What to do with a request.
#[derive(Debug, PartialEq, Eq)]
enum Route {
    Page,
    /// A WebSocket handshake, with the `Sec-WebSocket-Accept` value to answer.
    Upgrade(String),
    BadRequest,
    /// The request is not from the viewer's own page. A browser sends any site's requests to
    /// 127.0.0.1, so without this a web page could type into the machine: a request must name
    /// the viewer's address as its `Host` (which defeats DNS rebinding, where another site's
    /// name resolves to 127.0.0.1 and `Origin` and `Host` agree), and a socket's `Origin` must
    /// be that same host.
    Forbidden,
    NotFound,
    MethodNotAllowed,
}

/// Whether `host` (a `Host` header) names the viewer bound at `bound`: its address and port, or
/// for a loopback listener any loopback name, or for one on every address any address literal.
/// Never another name: that is what a rebinding site's requests carry.
fn host_allowed(host: &str, bound: SocketAddr) -> bool {
    let Some((name, port)) = host.rsplit_once(':') else {
        return false;
    };
    if port.parse::<u16>().ok() != Some(bound.port()) {
        return false;
    }
    let name = name
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
        .unwrap_or(name);
    let listener = bound.ip();
    if name.eq_ignore_ascii_case("localhost") {
        return listener.is_loopback() || listener.is_unspecified();
    }
    name.parse::<IpAddr>().is_ok_and(|address| {
        address == listener
            || listener.is_unspecified()
            || (listener.is_loopback() && address.is_loopback())
    })
}

fn route(request: &Request, bound: SocketAddr) -> Route {
    if request.method != "GET" {
        return Route::MethodNotAllowed;
    }
    let Some(host) = request
        .header("host")
        .filter(|host| host_allowed(host, bound))
    else {
        return Route::Forbidden;
    };
    match request.path.as_str() {
        "/" | "/index.html" => Route::Page,
        SOCKET_PATH => {
            let upgrade = request.header_lists("upgrade", "websocket")
                && request.header_lists("connection", "upgrade")
                && request.header("sec-websocket-version") == Some(WEBSOCKET_VERSION);
            let Some(key) = request.header("sec-websocket-key").filter(|_| upgrade) else {
                return Route::BadRequest;
            };
            // No `Origin`: not a browser, nothing to protect against.
            let same_origin = request
                .header("origin")
                .is_none_or(|origin| origin.eq_ignore_ascii_case(&format!("http://{host}")));
            if same_origin {
                Route::Upgrade(websocket::accept_key(key))
            } else {
                Route::Forbidden
            }
        }
        _ => Route::NotFound,
    }
}

fn write_response(
    stream: &mut impl Write,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

fn serve_connection(stream: TcpStream, bound: SocketAddr, screen: &Screen) -> io::Result<()> {
    stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let Some(request) = read_request(&mut reader)? else {
        return Ok(());
    };
    let plain = "text/plain; charset=utf-8";
    match route(&request, bound) {
        Route::Page => write_response(
            &mut writer,
            "200 OK",
            "text/html; charset=utf-8",
            VIEWER_PAGE.as_bytes(),
        ),
        Route::Upgrade(accept) => {
            write!(
                writer,
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\
                 Connection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
            )?;
            writer.flush()?;
            writer.set_read_timeout(None)?;
            run_session(reader, writer, screen)
        }
        Route::BadRequest => {
            write_response(&mut writer, "400 Bad Request", plain, b"bad request\n")
        }
        Route::Forbidden => write_response(&mut writer, "403 Forbidden", plain, b"forbidden\n"),
        Route::NotFound => write_response(&mut writer, "404 Not Found", plain, b"not found\n"),
        Route::MethodNotAllowed => {
            write_response(&mut writer, "405 Method Not Allowed", plain, b"GET only\n")
        }
    }
}

/// What the page sends.
#[derive(Debug, PartialEq, Eq)]
enum Command {
    /// `k USAGE 0|1`: a key by USB HID usage, released or pressed.
    Key { usage: u16, pressed: bool },
    /// `m X Y`: the pointer, in screen pixels.
    Move { x: u32, y: u32 },
    /// `b INDEX 0|1`: a button, numbered as `PointerButton` lists them.
    Button {
        button: PointerButton,
        pressed: bool,
    },
    /// `w CLICKS`: the wheel, positive away from the user.
    Wheel { clicks: i32 },
    /// `r`: let go of everything (the page lost the focus).
    ReleaseAll,
    /// `a`: the last update is drawn; the next may come.
    Ack,
}

fn parse_command(text: &str) -> Option<Command> {
    let mut words = text.split_ascii_whitespace();
    let kind = words.next()?;
    let mut number = || words.next()?.parse::<i64>().ok();
    let command = match kind {
        "k" => Command::Key {
            usage: u16::try_from(number()?).ok()?,
            pressed: number()? != 0,
        },
        "m" => Command::Move {
            x: u32::try_from(number()?).ok()?,
            y: u32::try_from(number()?).ok()?,
        },
        "b" => {
            const BUTTONS: [PointerButton; 5] = [
                PointerButton::Left,
                PointerButton::Right,
                PointerButton::Middle,
                PointerButton::Side,
                PointerButton::Extra,
            ];
            Command::Button {
                button: *BUTTONS.get(usize::try_from(number()?).ok()?)?,
                pressed: number()? != 0,
            }
        }
        "w" => Command::Wheel {
            clicks: i32::try_from(number()?).ok()?,
        },
        "r" => Command::ReleaseAll,
        "a" => Command::Ack,
        _ => return None,
    };
    Some(command)
}

fn apply(command: Command, input: &InputHandle) {
    match command {
        Command::Key { usage, pressed } => {
            input.key_usb(usage, pressed);
        }
        Command::Move { x, y } => {
            input.pointer_move(x, y);
        }
        Command::Button { button, pressed } => {
            input.pointer_button(button, pressed);
        }
        Command::Wheel { clicks } => {
            input.scroll(clicks);
        }
        Command::ReleaseAll => input.release_all(),
        Command::Ack => {}
    }
}

/// The binary message for one update.
fn update_message(update: &FrameUpdate, pixels: &[u8]) -> Vec<u8> {
    let png = png::encode_bgrx(update.rect.width, update.rect.height, pixels);
    let mut message = Vec::with_capacity(UPDATE_HEADER_BYTES + png.len());
    message.push(UPDATE_KIND_RECT);
    for value in [
        update.width,
        update.height,
        update.rect.x,
        update.rect.y,
        update.rect.width,
        update.rect.height,
    ] {
        message.extend_from_slice(&(value as u16).to_le_bytes());
    }
    message.extend_from_slice(&png);
    message
}

/// The whole screen as one update, for a viewer that just connected.
fn full_update(display: &DisplayHandle) -> (FrameUpdate, Vec<u8>) {
    let frame = display.frame();
    let update = FrameUpdate {
        generation: frame.generation,
        width: frame.width,
        height: frame.height,
        rect: Rect {
            x: 0,
            y: 0,
            width: frame.width,
            height: frame.height,
        },
    };
    debug_assert_eq!(
        frame.pixels.len(),
        frame.width as usize * frame.height as usize * BYTES_PER_PIXEL
    );
    (update, frame.pixels)
}

/// Runs one viewer: input on a thread of its own, updates on this one.
fn run_session(
    mut reader: BufReader<TcpStream>,
    writer: TcpStream,
    screen: &Screen,
) -> io::Result<()> {
    let writer = Arc::new(Mutex::new(writer));
    let (acks, acknowledged) = mpsc::channel::<()>();
    let input = screen.input.clone();
    let reader_writer = writer.clone();
    // Dropping `acks` when the reader ends tells the update loop the viewer is gone.
    std::thread::spawn(move || {
        let mut messages = MessageReader::default();
        while let Ok(message) = messages.read(&mut reader) {
            match message.opcode {
                OPCODE_TEXT => {
                    let Ok(text) = std::str::from_utf8(&message.payload) else {
                        continue;
                    };
                    for line in text.lines() {
                        match parse_command(line) {
                            Some(Command::Ack) => {
                                let _ = acks.send(());
                            }
                            Some(command) => apply(command, &input),
                            None => {}
                        }
                    }
                }
                OPCODE_PING => {
                    let mut stream = reader_writer.lock().unwrap();
                    let _ = write_frame(&mut *stream, OPCODE_PONG, &message.payload);
                }
                OPCODE_CLOSE => break,
                _ => {}
            }
        }
        // Keys held when the tab closed would otherwise stay down in the guest.
        input.release_all();
        let mut stream = reader_writer.lock().unwrap();
        let _ = write_frame(&mut *stream, OPCODE_CLOSE, &CLOSE_GOING_AWAY.to_be_bytes());
        let _ = stream.shutdown(std::net::Shutdown::Both);
    });

    let display = &screen.display;
    let (update, pixels) = full_update(display);
    let mut generation = update.generation;
    send(&writer, &update_message(&update, &pixels))?;
    let mut last_sent = Instant::now();
    let mut pixels = Vec::new();
    loop {
        // One update in flight: wait until the page has drawn the last one.
        match acknowledged.recv_timeout(UPDATE_WAIT) {
            Ok(()) => {}
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
        while display.wait_for_update(generation, UPDATE_WAIT) <= generation {
            if matches!(
                acknowledged.try_recv(),
                Err(mpsc::TryRecvError::Disconnected)
            ) {
                return Ok(());
            }
        }
        // Let changes that come right after each other go out as one update.
        let since = last_sent.elapsed();
        if since < MIN_UPDATE_INTERVAL {
            std::thread::sleep(MIN_UPDATE_INTERVAL - since);
        }
        let Some(update) = display.update_since(generation, &mut pixels) else {
            continue;
        };
        generation = update.generation;
        send(&writer, &update_message(&update, &pixels))?;
        last_sent = Instant::now();
    }
}

fn send(writer: &Mutex<TcpStream>, message: &[u8]) -> io::Result<()> {
    let mut stream = writer.lock().unwrap();
    write_frame(&mut *stream, OPCODE_BINARY, message)
}

#[cfg(test)]
mod tests;
