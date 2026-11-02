//! The built-in web viewer: a small HTTP server that serves one page (`web/viewer.html`), a
//! WebSocket at `/ws` and the whole screen as `/screenshot.png`. Over the socket the server sends
//! screen updates as PNG rectangles and the guest's pointer image, and the page sends keyboard
//! and mouse events and size requests as short text commands. Each viewer has a `ScreenMirror`
//! of what its page shows, so only tiles whose pixels changed are encoded, in bands on every
//! core. One update is in flight per viewer at a time (the page acknowledges each after drawing
//! it), so a slow browser or link gets fewer, larger updates instead of a growing backlog.
//! Protocol in `docs/plans/24-display-input.md`.

mod png;
mod websocket;

use std::io::{self, BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use uemulator::{Cursor, PointerButton, Rect, ScreenMirror};

use super::{Frontend, Screen};
use websocket::{
    MessageReader, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG, OPCODE_TEXT, write_frame,
};

const VIEWER_PAGE: &str = include_str!("web/viewer.html");
const SOCKET_PATH: &str = "/ws";
const SCREENSHOT_PATH: &str = "/screenshot.png";
/// The longest request head accepted; the page's requests are a few hundred bytes.
const MAX_REQUEST_HEAD: usize = 16 * 1024;
/// A client that does not finish its request head in this time is dropped.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// How long the update loop waits for the screen to change before it checks that the viewer is
/// still there.
const UPDATE_WAIT: Duration = Duration::from_millis(500);
/// At most this many updates per second per viewer: what a browser shows anyway.
const MIN_UPDATE_INTERVAL: Duration = Duration::from_millis(16);
/// Rectangles are encoded in bands of at most this many rows, the bands on as many threads as
/// the host has: deflate is nine tenths of an encode, and a full 1280x800 screen of photo-like
/// content takes 21 ms on one core, 3 ms in bands on fourteen.
const BAND_ROWS: u32 = 64;
/// An update smaller than this many pixels is encoded on the session's own thread: starting
/// threads would cost more than they save.
const PARALLEL_MIN_PIXELS: usize = 256 * 256;
/// Screen update: kind u8, then screen width, screen height and the number of rectangles as u16
/// little endian, then for each rectangle x, y, width and height as u16, the length of its PNG
/// as u32 and the PNG. The page draws all of them before it acknowledges, so it never shows half
/// an update. Kind 1, a single rectangle, is retired: a page left open across an upgrade must
/// not misread this as one (it stops updating until reloaded).
const UPDATE_KIND_RECTS: u8 = 4;
const RECTS_HEADER_BYTES: usize = 7;
const RECT_HEADER_BYTES: usize = 12;
/// Pointer message: kind u8, state u8 (`CURSOR_*`), hot spot x and y as u16 little endian, then
/// for an image its PNG with alpha.
const UPDATE_KIND_CURSOR: u8 = 2;
const CURSOR_HOST: u8 = 0;
const CURSOR_HIDDEN: u8 = 1;
const CURSOR_IMAGE: u8 = 2;
/// The first message of a session: kind u8, flags u8 (`HELLO_*`), then the pointer area's width
/// and height as u16 little endian, so the page knows what to show before any screen update
/// (a machine with input and no screen gets none).
const UPDATE_KIND_HELLO: u8 = 3;
const HELLO_HAS_DISPLAY: u8 = 1 << 0;
const HELLO_HAS_INPUT: u8 = 1 << 1;
/// WebSocket close status "going away", sent when the server ends a session.
const CLOSE_GOING_AWAY: u16 = 1001;
const WEBSOCKET_VERSION: &str = "13";

pub struct WebViewer {
    listener: TcpListener,
    address: SocketAddr,
    /// The host name `bind` was given, when it was one: the URL uses it.
    name: Option<String>,
    /// Who the viewer is, for the `Host` check.
    host: AllowedHosts,
}

/// What a request's `Host` may name: the address the viewer is bound to (`host_allowed` has the
/// rules for loopback and every-address listeners) and the host names the user gave, lowercase.
#[derive(Clone, Debug)]
struct AllowedHosts {
    bound: SocketAddr,
    names: Vec<String>,
}

impl WebViewer {
    /// Binds `address` (`HOST:PORT`; port 0 picks a free one). HOST may be a host name, which is
    /// resolved for the bind and then allowed in `Host` like `allowed_names` (names a reverse
    /// proxy or DNS gives the viewer).
    pub fn bind(address: &str, allowed_names: &[String]) -> Result<WebViewer, String> {
        let listener =
            TcpListener::bind(address).map_err(|e| format!("viewer: bind {address}: {e}"))?;
        let bound = listener
            .local_addr()
            .map_err(|e| format!("viewer: {address}: {e}"))?;
        let name = address
            .rsplit_once(':')
            .map(|(host, _)| host)
            .filter(|host| is_host_name(host))
            .map(str::to_ascii_lowercase);
        let names = allowed_names
            .iter()
            .map(|name| name.to_ascii_lowercase())
            .chain(name.clone())
            .collect();
        Ok(WebViewer {
            listener,
            address: bound,
            name,
            host: AllowedHosts { bound, names },
        })
    }

    pub fn url(&self) -> String {
        match &self.name {
            Some(name) => format!("http://{name}:{}/", self.address.port()),
            None => format!("http://{}/", self.address),
        }
    }
}

/// Whether `host` (from `HOST:PORT` or a `Host` header) is a name rather than an address
/// literal. `localhost` counts as an address: `host_allowed` treats it as loopback.
fn is_host_name(host: &str) -> bool {
    !host.is_empty()
        && !host.starts_with('[')
        && !host.eq_ignore_ascii_case("localhost")
        && host.parse::<IpAddr>().is_err()
}

impl Frontend for WebViewer {
    fn start(self: Box<Self>, screen: Screen) -> Result<String, String> {
        let mut message = format!("viewer: {}", self.url());
        if !self.address.ip().is_loopback() {
            message.push_str(
                " (reachable from the network: anyone who can open it controls the machine)",
            );
        }
        let (listener, host) = (self.listener, self.host);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (screen, host) = (screen.clone(), host.clone());
                std::thread::spawn(move || {
                    // A failed connection only ends that viewer.
                    let _ = serve_connection(stream, &host, &screen);
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
    Screenshot,
    /// A WebSocket handshake, with the `Sec-WebSocket-Accept` value to answer.
    Upgrade(String),
    BadRequest,
    /// The request is not from the viewer's own page. A browser sends any site's requests to
    /// 127.0.0.1, so without this a web page could type into the machine: a request must name
    /// the viewer's address, or a host name the user gave it, as its `Host` (which defeats DNS
    /// rebinding, where another site's name resolves to 127.0.0.1 and `Origin` and `Host`
    /// agree), and a socket's `Origin` must be that same host, over http or https (a TLS proxy).
    Forbidden,
    NotFound,
    MethodNotAllowed,
}

/// Splits a `Host` header into its name and its port, if it has one: `name`, `name:port`,
/// `[v6]`, `[v6]:port`.
fn split_host(host: &str) -> (&str, Option<&str>) {
    if let Some(rest) = host.strip_prefix('[') {
        return match rest.split_once(']') {
            Some((name, tail)) => (name, tail.strip_prefix(':')),
            None => (host, None),
        };
    }
    match host.rsplit_once(':') {
        Some((name, port)) => (name, Some(port)),
        None => (host, None),
    }
}

/// Whether `host` (a `Host` header) names the viewer: one of the host names the user gave it,
/// on any port or none (a reverse proxy on 443 forwards `Host: name`); or the bound address and
/// port, or for a loopback listener any loopback name, or for one on every address any address
/// literal. Never another name: that is what a rebinding site's requests carry.
fn host_allowed(host: &str, allowed: &AllowedHosts) -> bool {
    let (name, port) = split_host(host);
    if allowed
        .names
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(name))
    {
        return port.is_none_or(|port| port.parse::<u16>().is_ok());
    }
    let bound = allowed.bound;
    if port.and_then(|port| port.parse::<u16>().ok()) != Some(bound.port()) {
        return false;
    }
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

fn route(request: &Request, allowed: &AllowedHosts) -> Route {
    if request.method != "GET" {
        return Route::MethodNotAllowed;
    }
    let Some(host) = request
        .header("host")
        .filter(|host| host_allowed(host, allowed))
    else {
        return Route::Forbidden;
    };
    match request.path.as_str() {
        "/" | "/index.html" => Route::Page,
        SCREENSHOT_PATH => Route::Screenshot,
        SOCKET_PATH => {
            let upgrade = request.header_lists("upgrade", "websocket")
                && request.header_lists("connection", "upgrade")
                && request.header("sec-websocket-version") == Some(WEBSOCKET_VERSION);
            let Some(key) = request.header("sec-websocket-key").filter(|_| upgrade) else {
                return Route::BadRequest;
            };
            // No `Origin`: not a browser, nothing to protect against.
            let same_origin = request.header("origin").is_none_or(|origin| {
                ["http", "https"]
                    .iter()
                    .any(|scheme| origin.eq_ignore_ascii_case(&format!("{scheme}://{host}")))
            });
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

fn serve_connection(stream: TcpStream, allowed: &AllowedHosts, screen: &Screen) -> io::Result<()> {
    stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let Some(request) = read_request(&mut reader)? else {
        return Ok(());
    };
    let plain = "text/plain; charset=utf-8";
    match route(&request, allowed) {
        Route::Page => write_response(
            &mut writer,
            "200 OK",
            "text/html; charset=utf-8",
            VIEWER_PAGE.as_bytes(),
        ),
        Route::Screenshot => match &screen.display {
            Some(display) => {
                let frame = display.frame();
                let png = png::encode_bgrx(frame.width, frame.height, &frame.pixels);
                write_response(&mut writer, "200 OK", "image/png", &png)
            }
            None => write_response(&mut writer, "404 Not Found", plain, b"no screen\n"),
        },
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
    /// `s WIDTH HEIGHT`: ask the guest for this screen size (`DisplayHandle::request_size`).
    Size { width: u32, height: u32 },
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
        "s" => Command::Size {
            width: u32::try_from(number()?).ok()?,
            height: u32::try_from(number()?).ok()?,
        },
        "a" => Command::Ack,
        _ => return None,
    };
    Some(command)
}

fn apply(command: Command, screen: &Screen) {
    if let Command::Size { width, height } = command {
        if let Some(display) = &screen.display {
            display.request_size(width, height);
        }
        return;
    }
    let Some(input) = &screen.input else {
        return;
    };
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
        Command::Size { .. } | Command::Ack => {}
    }
}

/// The session's first message (`UPDATE_KIND_HELLO`).
fn hello_message(screen: &Screen) -> Vec<u8> {
    let (width, height) = match (&screen.display, &screen.input) {
        (Some(display), _) => display.size(),
        (None, Some(input)) => input.pointer_area(),
        (None, None) => (0, 0),
    };
    let mut flags = 0;
    if screen.display.is_some() {
        flags |= HELLO_HAS_DISPLAY;
    }
    if screen.input.is_some() {
        flags |= HELLO_HAS_INPUT;
    }
    let mut message = vec![UPDATE_KIND_HELLO, flags];
    message.extend_from_slice(&(width as u16).to_le_bytes());
    message.extend_from_slice(&(height as u16).to_le_bytes());
    message
}

/// The message that shows `cursor` (`UPDATE_KIND_CURSOR`).
fn cursor_message(cursor: &Cursor) -> Vec<u8> {
    let (state, hot_x, hot_y) = match cursor {
        Cursor::Host => (CURSOR_HOST, 0, 0),
        Cursor::Hidden => (CURSOR_HIDDEN, 0, 0),
        Cursor::Image(image) => (CURSOR_IMAGE, image.hot_x, image.hot_y),
    };
    let mut message = vec![UPDATE_KIND_CURSOR, state];
    message.extend_from_slice(&(hot_x as u16).to_le_bytes());
    message.extend_from_slice(&(hot_y as u16).to_le_bytes());
    if let Cursor::Image(image) = cursor {
        message.extend_from_slice(&png::encode_bgra(image.width, image.height, &image.pixels));
    }
    message
}

/// `rects` cut into bands of at most `BAND_ROWS` rows, the units that are encoded in parallel.
fn bands(rects: &[Rect]) -> Vec<Rect> {
    let mut bands = Vec::new();
    for rect in rects {
        let mut top = rect.y;
        while top < rect.y + rect.height {
            let height = BAND_ROWS.min(rect.y + rect.height - top);
            bands.push(Rect {
                y: top,
                height,
                ..*rect
            });
            top += height;
        }
    }
    bands
}

/// Each band of a screen (`pixels`, rows `stride` bytes apart) as a PNG, in order; on several
/// threads when there is enough to do.
fn encode_bands(pixels: &[u8], stride: usize, bands: &[Rect]) -> Vec<Vec<u8>> {
    let encode = |band: &Rect| png::encode_bgrx_rect(pixels, stride, *band);
    let area: usize = bands
        .iter()
        .map(|band| band.width as usize * band.height as usize)
        .sum();
    let threads = std::thread::available_parallelism()
        .map_or(1, |count| count.get())
        .min(bands.len());
    if threads <= 1 || area < PARALLEL_MIN_PIXELS {
        return bands.iter().map(encode).collect();
    }
    let (encode, next) = (&encode, &AtomicUsize::new(0));
    let mut pngs = vec![Vec::new(); bands.len()];
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(band) = bands.get(index) else {
                            return done;
                        };
                        done.push((index, encode(band)));
                    }
                })
            })
            .collect();
        for worker in workers {
            for (index, png) in worker.join().expect("encoding cannot panic") {
                pngs[index] = png;
            }
        }
    });
    pngs
}

/// The message (`UPDATE_KIND_RECTS`) that shows the page `rects` of `mirror`.
fn update_message(mirror: &ScreenMirror, rects: &[Rect]) -> Vec<u8> {
    let bands = bands(rects);
    let pngs = encode_bands(mirror.pixels(), mirror.stride(), &bands);
    let (width, height) = mirror.size();
    let length = RECTS_HEADER_BYTES
        + pngs
            .iter()
            .map(|png| RECT_HEADER_BYTES + png.len())
            .sum::<usize>();
    let mut message = Vec::with_capacity(length);
    message.push(UPDATE_KIND_RECTS);
    for value in [width, height, bands.len() as u32] {
        message.extend_from_slice(&(value as u16).to_le_bytes());
    }
    for (band, png) in bands.iter().zip(&pngs) {
        for value in [band.x, band.y, band.width, band.height] {
            message.extend_from_slice(&(value as u16).to_le_bytes());
        }
        message.extend_from_slice(&(png.len() as u32).to_le_bytes());
        message.extend_from_slice(png);
    }
    message
}

/// Runs one viewer: input on a thread of its own, updates on this one.
fn run_session(
    mut reader: BufReader<TcpStream>,
    writer: TcpStream,
    screen: &Screen,
) -> io::Result<()> {
    let writer = Arc::new(Mutex::new(writer));
    let (acks, acknowledged) = mpsc::channel::<()>();
    let reader_screen = screen.clone();
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
                            Some(command) => apply(command, &reader_screen),
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
        if let Some(input) = &reader_screen.input {
            input.release_all();
        }
        let mut stream = reader_writer.lock().unwrap();
        let _ = write_frame(&mut *stream, OPCODE_CLOSE, &CLOSE_GOING_AWAY.to_be_bytes());
        let _ = stream.shutdown(std::net::Shutdown::Both);
    });

    send(&writer, &hello_message(screen))?;
    let Some(display) = &screen.display else {
        // Input only: nothing to send; stay until the viewer leaves.
        loop {
            match acknowledged.recv() {
                Ok(()) => continue,
                Err(_) => return Ok(()),
            }
        }
    };
    let (mut cursor_generation, cursor) = display.cursor();
    send(&writer, &cursor_message(&cursor))?;
    // What the page shows; a new one syncs as the whole screen.
    let mut mirror = ScreenMirror::new();
    let whole = display.sync_mirror(&mut mirror).unwrap_or_default();
    send(&writer, &update_message(&mirror, &whole))?;
    let mut last_sent = Instant::now();
    loop {
        // One update in flight: wait until the page has drawn the last one.
        match acknowledged.recv_timeout(UPDATE_WAIT) {
            Ok(()) => {}
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        }
        // Then until the screen differs from what the page shows. A pointer change ends the
        // wait too; it goes out at once, outside the one-in-flight rule (it is a few hundred
        // bytes and the page draws nothing for it).
        loop {
            if matches!(
                acknowledged.try_recv(),
                Err(mpsc::TryRecvError::Disconnected)
            ) {
                return Ok(());
            }
            let shown = mirror.generation().unwrap_or(0);
            let now = display.wait_for_update(shown, UPDATE_WAIT);
            let (current, cursor) = display.cursor();
            if current != cursor_generation {
                cursor_generation = current;
                send(&writer, &cursor_message(&cursor))?;
            }
            if now <= shown {
                continue;
            }
            // Let changes that come right after each other go out as one update.
            let since = last_sent.elapsed();
            if since < MIN_UPDATE_INTERVAL {
                std::thread::sleep(MIN_UPDATE_INTERVAL - since);
            }
            // A flush that left every pixel as it was has nothing for the page.
            match display.sync_mirror(&mut mirror) {
                Some(rects) if !rects.is_empty() => {
                    send(&writer, &update_message(&mirror, &rects))?;
                    last_sent = Instant::now();
                    break;
                }
                _ => {}
            }
        }
    }
}

fn send(writer: &Mutex<TcpStream>, message: &[u8]) -> io::Result<()> {
    let mut stream = writer.lock().unwrap();
    write_frame(&mut *stream, OPCODE_BINARY, message)
}

#[cfg(test)]
mod tests;
