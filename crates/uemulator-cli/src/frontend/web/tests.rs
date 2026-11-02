use super::*;
use std::io::{Cursor, Read};
use uemulator::{DisplayHandle, InputHandle};

const KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";
const ACCEPT: &str = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";
const IO_TIMEOUT: Duration = Duration::from_secs(10);

fn parse(head: &str) -> Request {
    read_request(&mut Cursor::new(head.as_bytes()))
        .unwrap()
        .unwrap()
}

/// A viewer bound to `address` that answers to `names` as well.
fn viewer_at(address: &str, names: &[&str]) -> AllowedHosts {
    AllowedHosts {
        bound: address.parse().unwrap(),
        names: names.iter().map(|name| name.to_string()).collect(),
    }
}

/// Where the requests in these tests were sent: the default viewer address.
fn default_viewer() -> AllowedHosts {
    viewer_at("127.0.0.1:5800", &[])
}

fn route_default(request: &Request) -> Route {
    route(request, &default_viewer())
}

fn upgrade_request(origin: Option<&str>) -> String {
    upgrade_request_to("127.0.0.1:5800", origin)
}

/// A browser's WebSocket handshake for the viewer at `host`.
fn upgrade_request_to(host: &str, origin: Option<&str>) -> String {
    let mut head = format!(
        "GET /ws HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n\
         Connection: keep-alive, Upgrade\r\nSec-WebSocket-Key: {KEY}\r\n\
         Sec-WebSocket-Version: 13\r\n"
    );
    if let Some(origin) = origin {
        head.push_str(&format!("Origin: {origin}\r\n"));
    }
    head.push_str("\r\n");
    head
}

#[test]
fn a_request_head_is_split_into_method_path_and_lowercased_headers() {
    let request =
        parse("GET /index.html?x=1 HTTP/1.1\r\nHost: a:1\r\nX-Thing:  value \r\n\r\nbody");
    assert_eq!(request.method, "GET");
    assert_eq!(request.path, "/index.html");
    assert_eq!(request.header("host"), Some("a:1"));
    assert_eq!(request.header("x-thing"), Some("value"));
    assert_eq!(request.header("missing"), None);
}

#[test]
fn a_closed_connection_is_no_request_and_a_cut_one_is_an_error() {
    assert_eq!(read_request(&mut Cursor::new(b"")).unwrap(), None);
    assert!(read_request(&mut Cursor::new(b"GET / HTTP/1.1\r\nHost: x\r\n")).is_err());
    assert!(
        read_request(&mut Cursor::new(b"\r\n")).is_err(),
        "no request line"
    );
    let endless = format!("GET / HTTP/1.1\r\n{}", "X: y\r\n".repeat(MAX_REQUEST_HEAD));
    assert!(read_request(&mut Cursor::new(endless.as_bytes())).is_err());
}

#[test]
fn the_page_the_screenshot_and_the_socket_are_the_only_routes() {
    assert_eq!(
        route_default(&parse("GET / HTTP/1.1\r\nHost: 127.0.0.1:5800\r\n\r\n")),
        Route::Page
    );
    assert_eq!(
        route_default(&parse(
            "GET /index.html HTTP/1.1\r\nHost: localhost:5800\r\n\r\n"
        )),
        Route::Page
    );
    assert_eq!(
        route_default(&parse(
            "GET /screenshot.png HTTP/1.1\r\nHost: 127.0.0.1:5800\r\n\r\n"
        )),
        Route::Screenshot
    );
    assert_eq!(
        route_default(&parse(
            "GET /etc/passwd HTTP/1.1\r\nHost: 127.0.0.1:5800\r\n\r\n"
        )),
        Route::NotFound
    );
    assert_eq!(
        route_default(&parse("POST / HTTP/1.1\r\n\r\n")),
        Route::MethodNotAllowed
    );
    assert_eq!(
        route_default(&parse(&upgrade_request(None))),
        Route::Upgrade(ACCEPT.to_string())
    );
}

#[test]
fn the_socket_needs_a_complete_handshake() {
    assert_eq!(
        route_default(&parse("GET /ws HTTP/1.1\r\nHost: 127.0.0.1:5800\r\n\r\n")),
        Route::BadRequest
    );
    let old_version = upgrade_request(None).replace("Version: 13", "Version: 8");
    assert_eq!(route_default(&parse(&old_version)), Route::BadRequest);
    let no_upgrade = upgrade_request(None).replace("keep-alive, Upgrade", "keep-alive");
    assert_eq!(route_default(&parse(&no_upgrade)), Route::BadRequest);
}

#[test]
fn only_the_viewer_page_itself_may_open_the_socket() {
    assert_eq!(
        route_default(&parse(&upgrade_request(Some("http://127.0.0.1:5800")))),
        Route::Upgrade(ACCEPT.to_string())
    );
    assert_eq!(
        route_default(&parse(&upgrade_request(Some("https://evil.example")))),
        Route::Forbidden
    );
    assert_eq!(
        route_default(&parse(&upgrade_request(Some("http://127.0.0.1:5801")))),
        Route::Forbidden
    );
    // A TLS proxy in front: the page is https, the host the same.
    assert_eq!(
        route_default(&parse(&upgrade_request(Some("https://127.0.0.1:5800")))),
        Route::Upgrade(ACCEPT.to_string())
    );
}

#[test]
fn a_name_the_user_gave_the_viewer_is_allowed_on_any_port_and_others_still_are_not() {
    let named = viewer_at("0.0.0.0:5800", &["vm.example"]);
    for host in [
        "vm.example:5800",
        "VM.EXAMPLE:5800",
        "vm.example",
        "vm.example:443",
    ] {
        assert!(host_allowed(host, &named), "{host}");
    }
    for host in ["evil.example:5800", "vm.example.evil:5800", "vm.example:x"] {
        assert!(!host_allowed(host, &named), "{host}");
    }
    let proxied = upgrade_request_to("vm.example", Some("https://vm.example"));
    assert_eq!(
        route(&parse(&proxied), &named),
        Route::Upgrade(ACCEPT.to_string())
    );
    let rebound = upgrade_request_to("evil.example:5800", Some("http://evil.example:5800"));
    assert_eq!(route(&parse(&rebound), &named), Route::Forbidden);
    let crossed = upgrade_request_to("vm.example:5800", Some("http://evil.example:5800"));
    assert_eq!(route(&parse(&crossed), &named), Route::Forbidden);
}

#[test]
fn a_host_header_splits_into_name_and_port() {
    assert_eq!(split_host("a:1"), ("a", Some("1")));
    assert_eq!(split_host("a"), ("a", None));
    assert_eq!(split_host("[::1]:5800"), ("::1", Some("5800")));
    assert_eq!(split_host("[::1]"), ("::1", None));
}

#[test]
fn a_rebinding_site_is_refused_even_when_origin_and_host_agree() {
    let rebound = upgrade_request(Some("http://evil.example:5800"))
        .replace("Host: 127.0.0.1:5800", "Host: evil.example:5800");
    assert_eq!(route_default(&parse(&rebound)), Route::Forbidden);
    let page = parse("GET / HTTP/1.1\r\nHost: evil.example:5800\r\n\r\n");
    assert_eq!(route_default(&page), Route::Forbidden, "the page too");
    assert_eq!(
        route_default(&parse("GET / HTTP/1.1\r\n\r\n")),
        Route::Forbidden,
        "no Host"
    );
}

#[test]
fn the_host_must_name_the_viewer() {
    let loopback = &default_viewer();
    for host in [
        "127.0.0.1:5800",
        "localhost:5800",
        "LOCALHOST:5800",
        "[::1]:5800",
        "127.0.0.2:5800",
    ] {
        assert!(host_allowed(host, loopback), "{host}");
    }
    for host in [
        "127.0.0.1:5801",
        "127.0.0.1",
        "evil.example:5800",
        "10.0.0.1:5800",
        "[::1]",
    ] {
        assert!(!host_allowed(host, loopback), "{host}");
    }
    let lan = &viewer_at("192.168.1.5:5800", &[]);
    assert!(host_allowed("192.168.1.5:5800", lan));
    assert!(!host_allowed("localhost:5800", lan));
    assert!(!host_allowed("192.168.1.6:5800", lan));
    let every = &viewer_at("0.0.0.0:5800", &[]);
    assert!(host_allowed("10.1.2.3:5800", every));
    assert!(host_allowed("localhost:5800", every));
    assert!(!host_allowed("evil.example:5800", every));
}

#[test]
fn commands_parse_and_nonsense_is_ignored() {
    assert_eq!(
        parse_command("k 4 1"),
        Some(Command::Key {
            usage: 4,
            pressed: true
        })
    );
    assert_eq!(
        parse_command("k 224 0"),
        Some(Command::Key {
            usage: 224,
            pressed: false
        })
    );
    assert_eq!(
        parse_command("m 10 20"),
        Some(Command::Move { x: 10, y: 20 })
    );
    assert_eq!(
        parse_command("b 1 1"),
        Some(Command::Button {
            button: PointerButton::Right,
            pressed: true
        })
    );
    assert_eq!(parse_command("w -3"), Some(Command::Wheel { clicks: -3 }));
    assert_eq!(parse_command("r"), Some(Command::ReleaseAll));
    assert_eq!(
        parse_command("s 1024 768"),
        Some(Command::Size {
            width: 1024,
            height: 768
        })
    );
    assert_eq!(parse_command("a"), Some(Command::Ack));
    for nonsense in [
        "",
        "k",
        "k x 1",
        "k 70000 1",
        "m -1 0",
        "b 9 1",
        "z 1",
        "w 99999999999",
        "s 1",
    ] {
        assert_eq!(parse_command(nonsense), None, "{nonsense:?}");
    }
}

/// (x, y, width, height, PNG) of each rectangle of an update message, after checking its
/// header says `screen` and its lengths add up.
fn parse_update(message: &[u8], screen: (u16, u16)) -> Vec<(u16, u16, u16, u16, Vec<u8>)> {
    let word = |at: usize| u16::from_le_bytes([message[at], message[at + 1]]);
    assert_eq!(message[0], UPDATE_KIND_RECTS);
    assert_eq!((word(1), word(3)), screen);
    let mut at = RECTS_HEADER_BYTES;
    let mut rects = Vec::new();
    for _ in 0..word(5) {
        let length = u32::from_le_bytes(message[at + 8..at + 12].try_into().unwrap()) as usize;
        let png = message[at + RECT_HEADER_BYTES..at + RECT_HEADER_BYTES + length].to_vec();
        assert_eq!(png[..4], *b"\x89PNG");
        rects.push((word(at), word(at + 2), word(at + 4), word(at + 6), png));
        at += RECT_HEADER_BYTES + length;
    }
    assert_eq!(at, message.len(), "nothing after the last rectangle");
    rects
}

#[test]
fn rectangles_are_cut_into_bands_of_at_most_band_rows() {
    let tall = Rect {
        x: 8,
        y: 3,
        width: 5,
        height: 2 * BAND_ROWS + 1,
    };
    let small = Rect {
        x: 0,
        y: 0,
        width: 2,
        height: 2,
    };
    let rows = |y, height| Rect { y, height, ..tall };
    assert_eq!(
        bands(&[tall, small]),
        [
            rows(3, BAND_ROWS),
            rows(3 + BAND_ROWS, BAND_ROWS),
            rows(3 + 2 * BAND_ROWS, 1),
            small
        ]
    );
}

#[test]
fn an_update_message_carries_every_band_in_order_as_its_own_png() {
    let display = DisplayHandle::default();
    let mut mirror = ScreenMirror::new();
    let whole = display.sync_mirror(&mut mirror).unwrap();
    // The whole screen is big enough to be encoded on several threads; the order must hold.
    let rects = parse_update(&update_message(&mirror, &whole), (1280, 800));
    let expected: Vec<_> = bands(&whole)
        .iter()
        .map(|band| (0, band.y as u16, 1280, band.height as u16))
        .collect();
    let got: Vec<_> = rects
        .iter()
        .map(|(x, y, w, h, _)| (*x, *y, *w, *h))
        .collect();
    assert_eq!(got, expected);
    for ((_, _, width, height, png), band) in rects.iter().zip(bands(&whole)) {
        assert_eq!(
            *png,
            png::encode_bgrx_rect(mirror.pixels(), mirror.stride(), band)
        );
        assert_eq!((*width as u32, *height as u32), (band.width, band.height));
    }
}

#[test]
fn the_hello_says_what_the_machine_has() {
    let both = Screen {
        display: Some(DisplayHandle::default()),
        input: Some(InputHandle::default()),
    };
    assert_eq!(
        hello_message(&both),
        [
            UPDATE_KIND_HELLO,
            HELLO_HAS_DISPLAY | HELLO_HAS_INPUT,
            0x00,
            0x05,
            0x20,
            0x03
        ],
        "1280x800"
    );
    let input_only = Screen {
        display: None,
        input: Some(InputHandle::default()),
    };
    assert_eq!(hello_message(&input_only)[1], HELLO_HAS_INPUT);
}

#[test]
fn cursor_messages_carry_the_state_the_hotspot_and_an_image() {
    assert_eq!(
        cursor_message(&uemulator::Cursor::Host),
        [UPDATE_KIND_CURSOR, CURSOR_HOST, 0, 0, 0, 0]
    );
    assert_eq!(
        cursor_message(&uemulator::Cursor::Hidden),
        [UPDATE_KIND_CURSOR, CURSOR_HIDDEN, 0, 0, 0, 0]
    );
    let image = uemulator::CursorImage {
        width: 2,
        height: 1,
        hot_x: 1,
        hot_y: 0,
        pixels: vec![0; 8],
    };
    let message = cursor_message(&uemulator::Cursor::Image(image));
    assert_eq!(message[..6], [UPDATE_KIND_CURSOR, CURSOR_IMAGE, 1, 0, 0, 0]);
    assert_eq!(&message[6..10], b"\x89PNG");
}

#[test]
fn a_size_command_reaches_the_display_and_input_without_a_screen_is_fine() {
    let display = DisplayHandle::default();
    let screen = Screen {
        display: Some(display.clone()),
        input: None,
    };
    apply(
        Command::Size {
            width: 1024,
            height: 768,
        },
        &screen,
    );
    // Keys on a machine with no input go nowhere, without a panic.
    apply(
        Command::Key {
            usage: 4,
            pressed: true,
        },
        &screen,
    );
    apply(Command::ReleaseAll, &screen);
}

/// A viewer on a free loopback port with a blank screen and input.
fn start_viewer() -> SocketAddr {
    start_viewer_with(Screen {
        display: Some(DisplayHandle::default()),
        input: Some(InputHandle::default()),
    })
}

fn start_viewer_with(screen: Screen) -> SocketAddr {
    let viewer = WebViewer::bind("127.0.0.1:0", &[]).unwrap();
    let address = viewer.address;
    let message = Box::new(viewer).start(screen).unwrap();
    assert!(message.contains(&format!("http://{address}/")), "{message}");
    address
}

fn connect(address: SocketAddr) -> TcpStream {
    let stream = TcpStream::connect(address).unwrap();
    stream.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    stream
}

/// Reads the response head, up to the empty line.
fn response_head(stream: &mut TcpStream) -> String {
    let mut head = Vec::new();
    let mut byte = [0u8];
    while !head.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).unwrap();
        head.push(byte[0]);
    }
    String::from_utf8(head).unwrap()
}

/// A text frame as a browser sends it (masked).
fn client_text(text: &str) -> Vec<u8> {
    const MASK: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
    let mut frame = vec![0x80 | OPCODE_TEXT, 0x80 | text.len() as u8];
    frame.extend_from_slice(&MASK);
    frame.extend(text.bytes().enumerate().map(|(i, b)| b ^ MASK[i % 4]));
    frame
}

#[test]
fn the_server_serves_the_page() {
    let address = start_viewer();
    let mut stream = connect(address);
    let request = format!("GET / HTTP/1.1\r\nHost: {address}\r\n\r\n");
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    assert!(response.contains("Content-Type: text/html"));
    assert!(response.contains("<canvas"));
    assert!(response.contains(&format!("Content-Length: {}", VIEWER_PAGE.len())));
}

#[test]
fn a_socket_gets_the_hello_the_cursor_and_the_whole_screen_first_and_answers_pings() {
    let address = start_viewer();
    let mut stream = connect(address);
    let origin = format!("http://{address}");
    let handshake = upgrade_request_to(&address.to_string(), Some(&origin));
    stream.write_all(handshake.as_bytes()).unwrap();
    let head = response_head(&mut stream);
    assert!(head.starts_with("HTTP/1.1 101 "), "{head}");
    assert!(
        head.contains(&format!("Sec-WebSocket-Accept: {ACCEPT}")),
        "{head}"
    );
    let mut messages = MessageReader::default();
    let hello = messages.read(&mut stream).unwrap();
    assert_eq!(
        hello.payload[..2],
        [UPDATE_KIND_HELLO, HELLO_HAS_DISPLAY | HELLO_HAS_INPUT]
    );
    let cursor = messages.read(&mut stream).unwrap();
    assert_eq!(cursor.payload[..2], [UPDATE_KIND_CURSOR, CURSOR_HOST]);
    let first = messages.read(&mut stream).unwrap();
    assert_eq!(first.opcode, OPCODE_BINARY);
    let rects = parse_update(&first.payload, (1280, 800));
    let covered: u32 = rects
        .iter()
        .map(|(x, _, width, height, _)| {
            assert_eq!((*x, *width), (0, 1280), "full-width bands");
            u32::from(*height)
        })
        .sum();
    assert_eq!(covered, 800, "the whole screen");
    // Input and acks are accepted without an answer; a ping gets its pong.
    stream.write_all(&client_text("k 4 1")).unwrap();
    stream.write_all(&client_text("a")).unwrap();
    let ping = [0x80 | OPCODE_PING, 0x80 | 2, 0, 0, 0, 0, b'h', b'i'];
    stream.write_all(&ping).unwrap();
    let pong = messages.read(&mut stream).unwrap();
    assert_eq!((pong.opcode, pong.payload), (OPCODE_PONG, b"hi".to_vec()));
    // Closing ends the session with a close frame back.
    stream
        .write_all(&[0x80 | OPCODE_CLOSE, 0x80, 0, 0, 0, 0])
        .unwrap();
    assert_eq!(messages.read(&mut stream).unwrap().opcode, OPCODE_CLOSE);
}

#[test]
fn the_screenshot_is_the_whole_screen_as_a_png_and_missing_without_a_screen() {
    let address = start_viewer();
    let mut stream = connect(address);
    let request = format!("GET /screenshot.png HTTP/1.1\r\nHost: {address}\r\n\r\n");
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let text = String::from_utf8_lossy(&response);
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
    assert!(text.contains("Content-Type: image/png"), "{text}");
    let body = response.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    assert_eq!(&response[body..body + 4], b"\x89PNG");

    let address = start_viewer_with(Screen {
        display: None,
        input: Some(InputHandle::default()),
    });
    let mut stream = connect(address);
    let request = format!("GET /screenshot.png HTTP/1.1\r\nHost: {address}\r\n\r\n");
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 404 "), "{response}");
}

#[test]
fn an_input_only_session_gets_the_hello_and_takes_keys() {
    let input = InputHandle::default();
    let address = start_viewer_with(Screen {
        display: None,
        input: Some(input.clone()),
    });
    let mut stream = connect(address);
    let handshake = upgrade_request_to(&address.to_string(), None);
    stream.write_all(handshake.as_bytes()).unwrap();
    assert!(response_head(&mut stream).starts_with("HTTP/1.1 101 "));
    let mut messages = MessageReader::default();
    let hello = messages.read(&mut stream).unwrap();
    assert_eq!(hello.payload[..2], [UPDATE_KIND_HELLO, HELLO_HAS_INPUT]);
    stream.write_all(&client_text("k 4 1")).unwrap();
    let ping = [0x80 | OPCODE_PING, 0x80, 0, 0, 0, 0];
    stream.write_all(&ping).unwrap();
    // The pong proves the key line before it was read and applied.
    assert_eq!(messages.read(&mut stream).unwrap().opcode, OPCODE_PONG);
    input.release_all();
}

#[test]
fn a_viewer_bound_by_name_answers_to_it_and_says_it_in_the_url() {
    let viewer = WebViewer::bind("localhost:0", &["Proxy.Example".into()]).unwrap();
    assert_eq!(viewer.name, None, "localhost is an address");
    assert!(viewer.host.names.contains(&"proxy.example".to_string()));
    assert!(is_host_name("vm.example"));
    assert!(!is_host_name("127.0.0.1") && !is_host_name("[::1]") && !is_host_name("::1"));
}

#[test]
fn another_sites_socket_is_refused() {
    let address = start_viewer();
    let mut stream = connect(address);
    let handshake = upgrade_request_to(&address.to_string(), Some("http://evil.example"));
    stream.write_all(handshake.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 403 "), "{response}");
}

#[test]
fn binding_an_address_in_use_says_which() {
    let taken = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = taken.local_addr().unwrap().to_string();
    let error = WebViewer::bind(&address, &[]).err().unwrap();
    assert!(error.contains(&address), "{error}");
}
