import CoreGraphics
import Foundation

// The app's copy of the guest's screen (docs/plans/28-apple.md). Both engines write the
// rectangles that changed into it and commit; the screen view shows the committed image. Pixels
// are the library's format, blue, green, red, unused (XRGB8888 little-endian), which is
// CoreGraphics' 32-bit little-endian "skip first" layout, so rows are copied as they come.
//
// Committing makes a CGImage of the whole screen: CoreGraphics copies the bitmap (copy on write),
// about 4 MB at 1280x800, once per refresh at most. Cheap enough here; a Metal texture that takes
// only the rectangles is the upgrade if a phone shows it.
@MainActor
final class ScreenSurface {
    struct Rect {
        var x: Int
        var y: Int
        var width: Int
        var height: Int
    }

    enum Pointer {
        // The guest has no pointer of its own: show the host's.
        case host
        case hidden
        case image(CGImage, hotX: Int, hotY: Int)
    }

    private(set) var width = 0
    private(set) var height = 0
    private(set) var image: CGImage?
    private(set) var pointer = Pointer.host
    // Set by the view: a new image or pointer to show.
    var onImage: (() -> Void)?
    var onPointer: (() -> Void)?

    private var context: CGContext?
    private static let bytesPerPixel = 4
    private static let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    // XRGB8888 little-endian: B, G, R, unused in memory.
    private static let screenBitmapInfo = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    // The guest's pointer images are BGRA with straight (not premultiplied) alpha.
    private static let pointerBitmapInfo = CGImageAlphaInfo.first.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

    // A new guest (or a stopped one): nothing to show until its first frame.
    func reset() {
        commits = 0
        width = 0
        height = 0
        context = nil
        image = nil
        pointer = .host
        onImage?()
        onPointer?()
    }

    // Makes the bitmap `width` x `height` (black) if it is not already.
    func resize(width: Int, height: Int) {
        guard width != self.width || height != self.height || context == nil else { return }
        self.width = width
        self.height = height
        context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * Self.bytesPerPixel, space: Self.colorSpace,
                            bitmapInfo: Self.screenBitmapInfo)
        context?.setFillColor(gray: 0, alpha: 1)
        context?.fill(CGRect(x: 0, y: 0, width: width, height: height))
    }

    // Copies `rect`'s pixels into the bitmap. `source` points at the rectangle's top-left pixel and
    // its rows are `sourceStride` bytes apart: the whole screen's stride for the native engine,
    // `rect.width * 4` for the WebKit engine's packed rectangles.
    func copy(_ rect: Rect, from source: UnsafeRawPointer, sourceStride: Int) {
        guard let context, let data = context.data else { return }
        // Clipped: a frame for a size the surface no longer has must not write past the bitmap.
        let x = min(max(rect.x, 0), width)
        let y = min(max(rect.y, 0), height)
        let columns = min(rect.width, width - x)
        let rows = min(rect.height, height - y)
        guard columns > 0, rows > 0 else { return }
        let rowBytes = columns * Self.bytesPerPixel
        let destinationStride = context.bytesPerRow
        for row in 0..<rows {
            let destination = data + (y + row) * destinationStride + x * Self.bytesPerPixel
            destination.copyMemory(from: source + row * sourceStride, byteCount: rowBytes)
        }
    }

    // Frames committed since the last reset: for `-debug-log`.
    private(set) var commits = 0

    func commit() {
        commits += 1
        image = context?.makeImage()
        onImage?()
    }

    func setPointerHost() {
        pointer = .host
        onPointer?()
    }

    func setPointerHidden() {
        pointer = .hidden
        onPointer?()
    }

    // `pixels`: `height` rows of `width` BGRA pixels, straight alpha.
    func setPointer(width: Int, height: Int, hotX: Int, hotY: Int, pixels: Data) {
        guard width > 0, height > 0, pixels.count >= width * height * Self.bytesPerPixel,
              let provider = CGDataProvider(data: pixels as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * Self.bytesPerPixel, space: Self.colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: Self.pointerBitmapInfo), provider: provider,
                                  decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else {
            setPointerHost()
            return
        }
        pointer = .image(image, hotX: hotX, hotY: hotY)
        onPointer?()
    }
}

// What the screen view sends the guest (MachineModel forwards it to the running engine).
@MainActor
protocol ScreenInput: AnyObject {
    // USB HID usage, keyboard page 0x07.
    func key(usage: UInt16, pressed: Bool)
    // Screen pixels.
    func pointerMove(x: Int, y: Int)
    // 0 left, 1 right, 2 middle, 3 back, 4 forward.
    func pointerButton(_ button: Int, pressed: Bool)
    // Notches, positive away from the user.
    func scroll(clicks: Int)
    func releaseAll()
}

// HID usages (USB HID Usage Tables, keyboard page 0x07) of what the views send by name.
enum HIDUsage {
    static let a: UInt16 = 0x04
    static let digit1: UInt16 = 0x1E
    static let digit0: UInt16 = 0x27
    static let enter: UInt16 = 0x28
    static let backspace: UInt16 = 0x2A
    static let tab: UInt16 = 0x2B
    static let space: UInt16 = 0x2C
    static let leftShift: UInt16 = 0xE1

    // A character the iOS software keyboard typed, as the key and whether it needs shift on a US
    // layout. Nil for what a US keyboard cannot type.
    static func forCharacter(_ character: Character) -> (usage: UInt16, shift: Bool)? {
        guard let ascii = character.asciiValue else { return nil }
        switch character {
        case "a"..."z": return (a + UInt16(ascii - Character("a").asciiValue!), false)
        case "A"..."Z": return (a + UInt16(ascii - Character("A").asciiValue!), true)
        case "1"..."9": return (digit1 + UInt16(ascii - Character("1").asciiValue!), false)
        case "0": return (digit0, false)
        case "\n", "\r": return (enter, false)
        case "\t": return (tab, false)
        case " ": return (space, false)
        default: return symbols[character]
        }
    }

    private static let symbols: [Character: (usage: UInt16, shift: Bool)] = [
        "-": (0x2D, false), "_": (0x2D, true), "=": (0x2E, false), "+": (0x2E, true),
        "[": (0x2F, false), "{": (0x2F, true), "]": (0x30, false), "}": (0x30, true),
        "\\": (0x31, false), "|": (0x31, true), ";": (0x33, false), ":": (0x33, true),
        "'": (0x34, false), "\"": (0x34, true), "`": (0x35, false), "~": (0x35, true),
        ",": (0x36, false), "<": (0x36, true), ".": (0x37, false), ">": (0x37, true),
        "/": (0x38, false), "?": (0x38, true),
        "!": (0x1E, true), "@": (0x1F, true), "#": (0x20, true), "$": (0x21, true), "%": (0x22, true),
        "^": (0x23, true), "&": (0x24, true), "*": (0x25, true), "(": (0x26, true), ")": (0x27, true),
    ]

    // macOS virtual key codes (Carbon's kVK_*, ANSI layout positions) to HID usages. NSEvent has
    // no HID usage of its own. Positions, not characters: the guest applies its own layout.
    static let forMacKeyCode: [UInt16: UInt16] = [
        0: 0x04, 1: 0x16, 2: 0x07, 3: 0x09, 4: 0x0B, 5: 0x0A, 6: 0x1D, 7: 0x1B, 8: 0x06, 9: 0x19,
        10: 0x64, 11: 0x05, 12: 0x14, 13: 0x1A, 14: 0x08, 15: 0x15, 16: 0x1C, 17: 0x17, 18: 0x1E,
        19: 0x1F, 20: 0x20, 21: 0x21, 22: 0x23, 23: 0x22, 24: 0x2E, 25: 0x26, 26: 0x24, 27: 0x2D,
        28: 0x25, 29: 0x27, 30: 0x30, 31: 0x12, 32: 0x18, 33: 0x2F, 34: 0x0C, 35: 0x13, 36: 0x28,
        37: 0x0F, 38: 0x0D, 39: 0x34, 40: 0x0E, 41: 0x33, 42: 0x31, 43: 0x36, 44: 0x38, 45: 0x11,
        46: 0x10, 47: 0x37, 48: 0x2B, 49: 0x2C, 50: 0x35, 51: 0x2A, 52: 0x58, 53: 0x29,
        54: 0xE7, 55: 0xE3, 56: 0xE1, 57: 0x39, 58: 0xE2, 59: 0xE0, 60: 0xE5, 61: 0xE6, 62: 0xE4,
        64: 0x6C, 65: 0x63, 67: 0x55, 69: 0x57, 71: 0x53, 75: 0x54, 76: 0x58, 78: 0x56, 79: 0x6D,
        80: 0x6E, 81: 0x67, 82: 0x62, 83: 0x59, 84: 0x5A, 85: 0x5B, 86: 0x5C, 87: 0x5D, 88: 0x5E,
        89: 0x5F, 90: 0x6F, 91: 0x60, 92: 0x61, 96: 0x3E, 97: 0x3F, 98: 0x40, 99: 0x3C, 100: 0x41,
        101: 0x42, 103: 0x44, 105: 0x68, 106: 0x6B, 107: 0x69, 109: 0x43, 111: 0x45, 113: 0x6A,
        114: 0x49, 115: 0x4A, 116: 0x4B, 117: 0x4C, 118: 0x3D, 119: 0x4D, 120: 0x3B, 121: 0x4E,
        122: 0x3A, 123: 0x50, 124: 0x4F, 125: 0x51, 126: 0x52,
    ]
}

// Where an aspect-fit image of `imageSize` lands in `bounds`, and the image pixel under a point.
enum AspectFit {
    static func frame(imageSize: CGSize, in bounds: CGRect) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0, bounds.width > 0, bounds.height > 0 else { return .zero }
        let scale = min(bounds.width / imageSize.width, bounds.height / imageSize.height)
        let size = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(x: bounds.midX - size.width / 2, y: bounds.midY - size.height / 2,
                      width: size.width, height: size.height)
    }

    // `point` in the view's coordinates with the origin top left; clamped to the image.
    static func pixel(at point: CGPoint, imageSize: CGSize, in bounds: CGRect) -> (x: Int, y: Int)? {
        let frame = frame(imageSize: imageSize, in: bounds)
        guard frame.width > 0 else { return nil }
        let x = (point.x - frame.minX) / frame.width * imageSize.width
        let y = (point.y - frame.minY) / frame.height * imageSize.height
        let clampedX = Int(min(max(x, 0), imageSize.width - 1))
        let clampedY = Int(min(max(y, 0), imageSize.height - 1))
        return (clampedX, clampedY)
    }
}
