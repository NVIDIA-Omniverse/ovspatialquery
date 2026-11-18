/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#endif

namespace example_evidence
{

constexpr int kWidth  = 1280;
constexpr int kHeight = 720;

struct Options
{
    bool                  headless = false;
    int                   frames   = 3;
    std::filesystem::path evidenceRoot =
        std::filesystem::path(OVSPATIALQUERY_PROJECT_ROOT) / "_build" / "evidence" / "examples" / "manual-direct";
};

inline Options parseOptions(int argc, char** argv)
{
    Options result;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--headless")
        {
            result.headless = true;
        }
        else if (argument == "--frames" && index + 1 < argc)
        {
            result.frames = std::stoi(argv[++index]);
        }
        else if (argument == "--evidence" && index + 1 < argc)
        {
            result.evidenceRoot = argv[++index];
        }
        else
        {
            throw std::runtime_error("usage: --headless --frames N --evidence DIR");
        }
    }
    if (result.frames <= 0)
    {
        throw std::runtime_error("--frames must be positive");
    }
    return result;
}

inline std::string jsonString(const std::string& value)
{
    std::ostringstream output;
    output << '"';
    for (const unsigned char item : value)
    {
        switch (item)
        {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (item < 0x20u)
            {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(item)
                       << std::dec;
            }
            else
            {
                output << static_cast<char>(item);
            }
        }
    }
    output << '"';
    return output.str();
}

inline void writeText(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        throw std::runtime_error("could not open evidence output: " + path.string());
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream)
    {
        throw std::runtime_error("could not write evidence output: " + path.string());
    }
}

inline uint32_t rotateRight(uint32_t value, uint32_t amount) { return (value >> amount) | (value << (32u - amount)); }

inline std::string sha256(const std::vector<uint8_t>& bytes)
{
    static constexpr uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::vector<uint8_t> padded   = bytes;
    const uint64_t       bitCount = static_cast<uint64_t>(padded.size()) * 8u;
    padded.push_back(0x80u);
    while ((padded.size() % 64u) != 56u)
    {
        padded.push_back(0u);
    }
    for (int shift = 56; shift >= 0; shift -= 8)
    {
        padded.push_back(static_cast<uint8_t>(bitCount >> shift));
    }

    std::array<uint32_t, 8> hash = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    for (size_t offset = 0; offset < padded.size(); offset += 64u)
    {
        uint32_t words[64]{};
        for (size_t index = 0; index < 16u; ++index)
        {
            const size_t byte = offset + index * 4u;
            words[index]      = (static_cast<uint32_t>(padded[byte]) << 24u) |
                           (static_cast<uint32_t>(padded[byte + 1u]) << 16u) |
                           (static_cast<uint32_t>(padded[byte + 2u]) << 8u) | static_cast<uint32_t>(padded[byte + 3u]);
        }
        for (size_t index = 16u; index < 64u; ++index)
        {
            const uint32_t s0 =
                rotateRight(words[index - 15u], 7u) ^ rotateRight(words[index - 15u], 18u) ^ (words[index - 15u] >> 3u);
            const uint32_t s1 =
                rotateRight(words[index - 2u], 17u) ^ rotateRight(words[index - 2u], 19u) ^ (words[index - 2u] >> 10u);
            words[index] = words[index - 16u] + s0 + words[index - 7u] + s1;
        }
        uint32_t a = hash[0];
        uint32_t b = hash[1];
        uint32_t c = hash[2];
        uint32_t d = hash[3];
        uint32_t e = hash[4];
        uint32_t f = hash[5];
        uint32_t g = hash[6];
        uint32_t h = hash[7];
        for (size_t index = 0; index < 64u; ++index)
        {
            const uint32_t sum1       = rotateRight(e, 6u) ^ rotateRight(e, 11u) ^ rotateRight(e, 25u);
            const uint32_t choose     = (e & f) ^ ((~e) & g);
            const uint32_t temporary1 = h + sum1 + choose + constants[index] + words[index];
            const uint32_t sum0       = rotateRight(a, 2u) ^ rotateRight(a, 13u) ^ rotateRight(a, 22u);
            const uint32_t majority   = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temporary2 = sum0 + majority;
            h                         = g;
            g                         = f;
            f                         = e;
            e                         = d + temporary1;
            d                         = c;
            c                         = b;
            b                         = a;
            a                         = temporary1 + temporary2;
        }
        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const uint32_t value : hash)
    {
        output << std::setw(8) << value;
    }
    return output.str();
}

inline std::vector<uint8_t> readBytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("could not read artifact: " + path.string());
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

inline std::string sha256File(const std::filesystem::path& path) { return sha256(readBytes(path)); }

inline uint32_t crc32(const uint8_t* data, size_t size)
{
    uint32_t crc = 0xffffffffu;
    for (size_t index = 0; index < size; ++index)
    {
        crc ^= data[index];
        for (int bit = 0; bit < 8; ++bit)
        {
            crc = (crc >> 1u) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

inline void appendBigEndian(std::vector<uint8_t>& output, uint32_t value)
{
    output.push_back(static_cast<uint8_t>(value >> 24u));
    output.push_back(static_cast<uint8_t>(value >> 16u));
    output.push_back(static_cast<uint8_t>(value >> 8u));
    output.push_back(static_cast<uint8_t>(value));
}

inline void appendChunk(std::vector<uint8_t>& png, const char type[4], const std::vector<uint8_t>& payload)
{
    appendBigEndian(png, static_cast<uint32_t>(payload.size()));
    const size_t crcStart = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), payload.begin(), payload.end());
    appendBigEndian(png, crc32(png.data() + crcStart, png.size() - crcStart));
}

inline uint32_t adler32(const std::vector<uint8_t>& data)
{
    uint32_t first  = 1u;
    uint32_t second = 0u;
    for (const uint8_t value : data)
    {
        first  = (first + value) % 65521u;
        second = (second + first) % 65521u;
    }
    return (second << 16u) | first;
}

// Writes frame.png and returns semantic.json's "evidence" object for it.
inline std::string writePng(const std::filesystem::path& path, const std::vector<uint8_t>& rgba)
{
    constexpr int width    = kWidth;
    constexpr int height   = kHeight;
    const size_t  expected = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    if (rgba.size() != expected)
    {
        throw std::runtime_error("rendered image does not match fixed 1280x720 RGBA");
    }
    std::vector<uint8_t> scanlines;
    scanlines.reserve(expected + static_cast<size_t>(height));
    std::unordered_set<uint32_t> colors;
    uint64_t                     nonblack = 0u;
    for (int y = 0; y < height; ++y)
    {
        scanlines.push_back(0u);
        for (int x = 0; x < width; ++x)
        {
            const size_t   offset = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
            const uint32_t color  = (static_cast<uint32_t>(rgba[offset]) << 16u) |
                                   (static_cast<uint32_t>(rgba[offset + 1u]) << 8u) |
                                   static_cast<uint32_t>(rgba[offset + 2u]);
            colors.insert(color);
            if (color != 0u)
            {
                ++nonblack;
            }
            scanlines.insert(scanlines.end(), rgba.begin() + static_cast<std::ptrdiff_t>(offset),
                             rgba.begin() + static_cast<std::ptrdiff_t>(offset + 4u));
        }
    }
    if (nonblack == 0u || colors.size() < 2u)
    {
        throw std::runtime_error("rendered image is blank");
    }

    std::vector<uint8_t> compressed{0x78u, 0x01u};
    size_t               offset = 0u;
    while (offset < scanlines.size())
    {
        const size_t length = std::min<size_t>(65535u, scanlines.size() - offset);
        const bool   final  = offset + length == scanlines.size();
        compressed.push_back(final ? 1u : 0u);
        compressed.push_back(static_cast<uint8_t>(length));
        compressed.push_back(static_cast<uint8_t>(length >> 8u));
        const uint16_t inverse = static_cast<uint16_t>(~static_cast<uint16_t>(length));
        compressed.push_back(static_cast<uint8_t>(inverse));
        compressed.push_back(static_cast<uint8_t>(inverse >> 8u));
        compressed.insert(compressed.end(), scanlines.begin() + static_cast<std::ptrdiff_t>(offset),
                          scanlines.begin() + static_cast<std::ptrdiff_t>(offset + length));
        offset += length;
    }
    appendBigEndian(compressed, adler32(scanlines));

    std::vector<uint8_t> png{0x89u, 0x50u, 0x4eu, 0x47u, 0x0du, 0x0au, 0x1au, 0x0au};
    std::vector<uint8_t> header;
    appendBigEndian(header, static_cast<uint32_t>(width));
    appendBigEndian(header, static_cast<uint32_t>(height));
    header.insert(header.end(), {8u, 6u, 0u, 0u, 0u});
    appendChunk(png, "IHDR", header);
    appendChunk(png, "IDAT", compressed);
    appendChunk(png, "IEND", {});
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    if (!stream)
    {
        throw std::runtime_error("could not write PNG evidence");
    }
    stream.close();
    std::ostringstream json;
    json << "{\"height\":720,\"nonblack_pixels\":" << nonblack
         << ",\"png\":" << jsonString(std::filesystem::absolute(path).string())
         << ",\"sha256\":" << jsonString(sha256(png)) << ",\"unique_rgb_count\":" << colors.size()
         << ",\"width\":1280}";
    return json.str();
}

inline std::vector<uint8_t> copyRgba(const DLTensor& tensor)
{
    if (tensor.device.device_type != kDLCPU || tensor.data == nullptr || tensor.ndim != 3 ||
        tensor.dtype.code != kDLUInt || tensor.dtype.bits != 8u || tensor.dtype.lanes != 1u ||
        tensor.shape == nullptr || tensor.shape[0] != kHeight || tensor.shape[1] != kWidth ||
        (tensor.shape[2] != 3 && tensor.shape[2] != 4))
    {
        throw std::runtime_error("LdrColor did not expose fixed CPU uint8 HxWx3/4");
    }
    const int64_t        channels = tensor.shape[2];
    const int64_t        stride0  = tensor.strides ? tensor.strides[0] : tensor.shape[1] * channels;
    const int64_t        stride1  = tensor.strides ? tensor.strides[1] : channels;
    const int64_t        stride2  = tensor.strides ? tensor.strides[2] : 1;
    const uint8_t*       source   = static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset;
    std::vector<uint8_t> result(static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight) * 4u);
    for (int y = 0; y < kHeight; ++y)
    {
        for (int x = 0; x < kWidth; ++x)
        {
            const uint8_t* pixel = source + y * stride0 + x * stride1;
            const size_t   destination =
                (static_cast<size_t>(y) * static_cast<size_t>(kWidth) + static_cast<size_t>(x)) * 4u;
            result[destination]      = pixel[0];
            result[destination + 1u] = pixel[stride2];
            result[destination + 2u] = pixel[2 * stride2];
            result[destination + 3u] = channels == 4 ? pixel[3 * stride2] : 255u;
        }
    }
    return result;
}

#if defined(_WIN32)
struct ViewportState
{
    const std::vector<uint8_t>* bgra = nullptr;
};

inline LRESULT CALLBACK viewportProcedure(HWND window, UINT message, WPARAM word, LPARAM value)
{
    ViewportState* state = reinterpret_cast<ViewportState*>(GetWindowLongPtrW(window, GWLP_USERDATA));
    if (message == WM_CREATE)
    {
        const CREATESTRUCTW* create = reinterpret_cast<const CREATESTRUCTW*>(value);
        state                       = static_cast<ViewportState*>(create->lpCreateParams);
        SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(state));
    }
    if (message == WM_PAINT && state != nullptr && state->bgra != nullptr)
    {
        PAINTSTRUCT paint{};
        HDC         device = BeginPaint(window, &paint);
        BITMAPINFO  bitmap{};
        bitmap.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bitmap.bmiHeader.biWidth       = kWidth;
        bitmap.bmiHeader.biHeight      = -kHeight;
        bitmap.bmiHeader.biPlanes      = 1;
        bitmap.bmiHeader.biBitCount    = 32;
        bitmap.bmiHeader.biCompression = BI_RGB;
        SetDIBitsToDevice(device, 0, 0, kWidth, kHeight, 0, 0, 0, kHeight, state->bgra->data(), &bitmap,
                          DIB_RGB_COLORS);
        EndPaint(window, &paint);
        return 0;
    }
    if (message == WM_KEYDOWN && word == VK_ESCAPE)
    {
        DestroyWindow(window);
        return 0;
    }
    if (message == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(window, message, word, value);
}
#endif

inline void requireInteractiveViewportAvailable()
{
#if defined(__linux__)
    const char* display = std::getenv("DISPLAY");
    if (display == nullptr || display[0] == '\0')
    {
        throw std::runtime_error("interactive Linux viewport requires an X11 DISPLAY; "
                                 "start from a graphical session or use --headless");
    }
#endif
}

inline void presentInteractive(const std::string& title, const std::vector<uint8_t>& rgba,
                               const std::function<std::vector<uint8_t>()>& nextFrame)
{
#if defined(_WIN32)
    if (rgba.size() != static_cast<size_t>(kWidth * kHeight * 4))
    {
        throw std::runtime_error("interactive viewport received invalid RGBA");
    }
    std::vector<uint8_t> bgra = rgba;
    for (size_t offset = 0u; offset < bgra.size(); offset += 4u)
    {
        std::swap(bgra[offset], bgra[offset + 2u]);
    }
    const HINSTANCE   instance    = GetModuleHandleW(nullptr);
    constexpr wchar_t className[] = L"OvSpatialQueryExampleViewport";
    WNDCLASSW         windowClass{};
    windowClass.lpfnWndProc   = viewportProcedure;
    windowClass.hInstance     = instance;
    windowClass.hCursor       = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    windowClass.lpszClassName = className;
    const ATOM atom           = RegisterClassW(&windowClass);
    if (atom == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
    {
        throw std::runtime_error("could not register interactive viewport");
    }
    RECT            rectangle{0, 0, kWidth, kHeight};
    constexpr DWORD style = WS_OVERLAPPEDWINDOW;
    AdjustWindowRect(&rectangle, style, FALSE);
    ViewportState state{&bgra};
    std::wstring  wideTitle(title.begin(), title.end());
    HWND          window = CreateWindowExW(0, className, wideTitle.c_str(), style, CW_USEDEFAULT, CW_USEDEFAULT,
                                           rectangle.right - rectangle.left, rectangle.bottom - rectangle.top, nullptr, nullptr,
                                           instance, &state);
    if (window == nullptr)
    {
        throw std::runtime_error("could not create interactive viewport");
    }
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);
    MSG  message{};
    bool running = true;
    while (running)
    {
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
        {
            if (message.message == WM_QUIT)
            {
                running = false;
                break;
            }
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if (!running)
        {
            break;
        }
        const std::vector<uint8_t> frame = nextFrame();
        if (frame.size() != bgra.size())
        {
            throw std::runtime_error("interactive orbit returned invalid RGBA");
        }
        bgra = frame;
        for (size_t offset = 0u; offset < bgra.size(); offset += 4u)
        {
            std::swap(bgra[offset], bgra[offset + 2u]);
        }
        InvalidateRect(window, nullptr, FALSE);
    }
#elif defined(__linux__)
    requireInteractiveViewportAvailable();
    if (rgba.size() != static_cast<size_t>(kWidth * kHeight * 4))
    {
        throw std::runtime_error("interactive viewport received invalid RGBA");
    }
    Display* display = XOpenDisplay(nullptr);
    if (display == nullptr)
    {
        throw std::runtime_error("interactive Linux viewport requires an X11 DISPLAY; "
                                 "start from a graphical session or use --headless");
    }
    const int screen = DefaultScreen(display);
    Window    window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, kWidth, kHeight, 1,
                                           BlackPixel(display, screen), BlackPixel(display, screen));
    if (window == 0u)
    {
        XCloseDisplay(display);
        throw std::runtime_error("could not create interactive Linux viewport");
    }
    bool windowOwned = true;

    auto cleanup = [&]()
    {
        if (windowOwned)
        {
            XDestroyWindow(display, window);
            windowOwned = false;
        }
        XCloseDisplay(display);
    };

    try
    {
        XStoreName(display, window, title.c_str());
        XSelectInput(display, window, KeyPressMask | ExposureMask | StructureNotifyMask);
        Atom wmDelete = XInternAtom(display, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display, window, &wmDelete, 1);
        XMapWindow(display, window);
        GC                   gc   = DefaultGC(display, screen);
        std::vector<uint8_t> bgra = rgba;

        auto putFrame = [&]()
        {
            XImage image{};
            image.width            = kWidth;
            image.height           = kHeight;
            image.format           = ZPixmap;
            image.data             = reinterpret_cast<char*>(bgra.data());
            image.byte_order       = LSBFirst;
            image.bitmap_unit      = 32;
            image.bitmap_bit_order = LSBFirst;
            image.bitmap_pad       = 32;
            image.depth            = 24;
            image.bytes_per_line   = kWidth * 4;
            image.bits_per_pixel   = 32;
            image.red_mask         = 0x00ff0000;
            image.green_mask       = 0x0000ff00;
            image.blue_mask        = 0x000000ff;
            if (XInitImage(&image) == 0)
            {
                throw std::runtime_error("could not initialize Linux viewport image");
            }
            XPutImage(display, window, gc, &image, 0, 0, 0, 0, kWidth, kHeight);
            XFlush(display);
        };

        for (size_t offset = 0u; offset < bgra.size(); offset += 4u)
        {
            std::swap(bgra[offset], bgra[offset + 2u]);
        }
        putFrame();

        bool running = true;
        while (running)
        {
            while (XPending(display) > 0)
            {
                XEvent event{};
                XNextEvent(display, &event);
                if (event.type == ClientMessage && static_cast<Atom>(event.xclient.data.l[0]) == wmDelete)
                {
                    running = false;
                    break;
                }
                if (event.type == DestroyNotify)
                {
                    windowOwned = false;
                    running     = false;
                    break;
                }
                if (event.type == KeyPress && XLookupKeysym(&event.xkey, 0) == XK_Escape)
                {
                    running = false;
                    break;
                }
            }
            if (!running)
            {
                break;
            }
            const std::vector<uint8_t> frame = nextFrame();
            if (frame.size() != bgra.size())
            {
                throw std::runtime_error("interactive orbit returned invalid RGBA");
            }
            bgra = frame;
            for (size_t offset = 0u; offset < bgra.size(); offset += 4u)
            {
                std::swap(bgra[offset], bgra[offset + 2u]);
            }
            putFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    catch (...)
    {
        cleanup();
        throw;
    }
    cleanup();
#else
    (void)title;
    (void)rgba;
    (void)nextFrame;
    throw std::runtime_error("interactive viewport is supported on Windows and Linux/X11; "
                             "use --headless on this platform");
#endif
}

} // namespace example_evidence
