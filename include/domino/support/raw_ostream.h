#ifndef DOMINO_SUPPORT_RAW_OSTREAM_H_
#define DOMINO_SUPPORT_RAW_OSTREAM_H_

#include <domino/support/Format.h>
#include <domino/util/SmallVector.h>
#include <domino/util/StringRef.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace domino {

class Duration;
class format_object_base;
class FormattedString;
class FormattedNumber;
class FormattedBytes;

template <class T>
class [[nodiscard]] Expected;

namespace sys {
namespace fs {

enum FileAccess : unsigned;
enum OpenFlags : unsigned;
enum CreationDisposition : unsigned;

class FileLocker;

}  // namespace fs
}  // namespace sys

class raw_ostream {
 public:
  enum class OStreamKind {
    OK_OStream,
    OK_FDStream,
  };

 private:
  OStreamKind Kind;

  char *OutBufStart, *OutBufEnd, *OutBufCur;
  bool ColorEnabled = false;

  raw_ostream* TiedStream = nullptr;

  enum class BufferKind {
    Unbuffered = 0,
    InternalBuffer,
    ExternalBuffer,
  } BufferMode;

 public:
  enum class Colors {
    BLACK = 0,
    RED,
    GREEN,
    YELLOW,
    BLUE,
    MAGENTA,
    CYAN,
    WHITE,
    SAVEDCOLOR,
    RESET,
  };

  static constexpr Colors BLACK = Colors::BLACK;
  static constexpr Colors RED = Colors::RED;
  static constexpr Colors GREEN = Colors::GREEN;
  static constexpr Colors YELLOW = Colors::YELLOW;
  static constexpr Colors BLUE = Colors::BLUE;
  static constexpr Colors MAGENTA = Colors::MAGENTA;
  static constexpr Colors CYAN = Colors::CYAN;
  static constexpr Colors WHITE = Colors::WHITE;
  static constexpr Colors SAVEDCOLOR = Colors::SAVEDCOLOR;
  static constexpr Colors RESET = Colors::RESET;

  explicit raw_ostream(bool unbuffered = false,
                       OStreamKind K = OStreamKind::OK_OStream)
      : Kind(K),
        BufferMode(unbuffered ? BufferKind::Unbuffered
                              : BufferKind::InternalBuffer) {
    OutBufStart = OutBufEnd = OutBufCur = nullptr;
  }

  raw_ostream(const raw_ostream&) = delete;
  void operator=(const raw_ostream&) = delete;

  virtual ~raw_ostream();

  uint64_t tell() const { return current_pos() + GetNumBytesInBuffer(); }

  OStreamKind get_kind() const { return Kind; }

  virtual void reserveExtraSpace(uint64_t ExtraSize) {}

  void SetBuffered();

  void SetBufferSize(size_t Size) {
    flush();
    setBufferAndMode(new char[Size], Size, BufferKind::InternalBuffer);
  }

  size_t GetBufferSize() const {
    if (BufferMode != BufferKind::Unbuffered && OutBufStart == nullptr)
      return preferred_buffer_size();

    return OutBufEnd - OutBufStart;
  }

  void SetUnbuffered() {
    flush();
    SetBufferAndMode(nullptr, 0, BufferKind::Unbuffered);
  }

  size_t GetNumBytesInBuffer() const { return OutBufCur - OutBufStart; }

  void flush() {
    if (OutBufCur != OutBufStart) flush_nonempty();
  }

  raw_ostream& operator<<(char C) {
    if (OutBufCur >= OutBufEnd) return write(C);
    *OutBufCur++ = C;
    return *this;
  }

  raw_ostream& operator<<(unsigned char C) {
    if (OutBufCur >= OutBufEnd) return write(C);
    *OutBufCur++ = C;
    return *this;
  }

  raw_ostream& operator<<(signed char C) {
    if (OutBufCur >= OutBufEnd) return write(C);
    *OutBufCur++ = C;
    return *this;
  }

  raw_ostream& operator<<(StringRef Str) {
    size_t Size = Str.size();

    if (Size > (size_t)(OutBufEnd - OutBufCur)) return write(Str.data(), Size);
    if (Size) {
      memcpy(OutBufCur, Str.data(), Size);
      OutBufCur += Size;
    }
    return *this;
  }

  raw_ostream& operator<<(const char* Str) {
    return this->operator<<(StringRef(Str));
  }

  raw_ostream& operator<<(const std::string& Str) {
    return write(Str.data(), Str.length());
  }

  raw_ostream& operator<<(const std::string_view& Str) {
    return write(Str.data(), Str.length());
  }

  raw_ostream& operator<<(const SmallVectorImpl<char>& Str) {
    return write(Str.data(), Str.size());
  }

  raw_ostream& operator<<(unsigned long N);
  raw_ostream& operator<<(long N);
  raw_ostream& operator<<(unsigned long long N);
  raw_ostream& operator<<(long long N);
  raw_ostream& operator<<(const void* P);

  raw_ostream& operator<<(unsigned int N) {
    return this->operator<<((unsigned long)N);
  }

  raw_ostream& operator<<(int N) { return this->operator<<((long)N); }

  raw_ostream& operator<<(double N);

  raw_ostream& write_hex(unsigned long long N);

  raw_ostream& operator<<(Colors C);

  using uuid_t = uint8_t[16];
  raw_ostream& write_uuid(const uuid_t& UUID);

  raw_ostream& write_escaped(StringRef Str, bool UseHexEscapes = false);

  raw_ostream& write(unsigned char C);
  raw_ostream& write(const char* Ptr, size_t Size);

  raw_ostream& operator<<(const format_object_base& Fmt);

  raw_ostream& operator<<(const FormattedString& Fmt);

  raw_ostream& operator<<(const FormattedNumber& Fmt);

  raw_ostream& operator<<(const FormattedBytes& Fmt);

  raw_ostream& indent(unsigned NumSpaces);

  raw_ostream& write_zero(unsigned NumZeroes);

  virtual raw_ostream& changeColor(enum Colors Colors, bool Bold = false,
                                   bool BG = false);

  virtual raw_ostream& resetColor();

  virtual raw_ostream& reverseColor();

  virtual bool is_displayed() const { return true; }

  virtual bool has_colors() const { return false; }

  virtual void enable_colors(bool enable) { ColorEnabled = enable; }

  bool colors_enabled() const { return ColorEnabled; }

  void tie(raw_ostream* TieTo) { TiedStream = TieTo; }

 private:
  virtual void write_impl(const char* Ptr, size_t Size) = 0;

  virtual uint64_t current_pos() const = 0;

 protected:
  void SetBuffer(char* BufferStart, size_t Size) {
    SetBufferAndMode(BufferStart, Size, BufferKind::ExternalBuffer);
  }

  virtual size_t preferred_buffer_size() const;

  const char* getBufferStart() const { return OutBufStart; }

 private:
  void SetBufferAndMode(char* BufferStart, size_t Size, BufferKind Mode);

  void flush_nonempty();

  void copy_to_buffer(const char* Ptr, size_t Size);

  bool prepare_colors();

  void flush_tied_then_write(const char* Ptr, size_t Size);

  virtual void anchor();
};

template <typename OStream, typename T>
std::enable_if_t<!std::is_reference_v<OStream> &&
                     std::is_base_of<raw_ostream, OStream>::value,
                 OStream&&>
operator<<(OStream&& OS, const T& Value) {
  OS << Value;
  return std::forward<OStream>(OS);
}

class raw_pwrite_stream : public raw_ostream {
  virtual void pwrite_impl(const char* Ptr, size_t Size, uint64_t Offset) = 0;
  void anchor() override;

 public:
  explicit raw_pwrite_stream(bool Unbuffered = false,
                             OStreamKind K = OStreamKind::OK_OStream)
      : raw_ostream(Unbuffered, K) {}

  void pwrite(const char* Ptr, size_t Size, uint64_t Offset) {
    uint64_t Pos = tell();
    if (Pos) assert(Size + Offset <= Pos && "Cannot write past end of stream!");
    pwrite_impl(Ptr, Size, Offset);
  }
};

class raw_fd_ostream : public raw_pwrite_stream {
  int FD;
  bool ShouldClose;
  bool SupportSeeking = false;
  bool IsRegularFile = false;
  mutable std::optional<bool> HasColors;
  std::error_code EC;
  uint64_t pos = 0;

  void write_impl(const char* Ptr, size_t Size) override;

  void pwrite_impl(const char* Ptr, size_t Size, uint64_t Offset) override;

  uint64_t current_pos() const override { return pos; }

  size_t preferred_buffer_size() const override;

  void anchor() override;

 protected:
  void error_detected(std::error_code EC) { this->EC = EC; }

  int get_fd() const { return FD; }

  void inc_pos(uint64_t Delta) { pos += Delta; }

 public:
  raw_fd_ostream(StringRef Filename, std::error_code& EC);
  raw_fd_ostream(StringRef Filename, std::error_code& EC,
                 sys::fs::CreationDisposition Disp);
  raw_fd_ostream(StringRef Filename, std::error_code& EC,
                 sys::fs::FileAccess Access);
  raw_fd_ostream(StringRef Filename, std::error_code& EC,
                 sys::fs::OpenFlags Flags);
  raw_fd_ostream(StringRef Filename, std::error_code& EC,
                 sys::fs::CreationDisposition Disp, sys::fs::FileAccess Access,
                 sys::fs::OpenFlags Flags);
  raw_fd_ostream(int fd, bool shouldClose, bool unbuffered = false,
                 OStreamKind K = OStreamKind::OK_OStream);
  ~raw_fd_ostream() override;

  void close();

  bool supportsSeeking() const { return SupportSeeking; }

  bool is_regular_file() const { return IsRegularFile; }

  uint64_t seek(uint64_t off);

  bool is_displayed() const override;

  bool has_colors() const override;

  std::error_code error() const { return EC; }

  bool has_error() const { return bool(EC); }

  void clear_error() { EC = std::error_code(); }

  [[nodiscard]] Expected<sys::fs::FileLocker> lock();

  [[nodiscard]] Expected<sys::fs::FileLocker> tryLockFor(
      const Duration& Timeout);
};

raw_fd_ostream& outs();

raw_fd_ostream& errs();

raw_ostream& nulls();

class raw_fd_stream : public raw_fd_ostream {
 public:
  raw_fd_stream(StringRef Filename, std::error_code& EC);

  ssize_t read(char* Ptr, size_t Size);

  static bool classof(const raw_ostream* OS);
};

class raw_string_ostream : public raw_ostream {
  std::string& OS;

  void write_impl(const char* Ptr, size_t Size) override;

  uint64_t current_pos() const override { return OS.size(); }

 public:
  explicit raw_string_ostream(std::string& O) : OS(O) { SetUnbuffered(); }

  std::string& str() { return OS; }

  void reserveExtraSpace(uint64_t ExtraSpace) override {
    OS.reserve(tell() + ExtraSpace);
  }
};

class raw_svector_ostream : public raw_pwrite_stream {
  SmallVectorImpl<char>& OS;

  void write_impl(const char* Ptr, size_t Size) override;

  void pwrite_impl(const char* Ptr, size_t Size, uint64_t Offset) override;

  uint64_t current_pos() const override { return OS.size(); }

 public:
  explicit raw_svector_ostream(SmallVectorImpl<char>& O) : OS(O) {
    SetUnbuffered();
  }

  ~raw_svector_ostream() override = default;

  void flush() = delete;

  StringRef str() const { return StringRef(OS.data(), OS.size()); }

  void reserveExtraSpace(uint64_t ExtraSpace) override {
    OS.reserve(tell() + ExtraSpace);
  }
};

class raw_null_ostream : public raw_pwrite_stream {
  void write_impl(const char* Ptr, size_t Size) override;

  void pwrite_impl(const char* Ptr, size_t Size, uint64_t Offset) override;

  uint64_t current_pos() const override { return 0; }

 public:
  explicit raw_null_ostream() = default;
  ~raw_null_ostream() override;
};

class buffer_ostream : public raw_svector_ostream {
  raw_ostream& OS;
  SmallVector<char, 0> Buffer;

  void anchor() override;

 public:
  buffer_ostream(raw_ostream& OS) : raw_svector_ostream(Buffer), OS(OS) {}
  ~buffer_ostream() override { OS << str(); }
};

class buffer_unique_ostream : public raw_svector_ostream {
  std::unique_ptr<raw_ostream> OS;
  SmallVector<char, 0> Buffer;

  void anchor() override;

 public:
  buffer_unique_ostream(std::unique_ptr<raw_ostream> OS)
      : raw_svector_ostream(Buffer), OS(std::move(OS)) {
    this->OS->SetUnbuffered();
  }

  ~buffer_unique_ostream() override { *OS << str(); }
};

class Error;

Error writeToOutput(StringRef OutputFileName,
                    function_ref<Error(raw_ostream&)> WriteFn);

raw_ostream& operator<<(raw_ostream& OS, std::nullopt_t);

template <typename T, typename = decltype(std::declval<raw_ostream&>()
                                          << std::declval<const T&>())>
raw_ostream& operator<<(raw_ostrema& OS, const std::optional<T>& O) {
  if (O)
    OS << *O;
  else
    OS << std::nullopt;
  return OS;
}

}  // namespace domino

#endif  // DOMINO_SUPPORT_RAW_OSTREAM_H_