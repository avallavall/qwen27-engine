// Unicode tables for the tokenizer's pre-tokenizer (copied from llama.cpp, see unicode_tables.cpp).
#pragma once
#include <cstdint>

namespace q27 {

// Category flags, as the low byte of llama.cpp's unicode_cpt_flags. Every code point below 0x110000 has one.
enum : uint16_t {
  UCAT_UNDEFINED = 0x01, UCAT_NUMBER = 0x02, UCAT_LETTER = 0x04, UCAT_SEPARATOR = 0x08,
  UCAT_MARK = 0x10, UCAT_PUNCTUATION = 0x20, UCAT_SYMBOL = 0x40, UCAT_CONTROL = 0x80,
};

struct UnicodeRange {
  uint32_t first;  // the range ends where the next one starts; the last entry is {0x110000, 0}
  uint16_t flags;
};

extern const UnicodeRange kUnicodeRanges[];
extern const int kNumUnicodeRanges;
extern const uint32_t kUnicodeWhitespace[];  // code points matched by \s
extern const int kNumUnicodeWhitespace;

}  // namespace q27
