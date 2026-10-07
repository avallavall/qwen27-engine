// Host re-quantization of a GGUF tensor to another block type. Used for the MTP draft layer: a smaller copy only
// changes which tokens the drafts propose, never the output distribution (the target verifies every token).
// Ported from llama.cpp ggml-quants.c (MIT): dequantize_row_q6_K / _q4_K, quantize_row_q4_K_ref (make_qkx2_quants)
// and quantize_iq4_xs without importance weights. The rows are split over all hardware threads.
#pragma once
#include <cstdint>
#include <vector>

#include "gguf.h"

namespace q27 {

// All rows of t (Q6_K or Q4_K) in type `to` (Q4_K or IQ4_XS), GGUF block layout. Throws for other types.
std::vector<uint8_t> requant(const GTensor& t, GType to, int threads = 0);  // threads 0 = all hardware threads
// Parses "q4_k" / "iq4_xs" / "q6_k" (any case). Throws for other names.
GType parse_gtype(const std::string& s);

}  // namespace q27
