cmake_minimum_required(VERSION 3.28)
project(q27 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA 13.2 miscompiles IQ3_S on sm_120 (llama.cpp PR #27902). Refuse anything older than 13.4.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.4)
  message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} found at ${CMAKE_CUDA_COMPILER}. Use CUDA 13.4 or newer.")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 120a-real)
endif()

# -use_fast_math: llama.cpp builds its CUDA code with it; the same flags give the same rounding.
set(Q27_CUDA_FLAGS -use_fast_math --expt-relaxed-constexpr -lineinfo)
if(MSVC)
  list(APPEND Q27_CUDA_FLAGS -Xcompiler=/utf-8 -Xcompiler=/Zc:__cplusplus)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(q27core STATIC
  src/gguf.cpp
  src/qgemv.cu
  src/ops.cu
  src/attn.cu
  src/qgemm.cu
  src/prefill.cu
  src/model.cu
  src/sampling.cu
  src/vision.cu
  src/vision_img.cpp
)
target_include_directories(q27core PUBLIC src)
target_compile_options(q27core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(q27core PUBLIC CUDA::cudart_static CUDA::cuda_driver)

add_executable(bench_gemv tools/bench_gemv.cu)
target_compile_options(bench_gemv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(bench_gemv PRIVATE q27core)

add_executable(bench_pass tools/bench_pass.cu)
target_compile_options(bench_pass PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(bench_pass PRIVATE q27core)

add_executable(q27_ppl tools/q27_ppl.cu)
target_compile_options(q27_ppl PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(q27_ppl PRIVATE q27core)

add_executable(q27_gen tools/q27_gen.cu)
target_compile_options(q27_gen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(q27_gen PRIVATE q27core q27text)

add_executable(bench_attn tools/bench_attn.cu)
target_compile_options(bench_attn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(bench_attn PRIVATE q27core)

add_executable(q27_ctx tools/q27_ctx.cu)
target_compile_options(q27_ctx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(q27_ctx PRIVATE q27core)

add_executable(bench_gemm tools/bench_gemm.cu)
target_compile_options(bench_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(bench_gemm PRIVATE q27core)

add_executable(test_fused tools/test_fused.cu)
target_compile_options(test_fused PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(test_fused PRIVATE q27core)

add_executable(test_ar tools/test_ar.cu)
target_compile_options(test_ar PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(test_ar PRIVATE q27core)

add_executable(test_smem tools/test_smem.cu)
target_compile_options(test_smem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(test_smem PRIVATE q27core)

add_executable(test_link tools/test_link.cu)
target_compile_options(test_link PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(test_link PRIVATE q27core)

add_executable(test_ex tools/test_ex.cu)
target_compile_options(test_ex PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(test_ex PRIVATE q27core)

add_executable(bench_gdn tools/bench_gdn.cu)
target_compile_options(bench_gdn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(bench_gdn PRIVATE q27core)

add_executable(test_cache tools/test_cache.cu)
target_compile_options(test_cache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(test_cache PRIVATE q27core)

add_executable(test_vision tools/test_vision.cu)
target_compile_options(test_vision PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${Q27_CUDA_FLAGS}>)
target_link_libraries(test_vision PRIVATE q27core)

# ---- the server (needs q27text, defined below)
add_executable(q27_server tools/q27_server.cpp src/engine.cpp third_party/httplib/httplib.cpp)
target_include_directories(q27_server PRIVATE src third_party)
target_link_libraries(q27_server PRIVATE q27core q27text)

# ---- text side: tokenizer and chat template (C++ only), and their tests
add_library(q27text STATIC src/tokenizer.cpp src/unicode_tables.cpp src/chat_template.cpp src/gguf.cpp
  src/chat_parser.cpp)
target_include_directories(q27text PUBLIC src third_party)

add_executable(test_tokenizer tools/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE q27text)

add_executable(test_template tools/test_template.cpp)
target_link_libraries(test_template PRIVATE q27text)

add_executable(vocab_freq tools/vocab_freq.cpp)
target_link_libraries(vocab_freq PRIVATE q27text)

add_executable(test_parser tools/test_parser.cpp)
target_link_libraries(test_parser PRIVATE q27text)
