# Build on Windows and Ubuntu Server 26.04, and what changes on Linux

Topic: one C++/CUDA codebase that builds on Windows 11 (test bench) and Ubuntu
Server 26 (deployment). Researched 2026-10-05. Nothing was built or run on the GPU.

## Summary

1. **Build.** Use CMake + Ninja + nvcc 13.4 on both systems, with
   `CMAKE_CUDA_ARCHITECTURES=120a-real` (the same flag the working llama.cpp build uses).
   On Windows nvcc needs `cl.exe` as host compiler. clang-cl is only for plain `.cpp`
   files. On Linux use gcc 15.2 (Ubuntu default, supported by CUDA 13.4). Pin the same
   nvcc build on both systems and refuse any 13.2 at configure time.
2. **Libraries.** Use cpp-httplib and nlohmann/json (MIT, header-only, already used by
   llama.cpp). Take the Jinja engine from llama.cpp `common/jinja` (MIT) or hardcode the
   one template. Make cuBLAS optional. It is only useful for BF16 GEMMs (vision), and it
   costs about 520 MiB of DLLs. CUTLASS is header-only (BSD-3). Its sm_120 GEMMs only take
   FP8/FP6/FP4 inputs, so it cannot read IQ3_S. **Do not use NCCL.** It is Linux-only.
   Without P2P it moves data through host memory, which is the same path a custom
   all-reduce uses.
3. **Ubuntu.** "Ubuntu Server 26" means **26.04 LTS** (released 2026-04-23, kernel 7.0,
   GCC 15.2, glibc 2.43). Blackwell runs only on the **open** kernel modules. Linux driver
   **615.71.09** belongs to the same R615 branch as Windows 616.64. Install CUDA 13.4 from
   NVIDIA's `ubuntu2604` apt repo (`cuda-toolkit-13-4`), which no longer installs a driver.
   Never use Ubuntu's own `nvidia-cuda-toolkit` package: it is CUDA 12.4.1 and cannot
   build sm_120. Enable `nvidia-persistenced`.
4. **Linux changes.** No WDDM, so there is no launch batching, no 2 s TDR watchdog, no
   pagefile backing for VRAM, no silent spill of VRAM into system RAM, and no ~50% cap on
   pinned RAM. Card 0 gains 1,260-1,871 MiB. That is about +19k to +28k tokens of context
   if the engine can balance VRAM across both cards (estimate). P2P works only with a
   community-patched open kernel module, and a patch for 615.71.09 exists. The memory
   overclock can be applied headless through NVML (`nvmlDeviceSetClockOffsets`, tools:
   LACT or nvoc). `nvidia-settings` with Coolbits needs X, so it is not an option.
5. **Plan impact.** llama.cpp itself behaves differently on Linux. Its default 2-GPU
   all-reduce is NCCL on Linux and its own host-staged kernel on Windows. Re-measure the
   llama.cpp baseline on Ubuntu before comparing the new engine against it.
6. **Practical.** Dual boot from a second NVMe, or shrink the Windows disk. Power the
   OcuLink docks first, then the PC (same as now). The PCIe Gen 3 setting lives in the
   BIOS, so it stays the same under Linux. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on both
   systems so "card 0" is the same physical card.

---

## 1. Build

### 1.1 Toolchain on each system

| Item | Windows 11 (test bench) | Ubuntu Server 26.04 (target) |
|---|---|---|
| CUDA | 13.4 zips in `%USERPROFILE%\\cuda\v13.4`. `nvcc --version` gives **V13.4.59** (checked in this session) | NVIDIA apt repo `ubuntu2604`. It has `cuda-nvcc-13-4` 13.4.59 and 13.4.92 (13.4 Update 1) |
| nvcc host compiler | MSVC `cl.exe` 14.44 (VS 2022 BuildTools). The CUDA 13.4 Windows guide lists only MSVC (VS 2019/2022/2026) | gcc 15.2 (default) or clang 21. CUDA 13.4 supports GCC 6-16 and Clang 7-22 |
| C++ (non-CUDA) compiler | clang-cl 20.1.8 or MSVC. clang-cl gave no speed gain in llama.cpp (45.2 vs 45.3 ms/step, `LEEME.md:612`) | gcc 15.2 or clang 21 |
| CMake | 4.3.3 in PATH. VS BuildTools also ships **3.31.6**, which is too old for `120f` (see 1.2) | 4.2.3 (`cmake` package in resolute) |
| Generator | Ninja (the zip install has no MSBuild integration) | Ninja |
| Environment | Run `vcvarsall.bat x64` first (`LEEME.md:646`) | none |

The current llama.cpp build line is in `LEEME.md:643-648`:
`-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
-DCMAKE_CUDA_HOST_COMPILER=<cl.exe of VS 2022> -DGGML_CUDA=ON -DGGML_NATIVE=ON
-DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA_CCCL_VERSION=v3.4.3`.
After the build, `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` and
`libomp.dll` are copied next to the exe.

### 1.2 Architecture flag: `120a-real`

| Flag | Meaning | Use here? |
|---|---|---|
| `120` | Code for CC 12.0, forward compatible to later 12.x | No. It cannot use the FP4 block-scaled MMA |
| `120a` | Architecture-specific. Runs only on CC 12.0. Allows all sm_120 instructions | **Yes, `120a-real`** |
| `120f` | Family-specific. Runs on 12.0 and later 12.x (for example sm_121, DGX Spark) | Not needed. One rig, one GPU model |

- NVIDIA definitions of `a` and `f`: the CUDA 12.9 family-specific blog post.
- llama.cpp replaces any plain `12X` with `12Xa`, because "the Blackwell FP4 tensor core
  instructions are not forwards compatible and therefore need 12Xa"
  (`llama-rig2/ggml/src/ggml-cuda/CMakeLists.txt:87-104`).
- `120f` needs CMake >= 3.31.8 (3.x) or >= 4.0.2. Older CMake rejects the `f` suffix
  (`ggml-cuda/CMakeLists.txt:54-63`). The CMake bundled in VS BuildTools is 3.31.6, so it
  fails on `120f`. `120a-real` works with any CMake.
- `-real` embeds no PTX, so there is no JIT fallback. That is fine for one known GPU. It
  also avoids JIT time at startup.
- Do not use `native`. It needs the GPUs visible at configure time and can produce garbage
  when none are visible (`ggml-cuda/CMakeLists.txt:106-111`). An explicit value gives the
  same binary on both systems.
- CUTLASS accepts `120a` and `120f` for its sm_120 examples
  (`refs/cutlass/examples/79_blackwell_geforce_gemm/CMakeLists.txt`, regex `120a|120f|121a`;
  `refs/cutlass/CMakeLists.txt:188-198`).

### 1.3 Suggested CMake skeleton (sketch, not tested)

```cmake
cmake_minimum_required(VERSION 3.28)
project(q27 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)                 # CUDA 13.4: C++20 OK with MSVC 193x/195x and gcc 15
set(CMAKE_CUDA_ARCHITECTURES 120a-real)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)      # no cudart DLL to ship

find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_VERSION VERSION_LESS 13.4)
  message(FATAL_ERROR "CUDA >= 13.4 required. 13.2 miscompiles IQ3_S on sm_120 (llama.cpp PR #27902)")
endif()

set(Q27_CUDA_FLAGS -use_fast_math -lineinfo) # llama.cpp uses -use_fast_math (ggml-cuda/CMakeLists.txt:194)
if (MSVC)
  # CCCL 3.2+ needs a standard preprocessor (ggml-cuda/CMakeLists.txt:252-256).
  # CUTLASS also needs /Zc:__cplusplus and /bigobj (refs/cutlass/CMakeLists.txt:625-645).
  list(APPEND Q27_CUDA_FLAGS -Xcompiler=/Zc:preprocessor,/Zc:__cplusplus,/bigobj)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
endif()
```

Two CMake presets: `win-release` (Ninja, `CMAKE_CUDA_HOST_COMPILER=cl.exe`,
`CUDAToolkit_ROOT=%USERPROFILE%/cuda/v13.4`) and `linux-release` (Ninja, gcc).
A small `build.ps1` calls `vcvarsall.bat x64` and then `cmake --preset win-release`.

### 1.4 Pitfalls

| # | Pitfall | What to do | Source |
|---|---|---|---|
| 1 | CUDA 13.2 computes IQ3_S wrong on RTX 50. The first own llama.cpp build gave garbage text | Fail at configure below 13.4. Keep a logit / perplexity test in CI | `LEEME.md:620-623`, brief |
| 2 | Windows and Linux nvcc differ (13.4.59 here, the Ubuntu repo also has 13.4.92) | Use the same nvcc build on both. Re-run the IQ3_S perplexity check after any toolkit change | `nvcc --version`; repo listing |
| 3 | VS BuildTools' CMake 3.31.6 rejects `120f` | Use `120a-real`, or the CMake 4.3.3 in PATH | `ggml-cuda/CMakeLists.txt:54-63` |
| 4 | nvcc on Windows only accepts MSVC as host compiler | `CMAKE_CUDA_HOST_COMPILER=cl.exe`. clang-cl only for `.cpp` | CUDA 13.4 Windows install guide |
| 5 | MSVC + CCCL/CUTLASS: preprocessor and `__cplusplus` errors, too many sections in an object file | `/Zc:preprocessor /Zc:__cplusplus /bigobj` | `ggml-cuda/CMakeLists.txt:252-256`, `cutlass/CMakeLists.txt:625-645` |
| 6 | Host code inside `.cu` files goes through `cl.exe` on Windows | No GCC extensions in `.cu` host code (`__int128`, VLAs, `__attribute__`). Plain C++20 only | follows from #4 |
| 7 | `long` is 32 bits on Windows and 64 bits on Linux | Use `int64_t` / `size_t` for offsets. The GGUF file is 12.12 GB | MS "Data Type Ranges" |
| 8 | No static cuBLAS on Windows | Ship the DLLs on Windows. On Linux static cuBLAS is possible | `ggml-cuda/CMakeLists.txt:163-172` |
| 9 | Ubuntu's `nvidia-cuda-toolkit` package is CUDA **12.4.1**. It has no sm_120 (needs 12.8) | Install only from NVIDIA's repo | Launchpad; `ggml-cuda/CMakeLists.txt:32` |
| 10 | `-use_fast_math` changes float results | Use the same math flags as llama.cpp when matching logits, or accept ULP-level differences | `ggml-cuda/CMakeLists.txt:194` |
| 11 | mmap of the GGUF on Windows keeps 11.5 GB in the process RAM | Read the file with plain reads into a pinned staging buffer. Do not mmap | `LEEME.md:62`, `LEEME.md:376-382` |

---

## 2. Libraries: worth it or weight

| Library | What it would do | License | Size (measured or listed) | Verdict |
|---|---|---|---|---|
| CUDA runtime (cudart) | Everything | NVIDIA EULA, redistributable (Attachment A) | `cudart64_13.dll` 0.5 MB; static lib available | **Use, static** |
| cuBLAS / cuBLASLt | BF16/FP16 GEMM: vision encoder (mmproj is BF16, 888 MiB), small BF16 tensors (`ssm_alpha`, `ssm_beta`) | NVIDIA EULA, redistributable | Windows: `cublasLt64_13.dll` 470 MiB + `cublas64_13.dll` 52 MiB (local). Linux: `libcublas-13-4` deb 390 MB compressed | **Optional.** Good for a first vision milestone and as a speed/accuracy reference. Not needed for decode |
| CUTLASS 4.8.0 | Templates for tensor-core GEMM | BSD-3-Clause (the Python CuTe DSL is under the NVIDIA EULA; not needed) | `include/` 29 MB, header-only. Clone 230 MB | **Reference and optional.** See 2.1 |
| NCCL | 2-GPU all-reduce | Apache-2.0 + BSD-3 parts | `libnccl2` deb 269 MB compressed | **No.** See 2.2 |
| cpp-httplib 0.58.0 | HTTP server, SSE streaming | MIT | `httplib.h` 176 KB + `httplib.cpp` 639 KB | **Use.** Needs `ws2_32` on Windows |
| nlohmann/json 3.12.0 | Request and tool-call JSON | MIT | `json.hpp` 979 KB (slow to compile) | **Use** |
| Jinja engine | Render the chat template stored in the GGUF | llama.cpp `common/jinja` is MIT. 6,374 lines | small | **Use llama.cpp's**, or hardcode the Qwen template and test it byte for byte against the Jinja output. The server topic decides |

Facts behind the table:

- llama.cpp sends **every quantized matmul** (all IQ/K types in this file) to its own
  MMQ kernels on Turing and newer. It does not use cuBLAS for them
  (`llama-rig2/ggml/src/ggml-cuda/mmq.cu:318-373`, the `turing_mma_available` branch at
  :371-373). So cuBLAS only covers float GEMMs today.
- cuBLAS reserves its workspace at the first GEMM, not at load. This is why a server can
  start and then fail on the first request (`LEEME.md:363-365`).
- CUDA 13.4 lists cuBLAS 13.8.0.4. The release notes have no sm_120-specific cuBLAS
  notes (CUDA 13.4 release notes).
- Redistribution of cudart and cuBLAS files is allowed by Attachment A of the CUDA EULA.
  NCCL is not in that list.
- cpp-httplib and nlohmann versions: `llama-rig2/vendor/cpp-httplib/httplib.h:11`,
  `vendor/cpp-httplib/LICENSE`, `vendor/nlohmann/json.hpp:7` (SPDX MIT) and `:68`.
  `ws2_32`: `vendor/cpp-httplib/CMakeLists.txt:23-24`.
- llama.cpp Jinja engine: `llama-rig2/common/jinja/README.md:1-5` (PR #18462).

### 2.1 CUTLASS on sm_120 (GeForce)

- The sm_120 3.x "collective builder" only supports F8F6F4 inputs:
  `static_assert(... "SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA.")`
  (`refs/cutlass/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:80-81`,
  and `:115`).
- Only TN layout. Cluster shape fixed at 1x1x1, because GeForce has no multicast
  (`sm120_mma_builder.inl:84-88`; `refs/cutlass/media/docs/cpp/blackwell_functionality.md:670-676`).
- Supported sm_120 MMA kinds: `kind::f8f6f4`, `kind::mxf8f6f4.block_scale`,
  `kind::mxf4.block_scale`, `kind::mxf4nvf4.block_scale`
  (`blackwell_functionality.md:661-666`).
- GeForce examples: `79a` NVFP4 x NVFP4 -> BF16, `79b` NVFP4 -> NVFP4, `79c` MXFP8 x MXFP6,
  `79d` grouped NVFP4, `80a/b` sparse, `87a-c` FP8 blockwise
  (`refs/cutlass/examples/79_blackwell_geforce_gemm/`, `80_...`, `87_...`).
- There is **no FP16/BF16 sm_120 3.x GEMM**. FP16/BF16 tensor-core GEMM on sm_120 uses
  the Ampere `mma.sync.m16n8k16` instructions. llama.cpp does exactly this on sm_120: the
  BF16 `mma.sync` path is enabled for any `__CUDA_ARCH__ >= AMPERE`
  (`ggml-cuda/common.cuh:292-294`, `ggml-cuda/mma.cuh:1260-1267`). CUTLASS 2.x `Sm80`
  kernels use the same instructions.
- What this means here: the weights are IQ3_S / IQ4_XS / K-quants. CUTLASS cannot read
  them. Prefill needs a custom dequant + MMA kernel, like llama.cpp MMQ. FP8 or FP4
  repacks of these values are only allowed if they are exact. That needs checking per
  type before CUTLASS FP8/FP4 kernels could be used. CUTLASS can still serve the BF16
  vision GEMMs, and its examples are a reference for sm_120 tile shapes.
- Cost: long compile times and deep templates. Pin a release tag (4.8.0, 2026-09-17,
  `refs/cutlass/CHANGELOG.md:5`).

### 2.2 NCCL: not useful here

- NCCL has no official Windows support. A community PR for Windows (#1922) is still open.
  The engine needs a Windows all-reduce anyway, so NCCL would add a second code path.
- Without P2P, NCCL uses its SHM transport: "SHM is used between devices when
  peer-to-peer cannot happen, therefore, host memory is used" (NCCL env docs). This is
  the same physical path as a custom host-staged all-reduce: GPU -> pinned host RAM ->
  other GPU.
- llama.cpp's own internal all-reduce for 2 GPUs already does this. For small tensors it
  stages through pinned host memory and syncs inside the kernel by polling a host flag
  (`ggml-cuda/allreduce.cu:13-37`). Device-side system atomics are not available on
  PCIe-attached consumer GPUs (`allreduce.cu:56-58`).
- llama.cpp picks **NCCL by default on Linux** and its internal kernel elsewhere
  (`ggml-cuda/ggml-cuda.cu:1222-1228`). `GGML_CUDA_NCCL` defaults to ON
  (`llama-rig2/ggml/CMakeLists.txt:211`). If NCCL is found at configure time, the Linux
  llama.cpp baseline runs a **different all-reduce** than the Windows one. It can be
  switched with `GGML_CUDA_ALLREDUCE=nccl|internal|none` (`ggml-cuda.cu:1221-1241`).
- NCCL on the same two cards is known to work. A vLLM setup with 2x RTX 5060 Ti on AM5
  (topology PHB) used `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`
  (club-5060ti issue #7; also `LEEME.md:666-670`).
- Recommendation: do not link NCCL into the engine. Use NCCL only on Linux as a
  reference point in Phase 0: llama.cpp with `GGML_CUDA_ALLREDUCE=nccl` against
  `internal`, and `nccl-tests` all-reduce latency at 40 KB.

---

## 3. Ubuntu Server 26

### 3.1 Release and kernel

| Item | Value | Source |
|---|---|---|
| Release | Ubuntu 26.04 LTS "Resolute Raccoon", released 2026-04-23. 5 years of standard support (to April 2031) | cnx-software; linuxiac |
| Kernel | Linux 7.0. The CUDA 13.4 guide lists kernel 7.0.0-31 for 26.04 | cnx-software; CUDA 13.4 Linux install guide |
| GCC / glibc | GCC 15.2.0, glibc 2.43 | CUDA 13.4 Linux install guide |
| Clang | versions 17-22 available, default 21 | Ubuntu for Developers, LLVM availability |
| CMake | 4.2.3 | Launchpad `cmake` in resolute |
| CUDA 13.4 support | Ubuntu 26.04 LTS is a listed platform (`ubuntu2604`) | CUDA 13.4 Linux install guide |

### 3.2 NVIDIA driver

- **Open kernel modules are required.** "Blackwell and later are only supported by the
  open kernel modules" (NVIDIA driver README, chapter "Open Linux Kernel Modules").
  `LEEME.md:701` already noted this when a V100 was ruled out.
- From R615, NVIDIA's packages no longer ship the proprietary flavor: "Package
  installation of the proprietary kernel module flavor is deprecated and starting in 615
  is no longer provided" (Data Center driver 615.71.09 release notes).
- **Branch match.** NVIDIA's own release notes pair Linux **615.71.09** with Windows
  **616.92** in one R615 release. Windows GeForce 616.64 (2026-09-03) is in the same
  616.xx series. So 615.71.09 is the Linux driver closest to what runs today.
- 615.71.09 (2026-09-09) fixes "a rare correctness issue on NVIDIA Blackwell GPUs
  affecting kernels with two or more nested levels of thread divergence". Programs built
  with NVCC 13.2.2 or newer do not pay the possible slowdown of that fix
  (9to5linux; linuxcompatible.org). An engine built with nvcc 13.4 is fine.
- The open modules have slower GPU initialization (NVIDIA README, open modules). This
  makes persistence (3.4) more important.

Install options:

| Option | How | Secure Boot | Notes |
|---|---|---|---|
| A. NVIDIA apt repo (recommended to start) | `cuda-keyring`, then `apt install nvidia-driver-pinning-615`, then compute-only: `apt -V install libnvidia-compute nvidia-dkms-open` (or the full `nvidia-open`) | DKMS module is not signed by Canonical. Needs MOK enrollment or Secure Boot off | `cuda-drivers_615.71.09` is in the `ubuntu2604` repo. Check that `nvidia-smi` and `nvidia-persistenced` came with the compute-only set |
| B. Ubuntu archive | `ubuntu-drivers install --gpgpu nvidia:615-server` + `nvidia-utils-615-server`, or `nvidia-headless-615-open` | Prebuilt signed modules (`linux-modules-nvidia-*`) work with Secure Boot | On 2026-09-29, `nvidia-graphics-drivers-615` 615.71.09 was only in **resolute-proposed** |
| C. `.run` file | `NVIDIA-Linux-x86_64-615.71.09.run -M=open` | Unsigned | Needed for the P2P patch (section 4.6): user space from the `.run` with `--no-kernel-modules`, kernel module from the patched source |

Commands for option A, from NVIDIA's driver installation guide (Ubuntu section):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install nvidia-driver-pinning-615        # install the pin before the driver
sudo apt -V install libnvidia-compute nvidia-dkms-open
```

### 3.3 CUDA 13.4 on Ubuntu 26.04

- Network repo: `apt install cuda-toolkit-13-4`. "Starting with CUDA 13.4, the CUDA
  Toolkit and the NVIDIA driver are installed and versioned independently". The toolkit
  meta-package "does not install a driver" (CUDA 13.4 Linux install guide).
- The repo has `cuda-nvcc-13-4` **13.4.59** (the same build as Windows here) and 13.4.92
  (Update 1). It also has `libcublas-13-4` 13.8.0.4 and `libnccl2` 2.32.3 for cuda13.4.
- Runfile: `sh cuda_<version>_linux.run` installs to `/usr/local/cuda-13.4`. Then add
  `/usr/local/cuda-13.4/bin` to `PATH` and `/usr/local/cuda-13.4/lib64` to
  `LD_LIBRARY_PATH` (CUDA 13.4 Linux install guide). A runfile is the easy way to get the
  exact GA build (13.4.59) if the apt meta-package pulls Update 1.
- Driver floor: CUDA 13.4 features need R615. 13.x programs run on >= 580 through minor
  version compatibility (CUDA 13.4 release notes).
- Do not install Ubuntu's `nvidia-cuda-toolkit`. It is 12.4.1 (Launchpad).

### 3.4 Persistence

- "Once all clients have closed the device file, the GPU state will be unloaded unless
  persistence mode is enabled." The daemon `nvidia-persistenced` keeps it loaded. It is
  Linux only. `nvidia-smi -pm 1` uses the daemon when it runs (NVIDIA Driver
  Persistence docs). `-pm` does not survive a reboot (nvidia-smi docs). Enable the
  daemon's systemd service.
- Why it matters here: the engine is a long-running server, so the effect is small while
  it runs. It matters for restarts, for benchmarks that start fresh processes, and for
  `nvidia-smi` calls. Power limits and clock offsets are also set per boot (4.7). Apply
  them after the daemon starts.

### 3.5 Headless: more VRAM on card 0

- The Windows desktop takes 1,260-1,871 MiB on card 0 (`LEEME.md:141-143`,
  `LEEME.md:244-247`). On Ubuntu Server with no display server, that VRAM is free. It was
  never measured.
- Estimate: the KV cache costs 68 KiB per token for both cards together (`LEEME.md:351`).
  1,260 MiB / 68 KiB = 18,974 tokens. 1,871 MiB / 68 KiB = 28,175 tokens. So **+19k to
  +28k tokens**, but only if the engine can balance the split. With an even tensor split
  only card 0 gains. The engine then has to move something from card 1 to card 0 (for
  example the vision encoder, 888 MiB of weights) to use the space on both cards.
- Put the console on the motherboard video output. The Ryzen 5 9600X has a small Radeon
  iGPU (2 CUs, guru3d review). Then no NVIDIA card drives a display. If a monitor stays on
  card 0, the text console framebuffer still takes some VRAM. Estimate: 3840 x 2160 x 4 B
  = 33 MB.
- The same move would help on Windows too (`LEEME.md:663-664`, "PENDIENTE" item 2).

---

## 4. What changes against Windows

### 4.1 Overview

| Topic | Windows (WDDM) | Ubuntu (Linux driver) | Effect on the engine |
|---|---|---|---|
| Driver model | WDDM. GeForce cannot use TCC | Linux kernel driver, open modules | Linux numbers are the ones that count |
| Kernel launch | Batched by the driver. Old NVIDIA forum numbers: 5-20 us per launch with jitter on WDDM, lowest on Linux | Lowest launch latency | Windows profiles overstate launch and sync gaps. CUDA graphs help on both |
| Watchdog | TDR: 2 s default (`TdrDelay`) | None expected on a GPU without X. Read `cudaDevAttrKernelExecTimeout` | A persistent "megakernel" that runs for seconds only works on Linux |
| VRAM backing | Every allocation has a committed backing store. The pagefile grew to 50 GB here | No backing store | Windows only. Make sure the pagefile can grow on the test bench |
| VRAM overflow | Since driver 536.40 the driver can spill to system RAM silently. Slow, no error | `cudaMalloc` fails with out-of-memory | Set "CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback" for the engine exe on Windows, so tests fail the same way as Linux |
| Pinned host memory | About 50% of RAM, set by Windows. Reported ~14.5 GB on a 32 GB PC | No 50% cap is documented. Test the limit | The 8 GB prompt cache in pinned RAM fits on both. Check the total on Windows |
| VRAM on card 0 | 1.3-1.9 GB used by the desktop | Free | +19k to +28k tokens (estimate, 3.5) |
| P2P | Not available on GeForce | Only with a community-patched open module | Optional, Linux only (4.6) |
| NCCL | Not available | Available | Not used (2.2) |
| llama.cpp 2-GPU all-reduce | Internal host-staged kernel | NCCL by default | Re-measure the baseline on Linux |
| Memory overclock | MSI Afterburner, set by hand | NVML clock offsets, headless, as root | Needs a boot-time service (4.7) |
| GPU counters (Nsight Compute) | NVIDIA App setting | Root, or `NVreg_RestrictProfilingToAdminUsers=0` | Set it once on the Linux box |
| mmap of the model | The release function is a no-op on Windows, so 11.5 GB stays in RAM | Pages can be released | Read the file without mmap on both |

### 4.2 Launch latency and batching (Windows only)

- "With the default WDDM driver on Windows, you will likely see launch latencies
  fluctuating between 5us and 20us." "The lowest launch latencies are on Linux, and with
  the TCC driver on Windows." This is from 2018 (njuffa, NVIDIA forum). Treat the numbers
  as old. The direction still holds, because the WDDM submission path still exists.
- "NVIDIA GeForce GPUs (excluding GeForce GTX Titan GPUs) do not support TCC mode"
  (CUDA 13.4 Windows install guide). So the test bench cannot remove WDDM.
- Consequence for the plan. The brief estimates ~20 ms of overhead per step. Part of it
  may be WDDM cost that Linux does not have. **Profile one decode step on Linux early**
  (Nsight Systems CLI works headless), at the latest before the go/no-go table is final.
  Phase 0 on the Windows bench measures only an upper bound on launch and sync gaps.

### 4.3 Memory: backing store, pagefile and silent spill (Windows only)

- "Every graphics allocation in the WDDM model has a backing store." "A backing store
  refers to a committed memory buffer that holds the contents of a graphics allocation
  when it's not resident in video memory" (Microsoft Learn, "Sharing the backing store
  with KMD").
- This explains `LEEME.md:378` and `:391-392`: about 27 GB of pagefile commit for the
  VRAM in use, and `C:\pagefile.sys` up to 50 GB. It costs no RAM, only disk and commit
  limit. Ubuntu has no equivalent.
- WDDM can also evict allocations to system memory under pressure. Since driver 536.40
  the NVIDIA driver can back new CUDA allocations with system RAM when VRAM is full. It
  does not raise an error, and the program runs much slower. The setting "CUDA - Sysmem
  Fallback Policy" in the NVIDIA Control Panel turns this off per program
  (NVIDIA support answer 5490; oobabooga discussion #4484). On Linux the allocation
  fails instead.
- For the engine: on Windows, set "Prefer No Sysmem Fallback" for the engine exe. Then a
  too-large context fails at load, as it will on Linux, instead of running slowly.
- Find the final maximum context on Linux. The Windows limit is lower (desktop) and
  hides overflows.

### 4.4 TDR watchdog (Windows only)

- `TdrDelay`: "The default value is 2 seconds." `TdrLevel` default is "Recover on
  timeout" (Microsoft Learn, TDR registry keys). Microsoft says end users should not
  change these keys.
- CUDA reports this as `cudaDevAttrKernelExecTimeout` ("Specifies whether there is a run
  time limit on kernels", `%USERPROFILE%\\cuda\v13.4\include\driver_types.h:2080`). A
  kernel that runs too long returns `CUDA_ERROR_LAUNCH_TIMEOUT`, and the process must
  restart (`cuda.h:3083-3092`).
- Effect: a persistent kernel per decode step (~40 ms) is fine on both systems. A kernel
  that never exits (Mirage / Hazy "megakernel" style) would hit TDR on Windows. On Linux
  with no X server the limit is expected to be off. Read the attribute on both systems to
  confirm.

### 4.5 Pinned host memory and `cudaHostRegister`

- Windows 10/11: "The limit is entirely managed by Windows, and a typical limit is 50%
  of system memory." "the NVIDIA driver doesn't control or set the limit." There is no
  known way to change it (Robert Crovella, NVIDIA forum thread 228235).
  `cudaHostRegister` hits the same cap. One report: about 14.5 GB on a 32 GB machine
  (NVIDIA forum thread 77439).
- Linux: no fixed fraction is documented. One project capped `cudaHostRegister` at 8 GiB
  as a "Windows/WDDM workaround". Removing the cap on Linux made its prefill 3.2x faster
  (Strata issue #253, 2026-09-30). That number belongs to that engine, not to this one.
  The exact Linux limit for `cudaHostAlloc` (RLIMIT_MEMLOCK or not) is **not confirmed**
  by an NVIDIA source. Test it on the box.
- Planned pinned use: the prompt cache (8 GB today, `-cram 8192`), all-reduce staging
  buffers (KB to MB), and a load staging buffer. This fits under ~14.5 GB on Windows.
  It leaves little room if the prompt cache grows.
- Mapped pinned memory (`cudaHostAllocMapped` + `cudaHostGetDevicePointer`) works under
  WDDM. llama.cpp's internal all-reduce runs on Windows with it
  (`ggml-cuda/allreduce.cu:269-285`). The comment there says the explicit device pointer
  is needed where `cudaDevAttrCanUseHostPointerForRegisteredMem` is 0.
- llama.cpp registers host buffers with
  `cudaHostRegister(..., cudaHostRegisterPortable | cudaHostRegisterReadOnly)`
  (`ggml-cuda/ggml-cuda.cu:5210`).

### 4.6 P2P (Linux only, optional)

- Stock driver: no P2P on GeForce. `GGML_CUDA_P2P=1` gave zero gain on this rig
  (`LEEME.md:687`).
- Community patch: `aikitoria/open-gpu-kernel-modules`, branch **615.71.09-p2p**. It is
  "NVIDIA driver 615.71.09 with P2P for RTX 30, 40, and 50 series". It uses BAR1:
  "BAR1 exposes GPU memory over PCIe so another GPU can read and write it directly."
- Requirements from that README:
  - Above 4G Decoding and Resizable BAR on. Both are already on (`LEEME.md:464`).
  - Kernel parameters `amd_iommu=on iommu=pt`. The README warns that this "removes DMA
    isolation".
  - Install the user-space driver from the `.run` with `--no-kernel-modules`, then run
    `./install.sh`.
  - If P2P latency is high, ACS on the root complex may block GPU-to-GPU traffic.
- Evidence on the same GPU model: 4x RTX 5060 Ti 16 GB, PCIe Gen 3 x8, all pairs "PHB
  (same host bridge)", driver 595.91.07 patched, Ubuntu 22.04, `iommu=pt`. Result in
  vLLM: "+2% to +13% decode", "3-7% shorter verify step, with prefill unchanged". The
  author notes "Driver upgrades ... are still not automatically handled"
  (eduardopessin/consumer-multigpu-inference).
- Unknowns for this rig:
  - Both cards sit on OcuLink x4 on CPU root ports of a B650I board. `nvidia-smi topo -m`
    on Linux shows the topology. PHB means "Connection traversing PCIe as well as a PCIe
    Host Bridge" (nvidia-smi docs). The AM5 vLLM report above shows PHB for two 5060 Ti
    on a 7800X3D.
  - Whether the AM5 root complex forwards P2P writes between two root ports at full speed
    is not known. Test it with `cudaDeviceCanAccessPeer` and the CUDA samples
    `p2pBandwidthLatencyTest`.
- Possible gain. Bandwidth does not change, because each direction still crosses both x4
  links. P2P removes the round trips: the receiver polls a flag in its own VRAM instead
  of reading host memory over PCIe. That is a latency gain per all-reduce, and there are
  about 128 per token (`LEEME.md:413`). The split / all-reduce research topic has to
  size it.
- Costs: Secure Boot off or MOK signing, a pinned driver version, and DMA isolation off.
  Keep P2P as a late, optional milestone. Design the all-reduce so the P2P and
  host-staged paths share one interface.

### 4.7 Power limit and memory overclock on a headless Linux box

Today on Windows: MSI Afterburner, power limit 150 W (83%), core -29 MHz, memory
+461 MHz, fan 42% (`LEEME.md:440-441`). Memory must be set by hand in the Afterburner
window (`LEEME.md:451-453`). The profile can end up applied to one card only
(`LEEME.md:455-460`).

| Setting | Linux tool | Needs X? | Source |
|---|---|---|---|
| Power limit 150 W | `nvidia-smi -i 0 -pl 150` and `-i 1`. Root. Value must lie between the min and max that `nvidia-smi` reports. Not persistent | No | nvidia-smi docs; NVML `nvmlDeviceSetPowerManagementLimit` ("Requires root/admin permissions") |
| Memory and core offset | NVML `nvmlDeviceSetClockOffsets` ("Control current clock offset of some clock domain for a given PState"). The older `nvmlDeviceSetMemClkVfOffset` is deprecated. Driver >= 555 | No | NVML API reference (615) |
| Tool: LACT | Daemon `lactd`, config in `/etc/lact/config.yaml`. "a system service that does not depend on a graphical session". MIT | No | LACT README |
| Tool: nvoc | CLI for "Blackwell GPU (GeForce RTX 50-series ...)", needs root and "nvidia-open 555+". Ships a oneshot systemd unit for boot | No | martinstark/nvoc README |
| `nvidia-settings` + Coolbits | Needs a running X server | **Yes** | (not usable on Ubuntu Server without a dummy X) |
| Lock clocks | `nvidia-smi -lgc`, `-lmc` (root). They lock inside the stock range and cannot overclock | No | nvidia-smi docs |

Units. NVML takes the memory offset in transfer-rate MHz. That is twice the DRAM clock
MHz (LACT issue #1218, 2026-10-02). `nvidia-smi` shows 14,001 MHz stock on these cards
(`LEEME.md:445`). On 17 Sep the cards read 14,672 and 14,472 MHz (`LEEME.md:659`),
which is +671 and +471 over stock. The current profile says +461 (`LEEME.md:441`). So
Afterburner's unit probably matches `nvidia-smi clocks.mem`, but the readings do not
prove it.
**Estimate:** +461 in Afterburner is about **+922 in NVML units**. Set it, then check
with `nvidia-smi --query-gpu=index,clocks.mem --format=csv` and the `vram-bw` bench.
Adjust until `clocks.mem` matches the Windows reading. LACT limits the offset to NVML's advertised range. For an RTX 5090 that range is
+6000 in transfer-rate units (LACT #1218). The range for a 5060 Ti is unknown.

Offsets and power limits reset at reboot and at driver reload. Apply them from a oneshot
systemd unit that runs after `nvidia-persistenced`. Set both GPUs explicitly and read
both back.

Fan: Afterburner holds 42%. Linux uses the automatic fan curve unless a tool sets it.
NVML fan control on these cards was **not checked**. Temperatures under load were
57-59 C (`LEEME.md:438`), so the automatic curve is probably enough.

### 4.8 Profiling on Linux

- Nsight Systems and Nsight Compute need access to GPU performance counters. On Linux:
  `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/`, then
  `update-initramfs -u -k all` and reboot. R610+ also has `/dev/nvidia-caps/` device
  nodes for per-user access (NVIDIA ERR_NVGPUCTRPERM page). Otherwise run `ncu` as root.

---

## 5. Practical

### 5.1 Dual boot on this PC

- Best: Ubuntu on its own NVMe. Choose the OS in the UEFI boot menu. Then GRUB never
  touches the Windows disk. If there is only one disk, shrink the Windows partition
  first.
- Secure Boot:
  - Ubuntu's signed prebuilt NVIDIA modules (`linux-modules-nvidia-*`, through
    `ubuntu-drivers`) work with Secure Boot.
  - "DKMS drivers are not signed with Canonical's key and thus do not support secure
    boot" (Ubuntu Server docs). This covers NVIDIA's repo (`nvidia-dkms-open`) and the
    P2P patch.
  - For those, enroll a MOK key or turn Secure Boot off.
- If Windows uses BitLocker or device encryption, have the recovery key ready before
  changing boot settings. (General practice, not checked on this PC.)
- Copy the model files to ext4 on the Linux disk: the 12.12 GB GGUF and the 888 MiB
  mmproj (`LEEME.md:30-31`). Reading them from the NTFS partition also works. If Windows
  "Fast Startup" is on, Linux may mount the NTFS volume read-only.
- After the first working setup, hold the kernel and driver packages (`apt-mark hold`).
  This matters most for the P2P-patched module, which does not rebuild by itself on a
  driver upgrade.

### 5.2 OcuLink docks: power-on order

- Same as Windows: turn on both docks first, wait, then turn on the PC. "OcuLink no es
  hot-plug" (`LEEME.md:462-467`).
- Check on Linux: `nvidia-smi -L` must list two GPUs. `lspci | grep -i nvidia` must list
  two devices.
- If a dock was late, a reboot is the safe fix. A PCI rescan
  (`echo 1 > /sys/bus/pci/rescan`) may find the card, but large BAR windows may not get
  assigned after boot. Not tested.
- Errors: Linux logs Xid messages in the kernel log; "Grep for 'NVRM: Xid'" (NVIDIA Xid
  docs). Xid 79 "GPU has fallen off the bus" is the Linux form of the Windows
  `nvlddmkm 153 BusReset TDR` seen with Gen 4 (`LEEME.md:401-403`). Also watch for PCIe
  `AER` lines in `journalctl -k`. Blackwell runs GSP firmware. Xid 119/120 are GSP
  timeouts or errors.

### 5.3 PCIe Gen 3, ReBAR, IOMMU

- Gen 3 is forced in the BIOS (`LEEME.md:401-404`). The BIOS setting does not depend on
  the OS, so it stays the same under Linux.
- Check under load:
  `nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current --format=csv`.
  At idle the link drops to gen 1-2 for power saving (ASPM). That is not a fault
  (`LEEME.md:406-407`).
- ReBAR: `nvidia-smi -q -d MEMORY` shows "BAR1 Memory Usage". It should show 16384 MiB
  per card. The P2P patch needs that (the consumer-multigpu-inference setup shows
  16384 MiB).
- IOMMU: only change it for the P2P path (`iommu=pt`). Leave the Ubuntu default
  otherwise.

### 5.4 Same card numbering on both systems

- CUDA orders devices "from fastest to slowest using a simple heuristic (default)". The
  alternative is `PCI_BUS_ID` (CUDA programming guide, environment variables). With two
  identical cards the default order is not guaranteed to match between Windows and
  Linux. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the engine's start script on both systems.
  Log the PCI bus ID of each device at startup.

### 5.5 Startup self-check (cheap, both systems)

At startup the engine should print, per GPU:
- driver version, CUDA runtime version, PCI bus ID, PCIe gen and width;
- BAR1 size, free VRAM;
- `cudaDevAttrKernelExecTimeout` (`driver_types.h:2080`);
- `cudaDevAttrCanUseHostPointerForRegisteredMem` (`:2153`);
- `cudaDevAttrHostNativeAtomicSupported` (`:2148`);
- `cudaDeviceCanAccessPeer` for 0->1 and 1->0;
- current memory clock and power limit.

Every Windows/Linux difference in this file then shows up in the log, and a lost
overclock or a card on the wrong link is caught at once.

### 5.6 CPU frequency on Linux (to check)

- The decode loop has a host part: launches, sampling, MTP control. On Ryzen, Linux uses
  the `amd-pstate` driver. Its energy/performance preference is set in
  `/sys/devices/system/cpu/cpuX/cpufreq/energy_performance_preference` (kernel docs).
- Check the default on Ubuntu Server. Measure decode with `performance` against the
  default. Not measured.

---

## Open items (to measure on the Linux box)

1. llama.cpp baseline on Ubuntu (same `rig/full` commit, gcc + nvcc 13.4), with
   `GGML_CUDA_ALLREDUCE=internal` and `=nccl`. This is the real number to beat.
2. One Nsight Systems profile of a decode step on Linux, compared with the Windows one.
   Measure how much of the ~20 ms "overhead" is WDDM.
3. Real free VRAM on card 0 headless, and the largest context with vision.
4. Pinned memory limit on Linux with 32 GB RAM.
5. `nvidia-smi topo -m` and, if the P2P patch is tried, `p2pBandwidthLatencyTest` between
   the two OcuLink root ports.
6. NVML memory offset: confirm the unit mapping (+922?) with `clocks.mem` and `vram-bw`.
7. `cudaDevAttrKernelExecTimeout` on both systems.

---

## Sources

Local files (read only):
- `qwen38_27\LEEME.md`: lines 30-31, 62, 82, 141-143, 244-247, 351, 363-365, 376-382, 391-392, 398-407, 413, 438-445, 451-460, 462-467, 612, 618-650, 659, 663-670, 687, 701.
- `llama-rig2\ggml\src\ggml-cuda\CMakeLists.txt`: 21-69, 54-63, 87-111, 163-176, 184-192, 194, 252-256.
- `llama-rig2\ggml\CMakeLists.txt:211`.
- `llama-rig2\ggml\src\ggml-cuda\ggml-cuda.cu`: 994-1066 (NCCL all-reduce), 1173-1199 (init chain), 1221-1241 (platform default and env var), 5210 (`cudaHostRegister`).
- `llama-rig2\ggml\src\ggml-cuda\allreduce.cu`: 13-37, 56-58, 269-285.
- `llama-rig2\ggml\src\ggml-cuda\mmq.cu:318-373`.
- `llama-rig2\ggml\src\ggml-cuda\common.cuh:292-298, 365-380`; `mma.cuh:1260-1267`.
- `llama-rig2\vendor\cpp-httplib\httplib.h:11`, `vendor\cpp-httplib\LICENSE`, `vendor\cpp-httplib\CMakeLists.txt:23-24`, `vendor\nlohmann\json.hpp:7,68`, `common\jinja\README.md:1-5`.
- `%USERPROFILE%\\cuda\v13.4\bin\x64\` (DLL sizes, listed this session), `bin\nvcc.exe --version` (V13.4.59), `include\driver_types.h:2080,2148,2150,2153`, `include\cuda.h:832,3083-3092`.
- `qwen27-engine\refs\cutlass` (shallow clone made this session, commit 0b55a2f, 2026-09-23): `CHANGELOG.md:5`, `CMakeLists.txt:174-198, 625-645`, `LICENSE.txt:2,31-32`, `media/docs/cpp/quickstart.md:7-11`, `media/docs/cpp/blackwell_functionality.md:654-690`, `include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:80-88,115`, `examples/79_blackwell_geforce_gemm/`, `examples/80_blackwell_geforce_sparse_gemm/`, `examples/87_blackwell_geforce_gemm_blockwise/`.

Web:
- Ubuntu 26.04 release, kernel 7.0: https://www.cnx-software.com/2026/04/24/ubuntu-26-04-lts-resolute-raccoon-released-with-linux-7-0/ and https://linuxiac.com/ubuntu-26-04-lts-resolute-raccoon-released/
- CUDA 13.4 Linux installation guide (26.04 row, compilers, repo, toolkit/driver split): https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- CUDA 13.4 release notes (R615, >=580 compatibility, driver not bundled, cuBLAS 13.8.0.4): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- CUDA 13.4 Windows installation guide (MSVC table, GeForce has no TCC): https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
- NVIDIA ubuntu2604 repo listing (cuda-toolkit-13-4, cuda-nvcc-13-4 13.4.59/13.4.92, cuda-drivers 615.71.09, libnccl2 269 MB, libcublas 390 MB): https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/
- NVIDIA driver installation guide, Ubuntu (nvidia-open, compute-only, pinning): https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
- Open kernel modules, Blackwell only on open: https://download.nvidia.com/XFree86/Linux-x86_64/595.58.03/README/kernel_open.html
- Driver 615.71.09 (Linux) / 616.92 (Windows), proprietary packages dropped: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-615-71-09/index.html
- 615.71.09 Blackwell divergence fix: https://9to5linux.com/nvidia-615-linux-graphics-driver-improves-support-for-vulkan-native-games and https://www.linuxcompatible.org/story/nvidia-6157109-linux-driver-reflex-on-proton-and-blackwell-fixes-arrive-on-linux/ and https://www.phoronix.com/news/NVIDIA-615.71.09-Linux-Driver
- Windows GeForce 616.64 (2026-09-03): https://www.techpowerup.com/352335/nvidia-releases-geforce-616-64-whql-game-ready-drivers
- Ubuntu archive package 615 in resolute-proposed: https://www.ubuntuupdates.org/package/core/resolute/multiverse/proposed/nvidia-graphics-drivers-615
- NVIDIA forum, nvidia-open on 26.04 needs the ubuntu2604 repo: https://forums.developer.nvidia.com/t/error-unable-to-locate-package-nvidia-open-on-ubuntu-26-04/368145
- Ubuntu Server docs, NVIDIA drivers (ubuntu-drivers --gpgpu, signed modules, DKMS and Secure Boot): https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/
- Launchpad: cmake 4.2.3 https://launchpad.net/ubuntu/resolute/+source/cmake ; nvidia-cuda-toolkit 12.4.1 https://launchpad.net/ubuntu/resolute/+source/nvidia-cuda-toolkit ; no nccl in resolute https://launchpad.net/ubuntu/resolute/+source/nccl
- Clang versions on Ubuntu: https://ubuntu.com/developers/docs/reference/availability/llvm/
- Family-specific / architecture-specific targets: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/
- CUDA EULA, Attachment A (redistributable cudart, cuBLAS): https://docs.nvidia.com/cuda/eula/index.html
- NCCL env vars, SHM transport: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- NCCL Windows PR (open): https://github.com/NVIDIA/nccl/pull/1922 ; license: https://raw.githubusercontent.com/NVIDIA/nccl/master/LICENSE.txt
- vLLM on 2x 5060 Ti with NCCL_P2P_DISABLE=1: https://github.com/5p00kyy/club-5060ti/issues/7
- Kernel launch latency WDDM vs Linux (2018): https://forums.developer.nvidia.com/t/kernel-launch-latency/62455
- WDDM backing store: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/sharing-backing-store-with-kmd
- TDR registry keys (TdrDelay 2 s): https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
- Sysmem fallback (driver 536.40, per-app policy): https://nvidia.custhelp.com/app/answers/detail/a_id/5490/~/system-memory-fallback-for-stable-diffusion (403 when fetched; content from search snippet) and https://github.com/oobabooga/textgen/discussions/4484
- Pinned memory 50% cap on Windows: https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235 ; 14.5 GB on 32 GB: https://forums.developer.nvidia.com/t/cudahostregister-strange-unexpected-behaviour-under-windows-10/77439 (search snippet)
- 8 GiB cudaHostRegister cap lifted on Linux: https://github.com/Niko1221/Strata/issues/253
- P2P patch for 615.71.09: https://github.com/aikitoria/open-gpu-kernel-modules
- P2P on 4x RTX 5060 Ti, measured: https://github.com/eduardopessin/consumer-multigpu-inference
- NVML device commands (power limit, persistence, deprecated VF offsets): https://docs.nvidia.com/deploy/nvml-api/615/api/group__nvmlDeviceCommands.html ; clock offsets: https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html
- LACT: https://github.com/ilya-zlobintsev/LACT ; VRAM offset range and units: https://github.com/ilya-zlobintsev/LACT/issues/1218
- nvoc: https://github.com/martinstark/nvoc
- nvidia-smi docs (-pm, -pl, -lgc, -lmc, topo): https://docs.nvidia.com/deploy/nvidia-smi/index.html
- Persistence daemon: https://docs.nvidia.com/deploy/driver-persistence/persistence-daemon.html
- Xid messages and catalog: https://docs.nvidia.com/deploy/xid-errors/working-with-xid-errors.html and https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
- GPU counter permissions: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
- CUDA environment variables (CUDA_DEVICE_ORDER, CUDA_MODULE_LOADING): https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
- cudaDeviceProp fields: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
- amd-pstate: https://docs.kernel.org/admin-guide/pm/amd-pstate.html
- Ryzen 5 9600X iGPU: https://www.guru3d.com/review/review-ryzen-5-9600x-processor/
- MSVC data type sizes (`long` is 4 bytes): https://learn.microsoft.com/en-us/cpp/cpp/data-type-ranges
