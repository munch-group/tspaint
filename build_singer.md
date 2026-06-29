# Building SINGER with pixi + CMake

This documents how the SINGER C++ engine is built on **osx-arm64** using a
[pixi](https://pixi.sh) environment to provide the toolchain and CMake to drive
the compile. Upstream ships only shell scripts (`SINGER/SINGER/local_compile.sh`
etc.); this repo adds a `CMakeLists.txt` plus a pixi `build` task so the build is
reproducible from the pinned environment.

## Prerequisites

- **pixi** installed (`pixi --version` — developed with 0.53.0).
- **macOS Command Line Tools** (or Xcode). The conda compiler compiles against the
  system SDK at `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`; pixi's
  activation exports `SDKROOT` pointing there. Install with `xcode-select --install`
  if missing.

Everything else (compiler, CMake) comes from the pixi environment — nothing is
installed system-wide.

## Dependencies added via pixi

The build needs two conda packages, both in `[dependencies]` of `pixi.toml`:

| Package        | Version spec     | Why |
|----------------|------------------|-----|
| `cmake`        | `>=4.3.4,<5`     | Build-system generator. |
| `cxx-compiler` | `>=1.11.0,<2`    | conda-forge C++ toolchain metapackage; pulls `clangxx_osx-arm64` (Clang 19) and sets `CXX`, `SDKROOT`, etc. on activation. |

They were added with:

```bash
pixi add cmake          # (already present)
pixi add cxx-compiler
```

`pixi run` activates the environment, which exports `CXX=arm64-apple-darwin20.0.0-clang++`
so CMake auto-detects the conda compiler instead of the system Apple clang.

## What `CMakeLists.txt` does

- Globs every `*.cpp` in `SINGER/SINGER/` into one executable target, `singer`
  (`main.cpp` supplies `main()`; the rest is the inference engine). This mirrors the
  upstream `g++ -std=c++17 *.cpp` glob, including `Test.cpp` (its scratch functions
  are unused but compile cleanly).
- Compiles with **C++17**, `-O3 -g`, matching `local_compile.sh`.
- Deliberately does **not** define `NDEBUG`. SINGER ships with `assert()` enabled and
  `singer_master`'s auto-retry loop depends on the binary aborting (non-zero exit) on
  numerical edge cases, so CMake's default `Release` (which adds `-DNDEBUG`) is avoided.
- Does **not** pass `-static` — Apple's linker does not support fully static binaries
  (the upstream `-static` flag in `local_compile.sh` is Linux-only).
- Emits the binary into `SINGER/SINGER/singer`, because the Python wrappers
  (`singer_master`, `parallel_singer`, …) invoke `./singer` from that directory.

## Building

From the repo root:

```bash
pixi run build
```

The `build` task (in `pixi.toml`) runs:

```bash
cmake -B build -S . && cmake --build build -j
```

`cmake -B build -S .` is idempotent, so `pixi run build` is also the incremental
rebuild command. To build manually instead:

```bash
pixi run cmake -B build -S .       # configure (once)
pixi run cmake --build build -j    # compile
```

Clean the CMake build tree (object files, cache) with:

```bash
pixi run clean        # rm -rf build  (leaves the compiled SINGER/SINGER/singer in place)
```

## Output and verification

The build produces `SINGER/SINGER/singer` — a `Mach-O 64-bit executable arm64`.
Sanity check (no arguments → prints a missing-flag message and exits 1):

```bash
$ pixi run bash -c './SINGER/SINGER/singer'
-r flag missing or invalid value.
$ echo $?
1
```

The compile emits benign `-Wunqualified-std-cast-call` warnings (unqualified
`move(...)` instead of `std::move(...)`); there are no errors.

## Notes and caveats

- **The compiled binary overwrites a git-tracked file.** `SINGER/SINGER/singer` is
  committed upstream, so after building it shows as modified in `git status`. That is
  expected — it is the freshly compiled artifact.
- **Not portable off this machine.** The binary links `@rpath/libc++.1.dylib` with an
  absolute rpath into `.pixi/envs/default/lib`. It therefore runs anywhere on this
  machine (even outside `pixi run`) as long as the pixi env exists, but it is **not**
  a redistributable build. For distribution use the upstream Linux release in
  `releases/`, or bundle/relink libc++.
- **`build/` is git-ignored** (added to `.gitignore`).

## Running the Python tools (optional)

Compiling only produces the C++ `singer` binary. The user-facing wrappers and
converters (`singer_master`, `convert_to_tskit`, `parallel_singer`,
`multi_window_singer`, `merge_ARG.py`, `compute_trace.py`, …) are Python and need
additional packages. These are **not** required to build and have not been added yet;
add them with:

```bash
pixi add python numpy pandas tskit tszip parallel
```

(`tskit >= 1.0` requires Python ≥ 3.10; `parallel` is GNU parallel, used only by
`parallel_singer`.) After that, e.g.:

```bash
pixi run ./SINGER/SINGER/singer_master   # prints usage
```
