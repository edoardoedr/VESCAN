# Guide: building VMTK with VTK 9.5.2 / ITK 5.4.6 (conda, standalone)

Step-by-step guide, verified end-to-end on macOS (osx-arm64), for compiling VMTK
standalone (not inside Slicer) against VTK 9.5.2 and ITK 5.4.6 installed via
conda-forge, with working Python wrapping.

Every command below was actually run and verified in a real build session —
including the three issues found along the way (see "Notes and known issues"
section at the bottom).

## 0. Prerequisites

- conda/miniconda installed (`conda --version`)
- Xcode Command Line Tools (macOS): `xcode-select --install`
- git

## 1. Create the conda environment

```bash
conda create -n vescan python=3.11 -y
conda activate vescan
```

> Note: the Python version must have a build available on conda-forge for
> the chosen VTK/ITK version. Check with:
> `conda search -c conda-forge "vtk=9.5.2" | grep py311`

## 2. Install VTK, ITK, and the toolchain from conda-forge

**Important**: use conda-forge, not pip. The PyPI wheels for `vtk`/`itk` are
runtime-only (no headers, no `*Config.cmake` files) and don't allow compiling
C++ code such as `vtkVmtk` against them.

```bash
conda install -n vescan -c conda-forge \
  "vtk=9.5.2" \
  "itk=5.4.6" \
  "libitk-devel=5.4.6" \
  cmake compilers -y
```

- `vtk` (conda-forge) already includes headers + `vtk-config.cmake`.
- `itk` (conda-forge) is **runtime-only**, just like the pip wheels: you also
  need the separate `libitk-devel` package, which provides `ITKConfig.cmake`
  and the headers.

### Verify that find_package() works

```bash
conda activate vescan
mkdir -p /tmp/cmake-probe && cat > /tmp/cmake-probe/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.12)
project(probe)
find_package(VTK REQUIRED)
find_package(ITK REQUIRED)
message(STATUS "VTK_VERSION=${VTK_VERSION}")
message(STATUS "ITK_VERSION=${ITK_VERSION}")
EOF
cmake -S /tmp/cmake-probe -B /tmp/cmake-probe/build
rm -rf /tmp/cmake-probe
```

Expected output: `VTK_VERSION=9.5.2` and `ITK_VERSION=5.4.6`, no errors.

## 3. Clone VMTK

```bash
cd /path/to/vmtk_building
git clone https://github.com/vmtk/vmtk.git
```

(if the `vmtk/` folder already exists, skip this step)

## 4. Configure the build

```bash
conda activate vescan

cmake -S vmtk -B vmtk-build-dir \
  -DVMTK_USE_SUPERBUILD=OFF \
  -DVTK_DIR=$CONDA_PREFIX/lib/cmake/vtk-9.5 \
  -DITK_DIR=$CONDA_PREFIX/lib/cmake/ITK-5.4 \
  -DVMTK_PYTHON_VERSION=python3.11 \
  -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python3.11 \
  -DPython3_ROOT_DIR=$CONDA_PREFIX \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
```

Explanation of the non-obvious flags (see also the notes section at the bottom):

| Flag | Why it's needed |
|---|---|
| `VMTK_USE_SUPERBUILD=OFF` | Skips automatically downloading/building VTK/ITK from source: we already have them from conda-forge. |
| `Python3_EXECUTABLE`, `Python3_ROOT_DIR`, `Python3_FIND_STRATEGY=LOCATION` | Without these, `find_package(Python3)` picks the highest Python version found on the system (e.g. the conda `base` env's) instead of the active env's, breaking the Python wrapping's ABI. |
| `VMTK_PYTHON_VERSION=python3.11` | Works around a bug: `find_package(Python3 COMPONENTS Interpreter)` in VMTK's `CMakeLists.txt` doesn't populate the legacy variables (`PYTHON_VERSION_MAJOR`/`MINOR`) that this option is derived from by default. |
| `BUILD_SHARED_LIBS=ON` | Without it, vtkVmtk's Python wrapping is **silently disabled** (the `vtkMacroKitPythonWrap` macro requires `BUILD_SHARED_LIBS=ON`), and in the non-superbuild path this option has no sensible default. |

## 5. Build

```bash
LIBRARY_PATH=$CONDA_PREFIX/lib cmake --build vmtk-build-dir -j$(sysctl -n hw.ncpu)
```

> `LIBRARY_PATH=$CONDA_PREFIX/lib` is **required on macOS**: VMTK's
> `vtkVmtk/CMakeLists.txt` unconditionally overwrites
> `CMAKE_SHARED_LINKER_FLAGS`/`CMAKE_EXE_LINKER_FLAGS` with an old OpenGL/X11
> workaround, wiping out the search path to the conda env's libraries.
> This causes a link error (`library not found for -lfftw3_threads`, an ITK
> dependency) unless worked around this way. `LIBRARY_PATH` is read directly
> by the compiler/linker, bypassing the CMake flags that get overwritten.

This issue doesn't occur on Linux (the block that overwrites the flags is
guarded by `if(APPLE)`), so `LIBRARY_PATH=...` shouldn't be necessary there —
but this hasn't been verified in this session.

## 6. Quick check from the build tree (optional)

For a quick test, even before installing, you can import directly from the
build tree:

```bash
conda activate vescan
PYTHONPATH="$(pwd)/vmtk-build-dir/vtkVmtk/bin" \
python3 -c "
import vtkvmtkCommonPython
print('OK:', vtkvmtkCommonPython.vtkvmtkMath())
"
```

> `PYTHONPATH` is needed because the compiled modules (`vtkvmtkCommonPython.so`,
> etc.) live in `vmtk-build-dir/vtkVmtk/bin/` and were never installed into
> `site-packages`. **`DYLD_LIBRARY_PATH` is not needed**, though: CMake already
> embeds `LC_RPATH` entries pointing to `$CONDA_PREFIX/lib` and to the build
> tree itself into the `.so` files (the default behavior for build-tree
> binaries), so the dynamic linker finds the VTK/ITK/vtkvmtk libraries on its
> own — verifiable with
> `otool -l vmtk-build-dir/vtkVmtk/bin/vtkvmtkCommonPython.so | grep -A2 LC_RPATH`.

Note: importing `vtkvmtkCommonPython` directly only works here, from the flat
build tree. **This is not how VMTK is meant to be used normally** — see steps
7 and 8.

## 7. Install (required for actual use, not just testing)

**Don't run `cmake --install vmtk-build-dir` with no arguments**: the project
uses relative paths (e.g. `lib/python3.11/site-packages/vmtk`) resolved
against `CMAKE_INSTALL_PREFIX`, which defaults to `/usr/local` — a system
directory that requires `sudo` and, regardless, isn't where the conda env's
Python looks for packages. Install into the conda env itself instead:

```bash
conda activate vescan
cmake --install vmtk-build-dir --prefix "$CONDA_PREFIX"
```

This way:
- the executables (`vmtk`, `vmtksurfaceviewer`, etc.) end up in
  `$CONDA_PREFIX/bin`, already on `PATH` when the env is active;
- the `vmtk` Python package (`.py` scripts + compiled `vtkvmtk*Python.so`
  modules) ends up in `$CONDA_PREFIX/lib/python3.11/site-packages/vmtk`,
  already on `sys.path`.

No environment variables need to be set by hand after this step.

## 8. Correct usage after installation

VMTK's scripts do **not** import the `vtkvmtk*Python` modules directly: they
use the aggregator module `vmtk/vtkvmtk.py` (installed inside the package),
which does relative imports (`from .vtkvmtkCommonPython import *`) because the
`.so` files live in the same directory. The correct way to use the library
from Python is therefore:

```bash
conda activate vescan
python3 -c "
from vmtk import vtkvmtk
print(vtkvmtk.vtkvmtkMath())
"
```

or, for a full vmtk script:

```bash
vmtksurfaceviewer --help
```

---

## Notes and known issues (non-obvious, found during the real build)

1. **PyPI wheels are insufficient**: `pip install vtk itk` installs
   runtime-only packages with no headers or CMake files — unusable for
   compiling `vtkVmtk`. Always use conda-forge for C++ development.

2. **`itk` (conda-forge) alone isn't enough**: you also need the separate
   `libitk-devel` package to get `ITKConfig.cmake` and the C++ headers.

3. **`find_package(Python3)` is non-deterministic**: by default, CMake picks
   the highest Python version found on the system (the `VERSION` strategy),
   not necessarily the active conda environment's. It must be constrained
   explicitly.

4. **`BUILD_SHARED_LIBS` has no default in the standalone path**: in VMTK's
   `CMakeLists.txt`, the `BUILD_SHARED_LIBS=ON` option is only set explicitly
   inside the `VMTK_USE_SUPERBUILD=ON` branch (`CMakeLists.txt:171`). In the
   standalone path (`VMTK_USE_SUPERBUILD=OFF`, the one used in this guide) it
   must be specified by hand, otherwise it defaults to `OFF` and the Python
   wrapping is silently disabled with no visible error.

5. **Linker flag override on macOS**: `vtkVmtk/CMakeLists.txt:40-43`
   unconditionally overwrites `CMAKE_SHARED_LINKER_FLAGS` /
   `CMAKE_EXE_LINKER_FLAGS` with a legacy OpenGL/X11 workaround, wiping out
   any `-L` set by the environment (e.g. by conda-forge via `LDFLAGS`).
   This causes a link error on `fftw3_threads` (an ITK dependency) unless you
   use `LIBRARY_PATH` as shown in this guide.

6. **Stream tracer automatically disabled**: with VTK ≥ 9.2 (including 9.5.2),
   `VTK_VMTK_BUILD_STREAMTRACER` is disabled by default
   (`vtkVmtk/CMakeLists.txt:47-53`) because VMTK is no longer compatible with
   the rework of interpolated velocity fields introduced in that VTK version.
   This isn't a bug — it's expected.

7. **`cmake --install` without `--prefix` writes to `/usr/local`**: VMTK uses
   relative paths (`lib/${VMTK_PYTHON_VERSION}/site-packages/vmtk`) resolved
   against `CMAKE_INSTALL_PREFIX`, which defaults to `/usr/local` — requiring
   `sudo` and, regardless, not where the conda env's Python looks for
   packages. Always specify `--prefix "$CONDA_PREFIX"` (or the equivalent
   `-DCMAKE_INSTALL_PREFIX=...` at configure time).

8. **Directly importing `vtkvmtkCommonPython` is not the intended usage**:
   the compiled modules should be imported through the aggregator module
   `vmtk/vtkvmtk.py` (`from vmtk import vtkvmtk`), which uses relative imports
   (`from .vtkvmtkCommonPython import *`) assuming the `.so` files sit in the
   same directory as the `vmtk` package — true only after a proper install
   (point 7), not in the raw build tree.

## Versions verified in this guide

| Component | Version |
|---|---|
| VMTK | commit `ba7cf0f` ("Adapted to build against VTK 9.6") |
| VTK | 9.5.2 (conda-forge) |
| ITK | 5.4.6 (conda-forge, + `libitk-devel` 5.4.6) |
| Python | 3.11.14 |
| CMake | via conda-forge `cmake` package |
| Compiler | Clang 19.1.7 (conda-forge `compilers`) |
| Platform | macOS osx-arm64 |
