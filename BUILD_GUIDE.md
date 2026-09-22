# Guida: build di VMTK con VTK 9.5.2 / ITK 5.4.6 (conda, standalone)

Guida passo-passo, verificata end-to-end su macOS (osx-arm64), per compilare VMTK
standalone (non dentro Slicer) contro VTK 9.5.2 e ITK 5.4.6 installati via
conda-forge, con wrapping Python funzionante.

Ogni comando qui sotto è stato effettivamente eseguito e verificato in una sessione
di build reale — inclusi i tre problemi trovati lungo il percorso (vedi sezione
"Note e problemi noti" in fondo).

## 0. Prerequisiti

- conda/miniconda installato (`conda --version`)
- Xcode Command Line Tools (macOS): `xcode-select --install`
- git

## 1. Creare l'ambiente conda

```bash
conda create -n vescan python=3.11 -y
conda activate vescan
```

> Nota: la versione di Python deve avere un build disponibile su conda-forge per
> la versione di VTK/ITK scelta. Verifica con:
> `conda search -c conda-forge "vtk=9.5.2" | grep py311`

## 2. Installare VTK, ITK e il toolchain da conda-forge

**Importante**: usare conda-forge, non pip. I wheel PyPI di `vtk`/`itk` sono
runtime-only (nessun header, nessun file `*Config.cmake`) e non permettono di
compilare codice C++ come `vtkVmtk` contro di essi.

```bash
conda install -n vescan -c conda-forge \
  "vtk=9.5.2" \
  "itk=5.4.6" \
  "libitk-devel=5.4.6" \
  cmake compilers -y
```

- `vtk` (conda-forge) include già header + `vtk-config.cmake`.
- `itk` (conda-forge) è **runtime-only**, come i wheel pip: serve in aggiunta il
  pacchetto separato `libitk-devel`, che fornisce `ITKConfig.cmake` e gli header.

### Verifica che find_package() funzioni

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

Output atteso: `VTK_VERSION=9.5.2` e `ITK_VERSION=5.4.6`, nessun errore.

## 3. Clonare VMTK

```bash
cd /path/to/vmtk_building
git clone https://github.com/vmtk/vmtk.git
```

(se la cartella `vmtk/` esiste già, salta questo passo)

## 4. Configurare la build

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

Spiegazione dei flag non ovvi (vedi anche sezione note in fondo):

| Flag | Perché serve |
|---|---|
| `VMTK_USE_SUPERBUILD=OFF` | Salta lo scaricamento/compilazione automatica di VTK/ITK da sorgente: li abbiamo già da conda-forge. |
| `Python3_EXECUTABLE`, `Python3_ROOT_DIR`, `Python3_FIND_STRATEGY=LOCATION` | Senza questi, `find_package(Python3)` sceglie la versione Python più alta trovata sul sistema (es. quella dell'env conda `base`) invece di quella dell'env attivo, rompendo l'ABI del wrapping Python. |
| `VMTK_PYTHON_VERSION=python3.11` | Bypassa un bug: `find_package(Python3 COMPONENTS Interpreter)` nel `CMakeLists.txt` di VMTK non popola le variabili legacy (`PYTHON_VERSION_MAJOR`/`MINOR`) da cui questa opzione viene derivata di default. |
| `BUILD_SHARED_LIBS=ON` | Senza, il wrapping Python di vtkVmtk viene **silenziosamente disattivato** (il macro `vtkMacroKitPythonWrap` richiede `BUILD_SHARED_LIBS=ON`), e nel percorso non-superbuild questa opzione non ha un default sensato. |

## 5. Compilare

```bash
LIBRARY_PATH=$CONDA_PREFIX/lib cmake --build vmtk-build-dir -j$(sysctl -n hw.ncpu)
```

> `LIBRARY_PATH=$CONDA_PREFIX/lib` è **necessario su macOS**: il file
> `vtkVmtk/CMakeLists.txt` di VMTK sovrascrive incondizionatamente
> `CMAKE_SHARED_LINKER_FLAGS`/`CMAKE_EXE_LINKER_FLAGS` con un vecchio workaround
> OpenGL/X11, cancellando il percorso di ricerca verso le librerie dell'env conda.
> Questo causa un errore di link (`library not found for -lfftw3_threads`,
> dipendenza di ITK) se non lo si aggira così. `LIBRARY_PATH` è letto
> direttamente dal compilatore/linker, non passa dalle flag CMake che vengono
> sovrascritte.

Su Linux questo problema non si presenta (il blocco che sovrascrive i flag è
`if(APPLE)`), quindi `LIBRARY_PATH=...` non dovrebbe essere necessario — ma non è
stato verificato in questa sessione.

## 6. Verifica rapida dal build tree (opzionale)

Per un test veloce, prima ancora di installare, si può importare direttamente
dal build tree:

```bash
conda activate vescan
PYTHONPATH="$(pwd)/vmtk-build-dir/vtkVmtk/bin" \
python3 -c "
import vtkvmtkCommonPython
print('OK:', vtkvmtkCommonPython.vtkvmtkMath())
"
```

> `PYTHONPATH` è necessario perché i moduli compilati (`vtkvmtkCommonPython.so`,
> ecc.) vivono in `vmtk-build-dir/vtkVmtk/bin/` e non sono mai stati installati
> in `site-packages`. **Non serve invece `DYLD_LIBRARY_PATH`**: CMake incorpora
> già nei `.so` gli `LC_RPATH` verso `$CONDA_PREFIX/lib` e verso la build-tree
> stessa (comportamento di default per i binari in build-tree), quindi il
> dynamic linker trova da solo le librerie VTK/ITK/vtkvmtk — verificabile con
> `otool -l vmtk-build-dir/vtkVmtk/bin/vtkvmtkCommonPython.so | grep -A2 LC_RPATH`.

Nota: `import vtkvmtkCommonPython` diretto funziona solo qui, dal build tree
flat. **Non è il modo in cui VMTK va usato normalmente** — vedi step 7 e 8.

## 7. Installare (necessario per un uso reale, non solo di test)

**Non usare `cmake --install vmtk-build-dir` senza argomenti**: il progetto usa
path relativi (es. `lib/python3.11/site-packages/vmtk`) risolti rispetto a
`CMAKE_INSTALL_PREFIX`, che di default è `/usr/local` — una directory di
sistema che richiede `sudo` e che comunque non è dove il Python dell'env conda
cerca i pacchetti. Installa invece dentro l'env conda stesso:

```bash
conda activate vescan
cmake --install vmtk-build-dir --prefix "$CONDA_PREFIX"
```

Così:
- gli eseguibili (`vmtk`, `vmtksurfaceviewer`, ecc.) finiscono in
  `$CONDA_PREFIX/bin`, già nel `PATH` quando l'env è attivo;
- il pacchetto Python `vmtk` (script `.py` + moduli compilati
  `vtkvmtk*Python.so`) finisce in
  `$CONDA_PREFIX/lib/python3.11/site-packages/vmtk`, già su `sys.path`.

Nessuna variabile d'ambiente da impostare a mano dopo questo passo.

## 8. Uso corretto dopo l'installazione

Gli script di VMTK **non** importano i moduli `vtkvmtk*Python` direttamente:
usano il modulo aggregatore `vmtk/vtkvmtk.py` (installato dentro il pacchetto),
che fa import relativi (`from .vtkvmtkCommonPython import *`) perché i `.so`
vivono nella stessa directory. Il modo corretto di usare la libreria da Python
è quindi:

```bash
conda activate vescan
python3 -c "
from vmtk import vtkvmtk
print(vtkvmtk.vtkvmtkMath())
"
```

oppure, per un intero script vmtk:

```bash
vmtksurfaceviewer --help
```

---

## Note e problemi noti (non ovvi, trovati durante la build reale)

1. **Wheel PyPI insufficienti**: `pip install vtk itk` installa pacchetti
   runtime-only senza header né file CMake — inutilizzabili per compilare
   `vtkVmtk`. Usare sempre conda-forge per lo sviluppo C++.

2. **`itk` (conda-forge) da solo non basta**: serve il pacchetto separato
   `libitk-devel` per avere `ITKConfig.cmake` e gli header C++.

3. **`find_package(Python3)` non deterministico**: CMake, di default, sceglie
   la versione Python più alta trovata sul sistema (strategia `VERSION`), non
   necessariamente quella dell'ambiente conda attivo. Va vincolato
   esplicitamente.

4. **`BUILD_SHARED_LIBS` non ha un default nel percorso standalone**: nel
   `CMakeLists.txt` di VMTK, l'opzione `BUILD_SHARED_LIBS=ON` viene impostata
   esplicitamente solo dentro il ramo `VMTK_USE_SUPERBUILD=ON`
   (`CMakeLists.txt:171`). Nel percorso standalone (`VMTK_USE_SUPERBUILD=OFF`,
   quello usato in questa guida) bisogna specificarla a mano, altrimenti
   defaulta a `OFF` e il wrapping Python viene disattivato senza errori
   visibili.

5. **Override dei linker flag su macOS**: `vtkVmtk/CMakeLists.txt:40-43`
   sovrascrive incondizionatamente `CMAKE_SHARED_LINKER_FLAGS` /
   `CMAKE_EXE_LINKER_FLAGS` con un workaround OpenGL/X11 legacy, cancellando
   qualunque `-L` impostato dall'ambiente (es. da conda-forge via `LDFLAGS`).
   Causa un errore di link su `fftw3_threads` (dipendenza di ITK) a meno di
   usare `LIBRARY_PATH` come nella guida.

6. **Stream tracer disattivato automaticamente**: con VTK ≥ 9.2 (quindi anche
   con 9.5.2), `VTK_VMTK_BUILD_STREAMTRACER` viene disattivato di default
   (`vtkVmtk/CMakeLists.txt:47-53`) perché VMTK non è più compatibile con il
   rework dei campi di velocità interpolati introdotto in quella versione di
   VTK. Non è un errore, è previsto.

7. **`cmake --install` senza `--prefix` scrive in `/usr/local`**: VMTK usa
   path relativi (`lib/${VMTK_PYTHON_VERSION}/site-packages/vmtk`) risolti
   contro `CMAKE_INSTALL_PREFIX`, che di default è `/usr/local` — richiede
   `sudo` e comunque non è dove il Python dell'env conda cerca i pacchetti.
   Va sempre specificato `--prefix "$CONDA_PREFIX"` (o equivalente
   `-DCMAKE_INSTALL_PREFIX=...` in fase di configure).

8. **`import vtkvmtkCommonPython` diretto non è l'uso previsto**: i moduli
   compilati vanno importati tramite il modulo aggregatore
   `vmtk/vtkvmtk.py` (`from vmtk import vtkvmtk`), che usa import relativi
   (`from .vtkvmtkCommonPython import *`) assumendo che i `.so` stiano nella
   stessa directory del pacchetto `vmtk` — cosa vera solo dopo un install
   corretto (punto 7), non nel build tree grezzo.

## Versioni verificate in questa guida

| Componente | Versione |
|---|---|
| VMTK | commit `ba7cf0f` ("Adapted to build against VTK 9.6") |
| VTK | 9.5.2 (conda-forge) |
| ITK | 5.4.6 (conda-forge, + `libitk-devel` 5.4.6) |
| Python | 3.11.14 |
| CMake | via conda-forge `cmake` package |
| Compilatore | Clang 19.1.7 (conda-forge `compilers`) |
| Piattaforma | macOS osx-arm64 |
