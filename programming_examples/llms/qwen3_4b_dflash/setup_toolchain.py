# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build this branch's AIR compiler with pinned upstream dependencies.

Default: create a Python 3.12 environment and download dependencies.
--reuse-deps: use active Python and configured upstream dependency installs.
The system XRT and NPU driver must already be installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCK = json.loads((HERE / "toolchain.lock.json").read_text())
RELEASES = "https://github.com/Xilinx/mlir-aie/releases"


def run(command, **kwargs):
    print("+ " + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def output(command):
    return subprocess.check_output(list(map(str, command)), text=True).strip()


def package_path(python, package, directory):
    return Path(output([python, "-c", "import importlib.metadata as m; "
                        f"print(m.distribution({package!r}).locate_file({directory!r}))"]))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reuse-deps", action="store_true")
    ap.add_argument("--build-dir", type=Path, default=REPO / "build_dflash_toolchain")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--xrt-dir", type=Path, default=Path("/opt/xilinx/xrt"))
    ap.add_argument("--llvm-link", type=Path)
    ap.add_argument("--cmake-modules", type=Path)
    ap.add_argument("--configure-only", action="store_true")
    args = ap.parse_args()
    if sys.version_info[:2] != (3, 12):
        ap.error("Use Python 3.12, matching the tested compiler and XRT Python ABI")
    if args.jobs < 1:
        ap.error("--jobs must be positive")
    xrt = args.xrt_dir.resolve()
    if not (xrt / "include").is_dir() or not (xrt / "python").is_dir():
        ap.error("Install XRT with NPU support and Python bindings first; see docs/reviewer.md")
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    if args.reuse_deps:
        python = Path(sys.executable)
        paths = {}
        for key in ("MLIR_AIE_INSTALL_DIR", "PEANO_INSTALL_DIR", "LLVM_INSTALL_DIR"):
            if not os.environ.get(key):
                ap.error(f"--reuse-deps requires {key}; source your upstream environment first")
            paths[key] = Path(os.environ[key]).resolve()
    else:
        venv = build / "venv"
        if not (venv / "bin/python").exists():
            run([sys.executable, "-m", "venv", venv])
        python = venv / "bin/python"
        run([python, "-m", "pip", "install", "-r", HERE / "requirements-review.txt"])
        run([python, "-m", "pip", "install", f"mlir_aie=={LOCK['mlir_aie']}",
             "-f", RELEASES + "/expanded_assets/latest-wheels-4/"])
        run([python, "-m", "pip", "install", f"llvm-aie=={LOCK['llvm_aie']}",
             "-f", "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"])
        deps = build / "deps"
        deps.mkdir(exist_ok=True)
        if not (deps / "mlir/bin/mlir-opt").exists():
            run([python, "-m", "pip", "download", "--no-deps", f"mlir=={LOCK['mlir']}",
                 "-d", deps, "-f", RELEASES + "/expanded_assets/mlir-distro"])
            wheels = list(deps.glob(f"mlir-{LOCK['mlir']}-*.whl"))
            if len(wheels) != 1:
                raise RuntimeError(f"Expected one pinned LLVM wheel, found {wheels}")
            with zipfile.ZipFile(wheels[0]) as archive:
                archive.extractall(deps)
                for member in archive.infolist():
                    mode = member.external_attr >> 16
                    if mode and not member.is_dir():
                        (deps / member.filename).chmod(mode & 0o777)
        paths = {
            "MLIR_AIE_INSTALL_DIR": package_path(python, "mlir-aie", "mlir_aie"),
            "PEANO_INSTALL_DIR": package_path(python, "llvm-aie", "llvm-aie"),
            "LLVM_INSTALL_DIR": deps / "mlir",
        }

    versions = json.loads(output([python, "-c", "import importlib.metadata as m,json; "
        "print(json.dumps({p:m.version(p) for p in ['mlir-aie','llvm-aie','numpy','ml_dtypes','transformers']}))"]))
    for package, key in (("mlir-aie", "mlir_aie"), ("llvm-aie", "llvm_aie")):
        if versions[package] != LOCK[key]:
            ap.error(f"{package} is {versions[package]}, expected {LOCK[key]}")
    for path in paths.values():
        if not path.is_dir():
            ap.error(f"Missing toolchain directory: {path}")
    host_llvm = output([paths["LLVM_INSTALL_DIR"] / "bin/mlir-opt", "--version"])
    if "LLVM version 24" not in host_llvm:
        ap.error(f"Expected pinned LLVM 24 host tools, found {host_llvm}")

    link = args.llvm_link or (Path(os.environ["DFLASH_LLVM_LINK"]) if os.environ.get("DFLASH_LLVM_LINK") else None)
    if link is None:
        link = build / "llvm-link-bin/llvm-link"
        if not link.exists():
            wheel = build / LOCK["llvm_link_wheel"]
            if not wheel.exists():
                url = RELEASES + "/download/mlir-distro/" + urllib.parse.quote(wheel.name)
                print("Downloading LLVM 21 wheel for llvm-link: " + url, flush=True)
                urllib.request.urlretrieve(url, wheel)
            link.parent.mkdir(exist_ok=True)
            with zipfile.ZipFile(wheel) as archive:
                link.write_bytes(archive.read("mlir/bin/llvm-link"))
            link.chmod(0o755)
    link = link.expanduser().resolve()
    link_version = output([link, "--version"])
    if "LLVM version 21" not in link_version:
        ap.error("The reviewer setup requires the validated LLVM 21 llvm-link")
    modules = args.cmake_modules
    if modules is None:
        modules = build / "cmakeModules"
        if not modules.exists():
            run(["git", "clone", "https://github.com/Xilinx/cmakeModules.git", modules])
        run(["git", "-C", modules, "checkout", "--detach", LOCK["cmake_modules_commit"]])
    modules = modules.resolve()
    if not (modules / "FindXRT.cmake").is_file():
        ap.error(f"Invalid cmakeModules directory: {modules}")

    env = dict(os.environ)
    env["PATH"] = f"{python.parent}:{paths['MLIR_AIE_INSTALL_DIR']}/bin:/usr/bin:/bin:" + os.environ["PATH"]
    env["PYTHONPATH"] = f"{paths['MLIR_AIE_INSTALL_DIR']}/python:{xrt}/python"
    env["LD_LIBRARY_PATH"] = ":".join(str(p / "lib") for p in paths.values()) + f":{xrt}/lib"
    install = build / "install"
    cmake = build / "compiler"
    run([python.parent / "cmake", "-S", REPO, "-B", cmake, "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=/usr/bin/clang",
         "-DCMAKE_CXX_COMPILER=/usr/bin/clang++", f"-DCMAKE_INSTALL_PREFIX={install}",
         f"-DCMAKE_MODULE_PATH={modules}", f"-DLLVM_DIR={paths['LLVM_INSTALL_DIR']}/lib/cmake/llvm",
         f"-DMLIR_DIR={paths['LLVM_INSTALL_DIR']}/lib/cmake/mlir",
         f"-DAIE_DIR={paths['MLIR_AIE_INSTALL_DIR']}/lib/cmake/aie",
         f"-DPEANO_INSTALL_DIR={paths['PEANO_INSTALL_DIR']}",
         f"-DPython_EXECUTABLE={python}", f"-DPython3_EXECUTABLE={python}",
         "-DPython_FIND_VIRTUALENV=ONLY", "-DPython3_FIND_VIRTUALENV=ONLY",
         f"-DLLVM_EXTERNAL_LIT={python.parent}/lit", "-DLLVM_USE_LINKER=lld",
         "-DLLVM_ENABLE_ASSERTIONS=ON", "-DBUILD_SHARED_LIBS=OFF",
         "-DAIR_ENABLE_AIE=ON", "-DAIR_RUNTIME_TARGETS=x86_64", "-DAIE_ENABLE_BINDINGS_PYTHON=ON",
         f"-Dx86_64_TOOLCHAIN_FILE={REPO}/cmake/modules/toolchain_x86_64.cmake",
         f"-DXRT_ROOT={xrt}", f"-DXRT_BIN_DIR={xrt}/bin", f"-DXRT_LIB_DIR={xrt}/lib",
         f"-DXRT_INCLUDE_DIR={xrt}/include", "-DENABLE_RUN_XRT_TESTS=ON"], env=env)
    if args.configure_only:
        return
    run([python.parent / "cmake", "--build", cmake, "--target", "install", "-j", args.jobs], env=env)
    metadata = {"schema": "dflash.air.build.v1",
        "source_commit": output(["git", "-C", REPO, "rev-parse", "HEAD"]),
        "source_dirty": bool(output(["git", "-C", REPO, "status", "--porcelain", "--", "mlir", "python", "CMakeLists.txt"])),
        "python": str(python), "versions": versions, "host_llvm": host_llvm,
        "llvm_link": link_version, "air_opt_sha256": sha(install / "bin/air-opt"),
        "compiler_sources": {str(p.relative_to(REPO)): sha(p)
            for p in sorted((REPO / "mlir").rglob('*')) if p.is_file() and p.suffix in ('.cpp','.h','.td')},
        "dependencies": {k:str(v) for k,v in paths.items()}, "xrt": str(xrt)}
    (install / "dflash-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    lines = ["# Source this file to use the reviewer AIR build.",
             f"source {shlex.quote(str(python.parent / 'activate'))}",
             f"source {shlex.quote(str(xrt / 'setup.sh'))} >/dev/null",
             "unset DFLASH_BASE_ENV DFLASH_AIR_INSTALL",
             f"export MLIR_AIR_INSTALL_DIR={shlex.quote(str(install))}"]
    lines += [f"export {k}={shlex.quote(str(v))}" for k,v in paths.items()]
    lines += [f"export DFLASH_LLVM_LINK={shlex.quote(str(link))}",
        'export PATH="$MLIR_AIR_INSTALL_DIR/bin:$MLIR_AIE_INSTALL_DIR/bin:$LLVM_INSTALL_DIR/bin:$PATH"',
        f'export PYTHONPATH="$MLIR_AIR_INSTALL_DIR/python:$MLIR_AIE_INSTALL_DIR/python:{xrt}/python"',
        f'export LD_LIBRARY_PATH="$MLIR_AIR_INSTALL_DIR/lib:$MLIR_AIE_INSTALL_DIR/lib:$LLVM_INSTALL_DIR/lib:{xrt}/lib"']
    (build / "env.sh").write_text("\n".join(lines) + "\n")
    print(f"Ready: source {build / 'env.sh'}", flush=True)


if __name__ == "__main__":
    main()
