"""External C++ kernel compilation utilities.

Compiles all external .o files from source to avoid relying on stale
pre-compiled artifacts. Each function checks if the .o exists and skips
recompilation if so (delete the .o to force recompile).

Compiled .o files are placed in CWD (build_peano/) where aiecc finds them
via its link_with search path.
"""

import os
import subprocess
from pathlib import Path


def _get_peano_clang():
    """Find the Peano clang++ compiler."""
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_dir:
        return os.path.join(peano_dir, "bin", "clang++")
    raise RuntimeError("PEANO_INSTALL_DIR not set")


def _get_aie_include_dir():
    """Find the AIE API include directory (for aie_api/aie.hpp)."""
    # Try mlir-aie install path
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent
        / "my_install"
        / "mlir-aie"
        / "install"
        / "include",
    ]
    for p in candidates:
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    # Fallback: search from PEANO_INSTALL_DIR
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_dir:
        p = Path(peano_dir).parent.parent / "include"
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    raise RuntimeError("Cannot find aie_api/aie.hpp include directory")


_PEANO_FLAGS = [
    "-O2",
    "-std=c++20",
    "--target=aie2p-none-unknown-elf",
    "-DNDEBUG",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
]


def _compile_kernel(src_path, output_name, extra_flags=None, force=False):
    """Compile a C++ kernel to .o using Peano clang++.

    Args:
        src_path: Path to the .cc source file
        output_name: Name of the output .o file (placed in CWD)
        extra_flags: Additional compiler flags (e.g., -D defines)
        force: If True, recompile even if .o exists
    """
    if not force and Path(output_name).exists():
        return

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Kernel source not found: {src}")

    clang = _get_peano_clang()
    include_dir = _get_aie_include_dir()

    cmd = [clang] + _PEANO_FLAGS + [f"-I{include_dir}"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend(["-c", str(src), "-o", output_name])

    print(f"  Compiling {output_name} from {src.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Filter warnings, only show errors
        errors = [l for l in result.stderr.split("\n") if "error" in l.lower()]
        raise RuntimeError(f"Failed to compile {output_name}: {' '.join(errors[:3])}")


# ---------------------------------------------------------------------------
# Individual kernel compilation functions
# ---------------------------------------------------------------------------

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent  # programming_examples/


def compile_silu_and_mul():
    """Compile silu_and_mul.o from _llm_shared/kernel_builder/ffn_swiglu/silu_and_mul.cc."""
    src = Path(__file__).resolve().parent / "ffn_swiglu" / "silu_and_mul.cc"
    include_dir = _get_aie_include_dir()
    utils_header = Path(include_dir) / "aie_kernels" / "aie_kernel_utils.h"
    extra = []
    if utils_header.exists():
        extra = [f"-include", str(utils_header)]
    _compile_kernel(src, "silu_and_mul.o", extra_flags=extra)


def compile_rope():
    """Compile rope.o from our half-split RoPE kernel.

    Uses rope_halfsplit.cc (half-split rotation matching HuggingFace Llama)
    instead of upstream rope.cc (interleaved rotation). Same function name
    (@rope) and signature, so no MLIR changes needed.
    """
    src = Path(__file__).resolve().parent / "rope_halfsplit.cc"
    _compile_kernel(src, "rope.o")


def compile_attn_npu2(head_dim=64):
    """Compile attn_npu2.o (FlashAttention kernel) — back-compat wrapper.

    Defaults `lqp=lkp=dk=dv=head_dim`. This is the right setup for
    head_dim ≤ 64 (Q+K share via shared-buffers fits 64 KB L1). For
    head_dim ≥ 128 the per-tile L1 footprint exceeds 64 KB; use
    `compile_attn_npu2_split` instead with `lkp != dk` (e.g.,
    lkp=64, dk=128 yields dk_chunks=2 and a feasible L1 budget).

    IMPORTANT: passes `num_q_tiles=1` so the emitted `-Dlqp` define
    equals `head_dim` (the kernel's per-tile Q size that the IR builder
    in `attn_npu2_seqfirst` calls with at lqp=256, num_q_tiles=4 → 64
    per tile for head_dim=64). After the LESSON 3 refactor (commit
    6499cae0, 2026-04-18) `compile_attn_npu2_split` started dividing
    `lqp` by `num_q_tiles` (default 4), which silently broke llama3's
    flash_attn — kernel compiled with -Dlqp=16 vs IR providing 64-row
    tiles → all-NaN at Layer 1+. Caught by evaluate-deployment v2
    cross-deployment regression on 2026-04-19. Setting num_q_tiles=1
    here keeps the back-compat wrapper's `-Dlqp` equal to `head_dim`.
    """
    compile_attn_npu2_split(
        lqp=head_dim,
        lkp=head_dim,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=1,
    )


def compile_attn_npu2_split(lqp, lkp, dk, dv, num_q_tiles=4, output_name="attn_npu2.o"):
    """Compile attn_npu2.o with explicit (lqp, lkp, dk, dv) tile parameters.

    Use this when `lkp != dk` is required for L1 budget — e.g., Llama-3.2-3B
    and Llama-3-8B at head_dim=128 cannot use the default `lkp=dk=128`
    (per-core L1 ≈ 74 KB > 64 KB even with shared buffers). The L1-feasible
    config is `lqp=256, lkp=64, dk=dv=128` which gives `dk_chunks = dk/lkp = 2`
    and ≈ 50 KB L1 (proven by `flash_attention/.../run_npu2_makefile_peano_llama3_8b.lit`).

    IMPORTANT: the C++ kernel's `dk`/`dv` defines are **per-tile inner
    dimensions** (must equal `lkp`), and `dk_full`/`dv_full` are the **full
    head dimensions**. This matches the Makefile's flag convention exactly:
        -Dlqp=$(LQP_TILE)  -Dlkp=$(LKP)
        -Ddk=$(LKP)        -Ddk_full=$(DK)
        -Ddv=$(LKP)        -Ddv_full=$(DV)

    Passing `-Ddk=DK` (not `-Ddk=LKP`) when `dk_chunks > 1` produces a kernel
    that does the wrong per-tile arithmetic and outputs all-NaN at runtime —
    discovered the hard way during llama32_3b Phase 4 (2026-04-18).

    Args:
        lqp: Q chunk size per launch iteration (must be divisible by 4 for
             num_q_tiles=4; tile_size_q = lqp/4).
        lkp: K/V chunk size per iteration. dk_chunks = dk/lkp; lkp must
             divide dk. The kernel's per-tile dk/dv equal lkp.
        dk:  Full key/Q head dimension. The .o's `dk_full` macro.
        dv:  Full value head dimension. The .o's `dv_full` macro.
        output_name: .o file to produce (default 'attn_npu2.o'). Override
             when emitting multiple FA kernel variants in the same build.
    """
    assert dk % lkp == 0, f"dk={dk} must be divisible by lkp={lkp}"
    assert dv % lkp == 0, f"dv={dv} must be divisible by lkp={lkp}"
    assert (
        lqp % num_q_tiles == 0
    ), f"lqp={lqp} must be divisible by num_q_tiles={num_q_tiles}"
    lqp_tile = lqp // num_q_tiles  # per-tile Q size — what the .o expects
    src = _PROJ_ROOT / "flash_attention" / "kernel_fusion_based" / "attn_npu2.cc"
    _compile_kernel(
        src,
        output_name,
        extra_flags=[
            "-DBIT_WIDTH=8",
            f"-Dlqp={lqp_tile}",  # per-tile Q rows (= lqp / num_q_tiles)
            f"-Dlkp={lkp}",
            f"-Ddk={lkp}",  # per-tile dk == lkp (NOT the full head dim)
            f"-Ddk_full={dk}",  # full head dim
            f"-Ddv={lkp}",  # per-tile dv == lkp
            f"-Ddv_full={dv}",  # full head dim (== dk for square Q/V)
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DROUND_CONV_EVEN",
        ],
    )
    if output_name == "attn_npu2.o" and not Path("attn.o").exists():
        import shutil

        shutil.copy2("attn_npu2.o", "attn.o")


def compile_mv_k8192():
    """Compile mv_k8192.o with renamed GEMV symbols for K=8192 decode merge."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(
        src,
        "mv_k8192.o",
        extra_flags=[
            "-DDIM_M_OUTPUT=2",
            "-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16",
            "-Dlinalg_fill_bf16=dg_linalg_fill_bf16",
        ],
    )


def compile_mv(tile_m=8):
    """Compile mv.o (standard GEMV kernel) from source."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(src, "mv.o", extra_flags=[f"-DDIM_M_OUTPUT={tile_m}"])


def compile_mv_og(tile_m=8):
    """Compile mv_og.o with `og_matvec_*` renamed symbols.

    For Qwen3 decode o_gemv_ffn: O GEMV at K=2048 collides with Gate/Up at
    K=1024 in the same ELF (different memref<m_input × K> signatures hashing
    to the same `@matvec_vectorized_bf16_bf16` symbol). Rename via -D so O
    can link against its own .o copy and coexist with default mv.o + mv_dg.o.
    Functionally identical to mv.o (DIM_M_OUTPUT=tile_m); only symbol differs.
    """
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(
        src,
        "mv_og.o",
        extra_flags=[
            f"-DDIM_M_OUTPUT={tile_m}",
            "-Dmatvec_vectorized_bf16_bf16=og_matvec_vectorized_bf16_bf16",
            "-Dlinalg_fill_bf16=og_linalg_fill_bf16",
        ],
    )


def compile_mv_dg_qwen3(tile_m=8):
    """Compile mv_dg_qwen3.o with `dg_matvec_*` renamed symbols at tile_m=8.

    The existing `compile_mv_k8192()` produces mv_k8192.o with DIM_M_OUTPUT=2
    (for llama3's down_tile_m=2 at K=8192). Qwen3 decode uses down_tile_m=8
    (K=3072 fits the standard tile config). We need a separate copy with
    DIM_M_OUTPUT=8 + the same `dg_*` rename so it can link alongside the
    Qwen3 o+ffn ELF without overwriting llama3's mv_k8192.o on disk.
    """
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(
        src,
        "mv_dg_qwen3.o",
        extra_flags=[
            f"-DDIM_M_OUTPUT={tile_m}",
            "-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16",
            "-Dlinalg_fill_bf16=dg_linalg_fill_bf16",
        ],
    )


def compile_all_external_kernels(head_dim=64):
    """Compile all external C++ kernels from source.

    Call this before kernel compilation to ensure all .o files are fresh.
    Each kernel is only compiled if its .o doesn't already exist.
    Delete build_peano/*.o to force recompilation.
    """
    compile_silu_and_mul()
    compile_rope()
    compile_attn_npu2(head_dim=head_dim)
    compile_mv()
    compile_mv_k8192()
