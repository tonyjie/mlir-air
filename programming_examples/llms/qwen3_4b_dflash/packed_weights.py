# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Pure layout helpers extracted from pr1942-port 67831a05 dflash_prepass_waves.py.
# The AIR builders, process-environment mutations, and runtime are not imported.
from dataclasses import dataclass
import sys
import numpy as np
from target_config import FUSED
sys.path.insert(0, str(FUSED))
from proj_qmm_pack import BLOCK_BF16, COL_BLOCK, GROUP, N_GROUPS, PARALLEL, ROW_BLOCK

@dataclass(frozen=True)
class CascadeGeom:
    """Where `pack_q4k_cascade(iter_major=True, dual_chan=...)` puts each block.

    The packer emits `for cx: for (i, j, cy) in order:`, and with `dual_chan`
    the order hoists the ROW HALF `h` outermost so each of a column's two shim
    MM2S channels reads one contiguous DDR run. So a column's stream is

        [h=0: (i,j) in step order, cy in 0..half-1]
        [h=1: (i,j) in step order, cy in half..NCY-1]

    and that is what makes a row window cheap: restricting to `i` in
    [iter_lo, iter_lo+n_iter) is a CONTIGUOUS sub-run of each (cx, h) share,
    because `i` is the outer loop of `steps`. `_feed_wcol` already issues one
    put per channel, so selecting a window is a base offset and a length --
    no new descriptor shape.
    """

    NCX: int
    NCY: int
    nbi_pc: int  # row-block iterations per core
    nbj: int  # column-blocks = K / COL_BLOCK
    dual_chan: bool

    @property
    def half(self):
        return self.NCY // 2 if self.dual_chan else self.NCY

    @property
    def n_chan(self):
        return 2 if self.dual_chan else 1

    @property
    def blocks_per_step(self):
        """Blocks one (cx, h) sub-run emits per (i, j) step: one per cy."""
        return self.half

    @property
    def steps_per_chan(self):
        return self.nbi_pc * self.nbj

    @property
    def blocks_per_chan(self):
        return self.steps_per_chan * self.blocks_per_step

    @property
    def blocks_per_col(self):
        return self.blocks_per_chan * self.n_chan

    @property
    def n_blocks(self):
        return self.blocks_per_col * self.NCX

    def iter_window(self, row_lo, row_hi):
        """(iter_lo, n_iter) for matrix rows [row_lo, row_hi).

        Under `iter_major` the block for (cx, i, j, cy) is global row-block
        `gi = i*(NCX*NCY) + cx*NCY + cy`, so iteration `i` owns exactly the
        contiguous span of `NCX*NCY` row-blocks starting at `i*NCX*NCY` --
        `NCX*NCY*ROW_BLOCK` matrix rows. A window that is not iteration-aligned
        cannot be expressed as a feed offset, so refuse it here rather than
        emit a plausible wrong slab.
        """
        span = self.NCX * self.NCY * ROW_BLOCK
        if row_lo % span or row_hi % span:
            raise ValueError(
                f"rows [{row_lo}, {row_hi}) are not aligned to the {span}-row "
                f"iteration granule (NCX*NCY*ROW_BLOCK); a q/k/v split that "
                f"lands mid-iteration cannot be selected by a feed offset"
            )
        return row_lo // span, (row_hi - row_lo) // span

    def chan_window(self, iter_lo, n_iter):
        """(first_block, n_blocks) of that window WITHIN one (cx, h) sub-run."""
        return iter_lo * self.nbj * self.blocks_per_step, (
            n_iter * self.nbj * self.blocks_per_step
        )

    def block_coords(self):
        """(gi, j) per block, in emission order. The inverse of the packer."""
        out = []
        for cx in range(self.NCX):
            for h in range(self.n_chan):
                for i in range(self.nbi_pc):
                    for j in range(self.nbj):
                        for cyl in range(self.blocks_per_step):
                            cy = h * self.half + cyl
                            out.append(
                                (i * (self.NCX * self.NCY) + cx * self.NCY + cy, j)
                            )
        return out


def geom_for(M, K, fd):
    """The geometry `pack_q4k_cascade` will use for an [M, K] matrix."""
    nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
    n_cores = fd.NCX * fd.NCY
    if M % ROW_BLOCK or K % COL_BLOCK or nbi % n_cores:
        raise ValueError(
            f"[{M},{K}] does not tile: needs M % {ROW_BLOCK} == 0, "
            f"K % {COL_BLOCK} == 0 and (M/{ROW_BLOCK}) % {n_cores} == 0"
        )
    return CascadeGeom(
        NCX=fd.NCX,
        NCY=fd.NCY,
        nbi_pc=nbi // n_cores,
        nbj=nbj,
        dual_chan=bool(getattr(fd, "W_DUAL_CHAN", 0)),
    )


def unpack_blocks(packed):
    """(q, scale, mins) for every block in a packed slab, vectorized.

    `pack_q4k_block` is a per-block Python triple loop; inverting it the same
    way takes minutes on fc's 4000 blocks, and a gate nobody runs is not a gate.
    Layout (proj_qmm_pack.py:5-9):
        bf16  scales[8 groups][32 rows]   offset 0
        bf16  mins  [8 groups][32 rows]   offset 512
        uint8 qs[2 rowgrp][256 col][8]    offset 1024, byte = (w_odd<<4)|w_even
    """
    from ml_dtypes import bfloat16

    u8 = np.ascontiguousarray(packed).view(np.uint8).reshape(-1, BLOCK_BF16 * 2)
    nb = u8.shape[0]
    sm = 2 * N_GROUPS * ROW_BLOCK
    sc = u8[:, :sm].view(bfloat16).reshape(nb, N_GROUPS, ROW_BLOCK)
    mn = u8[:, sm : 2 * sm].view(bfloat16).reshape(nb, N_GROUPS, ROW_BLOCK)
    qs = u8[:, 2 * sm :].reshape(nb, ROW_BLOCK // PARALLEL, COL_BLOCK, PARALLEL // 2)
    # row = g*PARALLEL + kk*2 + p, p=0 low nibble (even row), p=1 high (odd)
    pair = np.stack([qs & 0xF, qs >> 4], axis=-1)  # [nb, g, col, kk, p]
    q = pair.transpose(0, 1, 3, 4, 2).reshape(nb, ROW_BLOCK, COL_BLOCK)
    # scales/mins come back group-major; the caller wants [block, row, group]
    return q, sc.transpose(0, 2, 1), mn.transpose(0, 2, 1)


def dequant_cascade(packed, M, K, geom, rows=None):
    """Reconstruct [M, K] (float32) from a cascade-packed slab.

    `rows` optionally restricts the output to a row range (lo, hi) so a gate on
    fc's 2560x12800 does not materialize 131 MB it will not look at.
    """
    q, sc, mn = unpack_blocks(packed)
    coords = geom.block_coords()
    if q.shape[0] != len(coords):
        raise ValueError(
            f"slab holds {q.shape[0]} blocks, geometry expects {len(coords)} "
            f"-- M/K or dual_chan disagree with how this was packed"
        )
    lo, hi = (0, M) if rows is None else rows
    out = np.zeros((hi - lo, K), np.float32)
    sc_e = np.repeat(sc.astype(np.float32), GROUP, axis=2)
    mn_e = np.repeat(mn.astype(np.float32), GROUP, axis=2)
    w = q.astype(np.float32) * sc_e + mn_e
    for b, (gi, j) in enumerate(coords):
        r0 = gi * ROW_BLOCK
        if r0 + ROW_BLOCK <= lo or r0 >= hi:
            continue
        out[r0 - lo : r0 - lo + ROW_BLOCK, j * COL_BLOCK : (j + 1) * COL_BLOCK] = w[b]
    return out
