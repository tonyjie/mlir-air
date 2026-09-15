# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-4B block-8 AIR dataflow, bound to explicit geometry by target_config.

The projection/feed/egress schedule is adapted from pr1942-port 67831a05.
Only its resident-residual, dual-weight-channel Qwen path is retained. Current
main supplies model geometry and primitive kernels. Attention uses a single
flattened query/key-block loop, so the shared-score handoff occurs for every
key block with current AIR's outermost-loop lock policy.

This prototype is deliberately limited to block 8. Block 16 requires a new
residual-storage and attention tiling design, not changing a constant here.
"""

def build_module():

    @module_builder
    def build():
        bf16 = BF16Type.get()
        f32 = F32Type.get()
        i32 = IntegerType.get_signless(32)
        idx_t = IndexType.get()
        l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l2 = IntegerAttr.get(T.i32(), MemorySpace.L2)
        x_l3 = MemRefType.get([X_SLOTS * BATCH * K], bf16)
        _w_blocks = UNI_DEC * W_TOTAL_BLOCKS + UNI_LM * VOCAB_W_BLOCKS
        w_l3 = MemRefType.get([_w_blocks * BLOCK_BF16], bf16)
        _wgrp_len = [min(W_GROUP, UNI_DEC - g * W_GROUP) * W_LAYER for g in range(N_WGRP)]
        _wvoc_len = UNI_LM * VOCAB_W_BLOCKS * BLOCK_BF16
        _w_extra = []
        rms_l3 = MemRefType.get([UNI_DEC * RMS_LAYER + UNI_DEC * ROPE_W_LEN * BATCH + K + 0 + 0 + 0], bf16)
        _y_elems = (HOST_ROUNDS + LAYER_RNDS) * PAYLOAD * BATCH + UNI_LM * VOCAB_SIZE_PADDED * BATCH + 0
        _y_scratch_off = _y_elems
        _y_elems += 0
        y_l3 = MemRefType.get([_y_elems], bf16)
        kvc_l3 = MemRefType.get([UNI_DEC * ATTN_MAXL * KVSZ_TOK], bf16)
        xblk_l1 = MemRefType.get([COL_BLOCK], bf16, memory_space=l1)
        wblk_l1 = MemRefType.get([BLOCK_BF16], bf16, memory_space=l1)
        yacc_l1 = MemRefType.get([ROW_BLOCK], f32, memory_space=l1)
        rcache_l1 = MemRefType.get([RCACHE_LEN], bf16, memory_space=l1)
        ypair_l1 = MemRefType.get([16 + PAIR_ROWS * ROW_BLOCK], bf16, memory_space=l1)
        xblk_mm_l1 = MemRefType.get([BATCH * COL_BLOCK], bf16, memory_space=l1)
        yacc_mm_l1 = MemRefType.get([BATCH * ROW_BLOCK], f32, memory_space=l1)
        wscr_mm_l1 = MemRefType.get([ROW_BLOCK * COL_BLOCK], bf16, memory_space=l1)
        _YPAIR = 16 + PAIR_ROWS * ROW_BLOCK * PACKET_BATCH
        ypair_mm_l1 = MemRefType.get([_YPAIR + -_YPAIR % L1_VEC_BF16], bf16, memory_space=l1)
        rms_l1 = MemRefType.get([K], bf16, memory_space=l1)
        rmsb_l1 = MemRefType.get([BATCH * K], bf16, memory_space=l1)
        rstg_l1 = MemRefType.get([BATCH * STG_W], bf16, memory_space=l1)
        rscl_l1 = MemRefType.get([BATCH], T.f32(), memory_space=l1)
        rms_w2k_l1 = MemRefType.get([2 * K], bf16, memory_space=l1)
        glu_x_l1 = MemRefType.get([BATCH * GLU_SLICE], bf16, memory_space=l1)
        glu_hid_l1 = MemRefType.get([BATCH * GLU_HID], bf16, memory_space=l1)
        qkv_l1 = MemRefType.get([M], bf16, memory_space=l1)
        ropeq_l1 = MemRefType.get([DQ_PADDED], bf16, memory_space=l1)
        ropekv_l1 = MemRefType.get([DK], bf16, memory_space=l1)
        ropelut_l1 = MemRefType.get([ROPE_W_LEN], bf16, memory_space=l1)
        aq_l1 = MemRefType.get([BATCH * DQ_PADDED_PER_CU], bf16, memory_space=l1)
        ak_l1 = MemRefType.get([16 * KVPC_DH], bf16, memory_space=l1)
        av_l1 = MemRefType.get([16 * KVPC_DH], bf16, memory_space=l1)
        as_l1 = MemRefType.get([ATTENTION_QUERY_TILE * SSZ_BLK], bf16, memory_space=l1)
        ao_l1 = MemRefType.get([BATCH * DQ_PADDED_PER_CU], bf16, memory_space=l1)
        ak_l2 = MemRefType.get([16 * KVPC_DH], bf16, memory_space=l2)
        av_l2 = MemRefType.get([16 * KVPC_DH], bf16, memory_space=l2)
        qkvmt_l2 = MemRefType.get([M], bf16, memory_space=l2)
        qmt_l2 = MemRefType.get([BATCH * DQ_PADDED], bf16, memory_space=l2)
        omt_l2 = MemRefType.get([BATCH * DQ], bf16, memory_space=l2)
        kvblk_l2 = MemRefType.get([2 * 16 * KVPC_DH], bf16, memory_space=l2)
        bufp2_l2 = MemRefType.get([K], bf16, memory_space=l2)
        qkvmt_l2 = MemRefType.get([BATCH * M], bf16, memory_space=l2)
        xmt_l2 = MemRefType.get([BATCH * XCHUNK], bf16, memory_space=l2)
        xmt_wide_l2 = MemRefType.get([BATCH * K], bf16, memory_space=l2)
        wfan_l2 = MemRefType.get([NCY // 2 * BLOCK_BF16], bf16, memory_space=l2)
        grp_l2 = MemRefType.get([GRP_ROWS_B], bf16, memory_space=l2)
        main_l2 = MemRefType.get([MAIN_ROWS_B], bf16, memory_space=l2)
        relay_l2 = MemRefType.get([PAYLOAD_B], bf16, memory_space=l2)
        down_l2 = MemRefType.get([BATCH * GLU_OUT], bf16, memory_space=l2)
        RELAY_COLS = [3, 5, 4][:NDEST]
        RELAY_COLS[0] = int(RELAY_COLS[0])
        zero = FuncOp('proj_qmm_zero', ([yacc_l1, i32], []), visibility='private')
        zero.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        acc256 = FuncOp('proj_qmm_acc256', ([xblk_l1, wblk_l1, yacc_l1], []), visibility='private')
        acc256.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        acc256_c = FuncOp('proj_qmm_acc256_c', ([xblk_l1, wblk_l1, yacc_l1, rcache_l1, i32, i32], []), visibility='private')
        acc256_c.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        rc_arm = FuncOp('proj_qmm_rc_arm', ([rcache_l1, i32], []), visibility='private')
        rc_arm.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        flush_row = FuncOp('proj_qmm_flush_row', ([yacc_l1, ypair_l1, i32], []), visibility='private')
        flush_row.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        mm_zero = FuncOp('proj_qmm_mm_zero', ([yacc_mm_l1, i32], []), visibility='private')
        mm_zero.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        mm_acc = FuncOp('proj_qmm_mm_acc', ([xblk_mm_l1, wblk_l1, yacc_mm_l1, wscr_mm_l1], []), visibility='private')
        mm_acc.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        mm_flush = FuncOp('proj_qmm_mm_flush_token' if TOKEN_PACKETS else 'proj_qmm_mm_flush_routed' if CIRCUIT_MAIN else 'proj_qmm_mm_flush_row', ([yacc_mm_l1, ypair_mm_l1, i32, i32], []), visibility='private')
        mm_flush.attributes['link_with'] = StringAttr.get('proj_qmm.o')
        rms_norm_aie = FuncOp('rms_norm_aie', ([rms_l1, rms_l1, rms_l1, i32], []), visibility='private')
        rms_norm_aie.attributes['link_with'] = StringAttr.get('rms_residual.o')
        rms_scale_row_aie = FuncOp('rms_scale_row_aie', ([rscl_l1, rmsb_l1, i32, i32], []), visibility='private')
        rms_scale_row_aie.attributes['link_with'] = StringAttr.get('rms_residual.o')
        rms_chunk_aie = FuncOp('rms_chunk_aie', ([rstg_l1, rmsb_l1, rms_l1, rscl_l1, i32, i32, i32], []), visibility='private')
        rms_chunk_aie.attributes['link_with'] = StringAttr.get('rms_residual.o')
        residual_acc_row_aie = FuncOp('residual_acc_row_aie', ([rmsb_l1, rstg_l1, i32, i32, i32], []), visibility='private')
        residual_acc_row_aie.attributes['link_with'] = StringAttr.get('rms_residual.o')
        residual_add_aie = FuncOp('residual_add_aie', ([rms_l1, rms_l1, rms_l1], []), visibility='private')
        residual_add_aie.attributes['link_with'] = StringAttr.get('rms_residual.o')
        glu_aie = FuncOp('glu_aie', ([glu_hid_l1, glu_x_l1, i32], []), visibility='private')
        glu_aie.attributes['link_with'] = StringAttr.get('glu.o')
        glu_row_aie = FuncOp('glu_row_aie', ([glu_hid_l1, glu_x_l1, i32, i32], []), visibility='private')
        glu_row_aie.attributes['link_with'] = StringAttr.get('glu.o')
        glu_copy_aie = FuncOp('glu_copy_aie', ([glu_hid_l1, glu_x_l1, i32, i32, i32], []), visibility='private')
        glu_copy_aie.attributes['link_with'] = StringAttr.get('glu.o')
        rope_compute = FuncOp('rope_compute', ([ropeq_l1, ropekv_l1, ropekv_l1, qkv_l1, ropelut_l1, i32], []), visibility='private')
        rope_compute.attributes['link_with'] = StringAttr.get('rope.o')
        m_l1 = MemRefType.get([ATTENTION_QUERY_TILE * 16], bf16, memory_space=l1)
        c_l1 = MemRefType.get([ATTENTION_QUERY_TILE * 8], f32, memory_space=l1)
        y_l1 = MemRefType.get([ATTENTION_QUERY_TILE * DQ_PADDED_PER_CU], f32, memory_space=l1)
        lden_l1 = MemRefType.get([ATTENTION_QUERY_TILE * 16], f32, memory_space=l1)
        if not PARALLEL_ATTENTION:
            attn_qk_blk = FuncOp('attn_qk_blk', ([aq_l1, ak_l1, m_l1, c_l1, as_l1, i32, i32], []), visibility='private')
            _set_attn_link(attn_qk_blk, 'attn_qk')
            attn_qk_blk_row = FuncOp('attn_qk_blk_tile' if ATTENTION_QUERY_TILE > 1 else 'attn_qk_blk_row', ([aq_l1, ak_l1, m_l1, c_l1, as_l1, i32, i32, i32] + ([i32] if ATTENTION_QUERY_TILE > 1 else []), []), visibility='private')
            _set_attn_link(attn_qk_blk_row, 'attn_qk')
            attn_kv_blk = FuncOp('attn_kv_blk_tile' if ATTENTION_QUERY_TILE > 1 else 'attn_kv_blk', ([as_l1, av_l1, y_l1, lden_l1, i32, i32] + ([i32] if ATTENTION_QUERY_TILE > 1 else []), []), visibility='private')
            _set_attn_link(attn_kv_blk, 'attn_kv')
            attn_kv_fin = FuncOp('attn_kv_fin', ([y_l1, lden_l1, ao_l1], []), visibility='private')
            _set_attn_link(attn_kv_fin, 'attn_kv')
            attn_kv_fin_row = FuncOp('attn_kv_fin_tile' if ATTENTION_QUERY_TILE > 1 else 'attn_kv_fin_row', ([y_l1, lden_l1, ao_l1, i32], []), visibility='private')
            _set_attn_link(attn_kv_fin_row, 'attn_kv')
        parallel_attention = ParallelAttentionGraph(ATTENTION_ENV) if PARALLEL_ATTENTION else None
        if NORMALIZATION_CACHE:
            assert BATCH == 8 and XCHUNK == 512
            cache_copy = FuncOp('normalized_cache_chunk', ([rstg_l1, rmsb_l1, i32], []), visibility='private')
            cache_copy.attributes['link_with'] = StringAttr.get('norm_cache.o')
        _rx_packet = FULL4 and (not (0 >= 3 and BATCH > 1))
        if None:
            _rx_packet = FULL4
        if _rx_packet:
            _rx = channel_decl('rmsX', size=[1], channel_type='npu_dma_packet')
        else:
            _rx = channel_decl('rmsX', size=[1])
        channel_decl('rmsW', size=[1])
        _rw2 = channel_decl('rmsW2', size=[1])
        if 0 >= 3 or None:
            _rx.operation.attributes['air.tile_dma_channel'] = IntegerAttr.get(i32, 0)
        # With replay, xnorm is a single producer/consumer stream into a full
        # normalized slab. A circuit stream lets five 8-KiB core chunks fill
        # one 40-KiB slab without packet-boundary aggregation or shared arbiters.
        _xn = channel_decl('xnorm', size=[1], **(
            {'channel_type': 'npu_dma_packet'} if NORMALIZATION_REPLAY == 1 else {}))
        if NORMALIZATION_REPLAY > 1:
            # Use one constant replay count for every phase and arm. This uses
            # current AIR's buffer-level refeed semantics, not per-fill counts.
            assert all(n % NORMALIZATION_REPLAY == 0 for n in (REFEED[0], GATEUP_REFEED, VOCAB_RNDS))
            channel_decl('xin', size=[1], channel_type='npu_dma_packet')
        if NORMALIZATION_CACHE:
            _fill = channel_decl('xnorm_fill', size=[1])
            _fill.operation.attributes['air.tile_dma_channel'] = IntegerAttr.get(i32, 0)
        _XCHUNK = XCHUNK

        def _xnorm_put(buf, width, ssa=False, **kw):
            assert width % _XCHUNK == 0, f'@xnorm producer width {width} is not a multiple of {_XCHUNK}; the X memtile stages whole chunks'
            return ChannelPut('xin' if NORMALIZATION_REPLAY > 1 else 'xnorm', buf, offsets=[idx(0), idx(0), idx(0)], sizes=[idx(width // _XCHUNK), idx(BATCH), idx(_XCHUNK)], strides=[idx(_XCHUNK), idx(width), idx(1)], **kw)
        _inX = Channel('inX', size=[1, 1], broadcast_shape=[NCX, NCY])
        _inX.operation.attributes['air.shared_resident_ring'] = UnitAttr.get()
        channel_decl('ropeLUT', size=[1])
        channel_decl('toRope', size=[1])
        _ropeQ = channel_decl('ropeQ', size=[1])
        channel_decl('toAttnQ', size=[N_ATTN_CU])
        channel_decl('toAttnKV', size=[len(ATTN_COL_GROUPS)], channel_type='npu_dma_packet')
        channel_decl('toK', size=[N_ATTN_CU])
        channel_decl('toV', size=[N_ATTN_CU])
        _apK = channel_decl('appendK', size=[1], channel_type='npu_dma_packet')
        _apV = channel_decl('appendV', size=[1], channel_type='npu_dma_packet')
        channel_decl('inKV_K', size=[len(ATTN_COL_GROUPS)])
        channel_decl('inKV_V', size=[len(ATTN_COL_GROUPS)])
        channel_decl('attnO', size=[N_ATTN_CU])
        for _wc in range(NCX):
            for _nm in (_wname(0, _wc), _wname(1, _wc)):
                channel_decl(_nm, size=[1])
        _wL2 = channel_decl('wL2ToL1', size=[NCX, NCY])
        _wL2.operation.attributes['air.shared_resident_ring'] = UnitAttr.get()
        _gather_type = {} if CIRCUIT_GATHER else {'channel_type': 'npu_dma_packet'}
        _outA = channel_decl('outA', size=[NCX, PAIRS_PC], **_gather_type)
        _main_type = {} if CIRCUIT_MAIN else {'channel_type': 'npu_dma_packet'}
        _toMain = channel_decl('toMain', size=[N_GRP], **_main_type)
        if CIRCUIT_MAIN and not CIRCUIT_GATHER:
            _outA.operation.attributes['packet_ids'] = ArrayAttr.get([IntegerAttr.get(i32, p) for p in (31, 30, 29)])
            _outA.operation.attributes['keep_pkt_header'] = UnitAttr.get()
        _outY = Channel('outY', size=[1, 1], broadcast_shape=[1, NDEST])
        _outY.operation.attributes['channel_type'] = StringAttr.get('npu_dma_packet')
        if CIRCUIT_MAIN:
            # The first projection core writes this routing word. Circuit
            # gathers carry it as data; only the final hop interprets it.
            _outY.operation.attributes['packet_ids'] = ArrayAttr.get([IntegerAttr.get(i32, p) for p in (31, 30, 29)])
        _outY.operation.attributes['air.tile_dma_channel'] = IntegerAttr.get(T.i32(), 1)
        channel_decl('toShim', size=[NDEST])
        _layerOut = channel_decl('layerOut', size=[1])
        if NORMALIZATION_CACHE:
            _layerOut.operation.attributes['air.tile_dma_channel'] = IntegerAttr.get(i32, 1)
        if NORMALIZATION_REPLAY > 1:
            # Normalized chunks and the final residual are independent output
            # streams. The normalization count is selected at runtime; merging
            # these puts can lower just the inner five-chunk loops to finite
            # DMA tasks. Separate channels retain lock-driven reader rings.
            _xn.operation.attributes['air.tile_dma_channel'] = IntegerAttr.get(i32, 0)
            _layerOut.operation.attributes['air.tile_dma_channel'] = IntegerAttr.get(i32, 1)
        channel_decl('gluOut', size=[1])
        channel_decl('vocOut', size=[1])

        def idx(v):
            return arith.ConstantOp.create_index(v)

        def grp_leads(g):
            _cpg = NCX // N_GRP
            out = []
            for lc in range(_cpg):
                for pp in range(PAIRS_PC):
                    out.append((g * _cpg + lc, pp))
            return out
        _fn_args = [x_l3, w_l3, rms_l3, y_l3] + [kvc_l3] + _w_extra + ([i32] if DYNSEQ else [])
        _w_base_n = 4 + 1
        WARG = [1] + [_w_base_n + i for i in range(len(_w_extra))]

        @FuncOp.from_py_func(*_fn_args)
        def q4nx_decode(*_fa):
            X, W, RMS, Y = (_fa[0], _fa[1], _fa[2], _fa[3])

            def launch_body(*_la):
                X, W, RMS, Y = (_la[4], _la[5], _la[6], _la[7])
                _WBUFS = [W]
                _WEXTRA = None
                _wsel = [None]
                KVC = _la[8]
                L_rt = _la[4 + len(_fa) - 1] if DYNSEQ else None

                def _live_rounds(length):
                    # L is the anchor's one-based position. All queries share
                    # ceil((L + BATCH - 1)/16) transferred blocks; each still
                    # applies its own causal or bidirectional mask.
                    total = arith.addi(length, arith.ConstantOp(IntegerAttr.get(i32, BATCH + 14), None).result)
                    rounds = arith.divui(total, arith.ConstantOp(IntegerAttr.get(i32, 16), None).result)
                    return arith.index_cast(idx_t, rounds)
                a_iv = _la[-1] if len(_la) > 4 + len(_fa) else None

                def _lb(slab):
                    return 0 if a_iv is None else arith.muli(a_iv, idx(slab))

                def _lo(base, extra):
                    if a_iv is None:
                        return base + extra
                    return arith.addi(base, idx(extra)) if extra else base

                def _loi(base, extra):
                    if a_iv is None:
                        return idx(base + extra)
                    return arith.addi(base, idx(extra)) if extra else base

                def _loi_slot(base, extra):
                    if not DYNSEQ_APPEND:
                        return _loi(base, extra + (ATTN_L - 1) * REGION_W)
                    position = arith.subi(L_rt, arith.ConstantOp(IntegerAttr.get(i32, 1), None).result)
                    slot = arith.muli(position, arith.ConstantOp(IntegerAttr.get(i32, REGION_W), None).result)
                    offset = arith.addi(base if a_iv is not None else idx(base), arith.index_cast(idx_t, slot))
                    return arith.addi(offset, idx(extra)) if extra else offset
                _X_SLOT = BATCH * K

                def _x_in():
                    return _lb(_X_SLOT) if HIDDEN_TAPS else 0

                def _x_out():
                    return _lo(_lb(_X_SLOT), _X_SLOT) if HIDDEN_TAPS else 0
                _x_final = UNI_DEC * _X_SLOT if HIDDEN_TAPS else 0
                blk = BLOCK_BF16
                wstep = NCY * blk

                def _feed_wcol(_cx, _cbase, nsteps, Wb):
                    nch = 2
                    cstep = wstep // nch
                    span = nsteps * cstep
                    half = nsteps // 2 * cstep
                    for ci in range(nch):
                        base = _cbase if ci == 0 else arith.addi(_cbase, idx(span))
                        ch = _wname(ci, _cx)
                        ix = [idx(0)]
                        ChannelPut(ch, Wb, indices=ix, offsets=[base], sizes=[half], strides=[1])
                        ChannelPut(ch, Wb, indices=ix, offsets=[arith.addi(base, idx(half))], sizes=[span - half], strides=[1])

                def _feed_wcols(_wcol0, _colspan, nsteps):

                    def _fan(Wb):
                        for _cxi in range(NCX):
                            _cbase = _wcol0 + _cxi * _colspan if isinstance(_wcol0, int) else arith.addi(_wcol0, idx(_cxi * _colspan))
                            if isinstance(_cbase, int):
                                _cbase = idx(_cbase)
                            _feed_wcol(_cxi, _cbase, nsteps, Wb)
                        return
                    if _wsel[0] == 'extra':
                        _fan(_WEXTRA)
                    else:
                        _fan(W)
                for _layer in range(NLAYERS if a_iv is None else 1):
                    _wbase = _lb(W_LAYER)
                    _wgi = None
                    _rbase = _lb(RMS_LAYER)
                    _kbase = _lb(KV_LAYER)
                    _ybase = _lb(Y_LAYER)
                    _u1 = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
                    _u0 = arith.ConstantOp(IntegerAttr.get(i32, 0), None).result
                    if a_iv is None:
                        _uarm = _u1
                    else:
                        _ucmp = arith.cmpi(arith.CmpIPredicate.slt, a_iv, idx(UNI_DEC))
                        _uarm = arith.select(_ucmp, _u1, _u0)
                    _uarm_i = arith.index_cast(idx_t, _uarm)
                    _final_norm_off = UNI_DEC * RMS_LAYER + UNI_DEC * ROPE_W_LEN * BATCH

                    def _uni_voc():
                        _wsel[0] = None
                        _vwb = arith.addi(idx(UNI_DEC * W_LAYER), arith.muli(arith.subi(a_iv, idx(UNI_DEC)), idx(VOCAB_W_BLOCKS * BLOCK_BF16)))
                        _vyb = arith.addi(idx((HOST_ROUNDS + LAYER_RNDS) * PAYLOAD * BATCH), arith.muli(arith.subi(a_iv, idx(UNI_DEC)), idx(VOCAB_SIZE_PADDED * BATCH)))
                        ChannelPut('rmsX', X, offsets=[_x_final], sizes=[BATCH * K], strides=[1])
                        ChannelPut('rmsW', RMS, offsets=[_final_norm_off], sizes=[K], strides=[1])
                        ChannelPut('rmsW2', RMS, offsets=[0], sizes=[K], strides=[1])
                        assert VOCAB_PER_COL % NCY == 0
                        _colspan = VOCAB_PER_COL * blk
                        _feed_wcols(_vwb, _colspan, VOCAB_PER_COL // NCY)
                        ChannelGet('vocOut', Y, indices=[idx(0)], offsets=[_vyb], sizes=[VOCAB_RNDS // VOC_SLICE_RNDS, BATCH, GLU_SLICE], strides=[GLU_SLICE, VOCAB_SIZE_PADDED, 1])
                        yield_([])

                    def _uni_dec():
                        _wsel[0] = _wgi
                        ChannelPut('rmsX', X, offsets=[_x_in()], sizes=[BATCH * K], strides=[1])
                        ChannelPut('rmsW', RMS, offsets=[_rbase], sizes=[K], strides=[1])
                        ChannelPut('rmsW2', RMS, offsets=[_lo(_rbase, K)], sizes=[K], strides=[1])
                        _lut_off = UNI_DEC * RMS_LAYER
                        _rope_off = _lo(_lb(BATCH * ROPE_W_LEN), _lut_off)
                        _tl_range = range(BATCH)
                        for _tl in _tl_range:
                            _lo_t = _tl * ROPE_W_LEN
                            if not _lo_t:
                                _off_t = _rope_off
                            elif isinstance(_rope_off, int):
                                _off_t = _rope_off + _lo_t
                            else:
                                _off_t = arith.addi(_rope_off, idx(_lo_t))
                            ChannelPut('ropeLUT', RMS, offsets=[_off_t], sizes=[ROPE_W_LEN], strides=[1])

                        def _emit_append(_kbase=_kbase):
                            _bt = [idx(BATCH)]
                            _bs = [idx(REGION_W)]
                            _bo = [idx(0), idx(0)]
                            ChannelGet('appendK', KVC, indices=[idx(0)], offsets=_bo + [_loi_slot(_kbase, 0)], sizes=_bt + [idx(NGRP), idx(REGION_W)], strides=_bs + [idx(REGION_STRIDE), idx(1)])
                            ChannelGet('appendV', KVC, indices=[idx(0)], offsets=_bo + [_loi_slot(_kbase, _vreg_off(0))], sizes=_bt + [idx(NGRP), idx(REGION_W)], strides=_bs + [idx(REGION_STRIDE), idx(1)])
                            return

                        def _emit_readback(_kbase=_kbase):
                            _emit_readback_one(_kbase)

                        def _emit_readback_one(_kbase=_kbase):
                            _NRB = int('1')
                            _nb = RB_ROUNDS
                            _cbk = (_nb + _NRB - 1) // _NRB
                            _KV1D = int('0')
                            _ci = 0
                            while _ci < _nb:
                                _cb = min(_cbk, _nb - _ci)
                                _coff = _ci * 16 * REGION_W
                                _cbv = lambda: _live_rounds(L_rt) if DYNSEQ_RB else idx(_cb)
                                # Current AIR lowers a runtime-length linear
                                # transfer or a runtime repeat, but not both
                                # dimensions together. Emit the eight reads
                                # explicitly and retain one contiguous live span.
                                _rb = [idx(1 if DYNSEQ_RB else KV_QUERY_GROUPS)]
                                _r0 = [idx(0)]
                                for _query in range(KV_QUERY_GROUPS if DYNSEQ_RB else 1):
                                    for gi in range(NGRP):
                                        ChannelPut('inKV_K', KVC, indices=[idx(gi)], offsets=_r0 + [_loi(_kbase, _kreg_off(gi) + _coff)], sizes=_rb + [idx(_cb * 16 * REGION_W)] if _KV1D else _rb + [_cbv(), idx(16), idx(REGION_W)], strides=_r0 + [idx(1)] if _KV1D else _r0 + [idx(16 * REGION_W), idx(REGION_W), idx(1)])
                                        ChannelPut('inKV_V', KVC, indices=[idx(gi)], offsets=_r0 + [_loi(_kbase, _vreg_off(gi) + _coff)], sizes=_rb + [idx(_cb * 16 * REGION_W)] if _KV1D else _rb + [_cbv(), idx(16), idx(REGION_W)], strides=_r0 + [idx(1)] if _KV1D else _r0 + [idx(16 * REGION_W), idx(REGION_W), idx(1)])
                                _ci += _cb
                            return
                        woff = 0
                        for p in range(NPH):
                            per_col = PER_COL_PH[p]
                            assert per_col % NCY == 0
                            _colspan = per_col * blk
                            _wcol0 = _lo(_wbase, woff)
                            _feed_wcols(_wcol0, _colspan, per_col // NCY)
                            woff += NCX * per_col * blk
                            if MULTIBLK and p == 0:
                                _emit_append()
                                _emit_readback()
                        roff = 0
                        for p in HOST_DRAIN:
                            if p == 0:
                                pass
                            else:
                                for rr in range(ROUNDS_PER_DEST[p]):
                                    ChannelGet('toShim', Y, indices=[idx(p)], offsets=[_lo(_ybase, (roff + rr) * PAYLOAD)], sizes=[PAYLOAD], strides=[1])
                            roff += ROUNDS_PER_DEST[p]
                        _out_bo = X
                        _out_base = _x_out()
                        ChannelGet('layerOut', _out_bo, indices=[idx(0)], offsets=[_out_base], sizes=[BATCH * LAYER_RNDS * PAYLOAD], strides=[1])
                        yield_([])
                    index_switch([], _uarm_i, [0], case_body_builder=lambda op, i, cv: _uni_voc(), default_body_builder=lambda op: _uni_dec())
                _seg_opers = ([a_iv] if a_iv is not None else []) + ([L_rt] if DYNSEQ else [])

                @segment(name='seg', operands=_seg_opers)
                def seg(*_sa):
                    _seg_iv = _sa[0] if a_iv is not None else None
                    _seg_L = _sa[-1] if DYNSEQ else None

                    def _seg_blocks():
                        r = _seg_rounds()
                        return arith.muli(r, idx(BATCH // ATTENTION_QUERY_TILE))

                    def _seg_rounds():
                        return _live_rounds(_seg_L) if DYNSEQ_MEM else idx(ATTN_ROUNDS)
                    if _seg_iv is not None:
                        _seg_cmp = arith.cmpi(arith.CmpIPredicate.slt, _seg_iv, idx(UNI_DEC))
                        _seg_arm = arith.select(_seg_cmp, arith.ConstantOp(IntegerAttr.get(i32, 1), None).result, arith.ConstantOp(IntegerAttr.get(i32, 0), None).result)
                        _seg_arm_i = arith.index_cast(idx_t, _seg_arm)
                    else:
                        _seg_cmp = None
                        _seg_arm_i = None
                        _seg_arm = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result

                    def _feed_inX(src, total_chunks):
                        for _rc in for_(idx(0), idx(total_chunks), idx(1)):
                            xb = AllocOp(xmt_l2, [], [])
                            xb.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), XMT_PCOL)
                            ChannelGet(src, xb, offsets=[0], sizes=[BATCH * XCHUNK], strides=[1])
                            for _jj in for_(idx(0), idx(XCHUNK_MUL), idx(1)):
                                _sz, _st = _XFEED_BD
                                _o0 = arith.muli(_jj, idx(COL_BLOCK // _st[0]))
                                ChannelPut('inX', xb, offsets=[_o0] + [idx(0)] * (len(_sz) - 1), sizes=[idx(v) for v in _sz], strides=[idx(v) for v in _st])
                                yield_([])
                            DeallocOp(xb)
                            yield_([])
                    # Projection inputs use K, DQ, K and GLU_OUT respectively.
                    # Derive the total from their actual packed block counts.
                    assert N_XBLK % XCHUNK_MUL == 0
                    _xc_dec = N_XBLK // XCHUNK_MUL
                    _xc_voc = VOCAB_RNDS * (K // XCHUNK)
                    if NORMALIZATION_REPLAY > 1:
                        # The core emits [chunk, token, within-chunk]. Preserve
                        # that stream order in the full normalized L2 slab.
                        _fill_dec = (REFEED[0] + GATEUP_REFEED) // NORMALIZATION_REPLAY
                        _fill_voc = VOCAB_RNDS // NORMALIZATION_REPLAY
                        def _norm_relay(nfills):
                            # Keep the slab resident across fills, but allocate
                            # it inside each arm so dependency tokens dominate
                            # their uses. Inspect whether lowering merges the
                            # mutually exclusive arm allocations before running.
                            slab = AllocOp(xmt_wide_l2, [], [])
                            slab.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), 2)
                            for _fill in for_(idx(0), idx(nfills), idx(1)):
                                ChannelGet('xnorm', slab, offsets=[idx(0)], sizes=[idx(BATCH * K)], strides=[idx(1)])
                                # Let air-annotate-refeed derive the producer and
                                # consumer lock counts from the existing helper.
                                refeed(NORMALIZATION_REPLAY, lambda: ChannelPut(
                                    'xin', slab, offsets=[idx(0)], sizes=[idx(BATCH * K)], strides=[idx(1)]))
                                yield_([])
                            DeallocOp(slab)
                        if _seg_arm_i is None:
                            _norm_relay(_fill_dec)
                        else:
                            def _norm_voc():
                                _norm_relay(_fill_voc)
                                yield_([])
                            def _norm_dec():
                                _norm_relay(_fill_dec)
                                yield_([])
                            index_switch([], _seg_arm_i, [0],
                                case_body_builder=lambda op, i, cv: _norm_voc(),
                                default_body_builder=lambda op: _norm_dec())
                    if _seg_arm_i is not None:
                        _NCH = K // XCHUNK

                        def _xs_voc():
                            _feed_inX('xin' if NORMALIZATION_REPLAY > 1 else 'xnorm', _xc_voc)
                            yield_([])

                        def _xs_dec():
                            _feed_inX('xin' if NORMALIZATION_REPLAY > 1 else 'xnorm', _xc_dec)
                            yield_([])
                        index_switch([], _seg_arm_i, [0], case_body_builder=lambda op, i, cv: _xs_voc(), default_body_builder=lambda op: _xs_dec())
                    else:
                        _feed_inX('xin' if NORMALIZATION_REPLAY > 1 else 'xnorm', _xc_dec)
                    _fan_groups = [(0, 0, NCY // 2), (1, NCY // 2, NCY)]
                    for cx in range(NCX):
                        for _ci, _cy0, _cy1 in _fan_groups:
                            _wch = _wname(_ci, cx)
                            for _ in for_(idx(0), idx(W_FAN_STEPS), idx(1)):
                                wf = AllocOp(wfan_l2, [], [])
                                wf.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), PCOL[cx])
                                ChannelGet(_wch, wf, indices=[idx(0)])
                                for cy in range(_cy0, _cy1):
                                    ChannelPut('wL2ToL1', wf, indices=[idx(cx), idx(cy)], offsets=[(cy - _cy0) * BLOCK_BF16], sizes=[BLOCK_BF16], strides=[1])
                                DeallocOp(wf)
                                yield_([])

                    def _egress(_nrc):
                        for _r in for_(idx(0), idx(_nrc * (BATCH if TOKEN_PACKETS else 1)), idx(1)):
                            for g in range(N_GRP):
                                grp = AllocOp(grp_l2, [], [])
                                grp.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), GRP_PCOL[g])
                                for k, (cx, pp) in enumerate(grp_leads(g)):
                                    _pay = PAIR_PAY * PACKET_BATCH
                                    off = 0 if k == 0 else HDR + k * _pay
                                    sz = HDR + _pay if k == 0 else _pay
                                    ChannelGet('outA', grp, indices=[idx(cx), idx(pp)], offsets=[off], sizes=[sz], strides=[1])
                                _skip = HDR if CIRCUIT_MAIN and g else 0
                                ChannelPut('toMain', grp, indices=[idx(g)], offsets=[_skip], sizes=[GRP_ROWS_B - _skip], strides=[1])
                                DeallocOp(grp)
                            ml = AllocOp(main_l2, [], [])
                            ml.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), MAIN_PCOL)
                            for g in range(N_GRP):
                                _gpay = LEADS_PER_GRP * PAIR_PAY * PACKET_BATCH
                                off = 0 if g == 0 else GRP_ROWS_B + (g - 1) * _gpay
                                sz = GRP_ROWS_B if g == 0 else _gpay
                                ChannelGet('toMain', ml, indices=[idx(g)], offsets=[off], sizes=[sz], strides=[1])
                            ChannelPut('outY', ml, indices=[idx(0), idx(0)], offsets=[0], sizes=[MAIN_ROWS_B], strides=[1])
                            DeallocOp(ml)
                            yield_([])
                    if _seg_arm_i is not None:

                        def _egr_voc():
                            _egress(VOCAB_RNDS)
                            yield_([])

                        def _egr_dec():
                            _egress(N_ROUNDS)
                            yield_([])
                        index_switch([], _seg_arm_i, [0], case_body_builder=lambda op, i, cv: _egr_voc(), default_body_builder=lambda op: _egr_dec())
                    else:
                        _egress(N_ROUNDS)

                    def _qkv_transpose():
                        qmt = AllocOp(qkvmt_l2, [], [])
                        qmt.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), RELAY_COLS[0])
                        _qo, _qs, _qt = outy_tokmajor(M, rounds=ROUNDS_PER_DEST[0])
                        ChannelGet('outY', qmt, indices=[idx(0), idx(0)], offsets=[idx(v) for v in _qo], sizes=[idx(v) for v in _qs], strides=[idx(v) for v in _qt])
                        _pt_all = [(0, BATCH * M)] if not None else [(_t * M, M) for _t in range(BATCH)]
                        for _off, _len in _pt_all:
                            _pt = ChannelPut('toRope', qmt, offsets=[idx(_off)], sizes=[idx(_len)], strides=[idx(1)])
                            if RELAY_COLS[0] in _ATTN_COLS:
                                _pt.operation.attributes['air.memtile_dma_channel_min'] = IntegerAttr.get(T.i32(), 1)
                        DeallocOp(qmt)
                    if _seg_arm_i is not None:

                        def _qt_dec():
                            _qkv_transpose()
                            yield_([])
                        index_switch([], _seg_arm_i, ARM_IDLE_CASES, case_body_builder=lambda op, i, cv: yield_([]), default_body_builder=lambda op: _qt_dec())
                    else:
                        _qkv_transpose()
                    for p in HOST_DRAIN:
                        if p == 0:
                            continue
                        for _rp in for_(idx(0), idx(ROUNDS_PER_DEST[p]), idx(1)):
                            rb = AllocOp(relay_l2, [], [])
                            rb.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), RELAY_COLS[p])
                            ChannelGet('outY', rb, indices=[idx(0), idx(p)], offsets=[0], sizes=[PAYLOAD], strides=[1])
                            ChannelPut('toShim', rb, indices=[idx(p)], offsets=[0], sizes=[PAYLOAD], strides=[1])
                            DeallocOp(rb)
                            yield_([])
                    _arm_rope = _seg_arm

                    @herd(name='rope', sizes=[1, 1], operands=[_arm_rope])
                    def rope_h(tx, ty, _sx, _sy, _arm):

                        def _dec_one():
                            a_qkv = AllocOp(qkv_l1, [], [])
                            ChannelGet('toRope', a_qkv, offsets=[idx(0)], sizes=[idx(M)], strides=[idx(1)])
                            a_lut = AllocOp(ropelut_l1, [], [])
                            ChannelGet('ropeLUT', a_lut, indices=[idx(0)])
                            a_q = AllocOp(ropeq_l1, [], [])
                            a_k = AllocOp(ropekv_l1, [], [])
                            a_v = AllocOp(ropekv_l1, [], [])
                            CallOp(rope_compute, [a_q, a_k, a_v, a_qkv, a_lut, _arm])
                            ChannelPut('ropeQ', a_q, indices=[idx(0)], offsets=[idx(0)], sizes=[idx(DQ_PADDED)], strides=[idx(1)])
                            ChannelPut('appendK', a_k, indices=[idx(0)], offsets=[idx(0)], sizes=[idx(DK_TOT_A)], strides=[idx(1)])
                            ChannelPut('appendV', a_v, indices=[idx(0)], offsets=[idx(0)], sizes=[idx(DK_TOT_A)], strides=[idx(1)])
                            DeallocOp(a_qkv)
                            DeallocOp(a_lut)
                            DeallocOp(a_q)
                            DeallocOp(a_k)
                            DeallocOp(a_v)

                        def _dec():
                            for _t in for_(idx(0), idx(BATCH), idx(1)):
                                _dec_one()
                                yield_([])
                            yield_([])

                        def _voc():
                            yield_([])
                        index_switch([], arith.index_cast(idx_t, _arm), ARM_IDLE_CASES, case_body_builder=lambda op, i, cv: _voc(), default_body_builder=lambda op: _dec())
                    rope_h.attributes['link_with'] = StringAttr.get('rope.o')
                    rope_h.attributes['x_loc'] = IntegerAttr.get(T.i64(), RMS_PCOL)
                    rope_h.attributes['y_loc'] = IntegerAttr.get(T.i64(), 3)

                    def _qmtb_dec():
                        qmtb = AllocOp(qmt_l2, [], [])
                        qmtb.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), 5)
                        ChannelGet('ropeQ', qmtb, indices=[idx(0)], offsets=[idx(0)], sizes=[idx(BATCH * DQ_PADDED)], strides=[idx(1)])
                        _qmtb_fan(qmtb)
                        DeallocOp(qmtb)

                    def _qmtb_fan(qmtb):
                        if PARALLEL_ATTENTION:
                            parallel_attention.query(qmtb)
                            return
                        _bt = [idx(BATCH)]
                        _bs = [idx(DQ_PADDED)]
                        _bo = [idx(0)]
                        for c in range(N_ATTN_CU):
                            ChannelPut('toAttnQ', qmtb, indices=[idx(c)], offsets=_bo + [idx(0), idx(c * Q_HEADS_PADDED_PER_CU), idx(0)], sizes=_bt + [idx(DH // 8), idx(Q_HEADS_PADDED_PER_CU), idx(8)], strides=_bs + [idx(8), idx(DH), idx(1)])
                    if _seg_arm_i is not None:

                        def _q_voc():
                            yield_([])

                        def _q_dec():
                            _qmtb_dec()
                            yield_([])
                        index_switch([], _seg_arm_i, ARM_IDLE_CASES, case_body_builder=lambda op, i, cv: _q_voc(), default_body_builder=lambda op: _q_dec())
                    else:
                        _qmtb_dec()

                    if PARALLEL_ATTENTION:
                        _Lc = _seg_L if DYNSEQ_RTP else arith.ConstantOp(IntegerAttr.get(i32, ATTN_L), None).result
                        parallel_attention.emit(_Lc, _seg_arm)
                    else:
                        def _make_cu(c):
                            col, qk_row, kv_row = ATTN_CU_LOC[c]
                            a_sh = AllocOp(as_l1, [], [])
                            L_c = _seg_L if DYNSEQ_RTP else arith.ConstantOp(IntegerAttr.get(i32, ATTN_L), None).result

                            def _reblock_dec():
                                _reblock_one()

                            def _reblock_one():
                                _gi = ATTN_CU_GROUP[c]
                                _gcol, _cus = ATTN_COL_GROUPS[_gi]
                                if c != _cus[0]:
                                    return
                                _gw = len(_cus) * KVPC_DH
                                for _blk in for_(idx(0), _seg_blocks(), idx(1)):
                                    _kbuf = AllocOp(kvblk_l2, [], [])
                                    _kbuf.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), col)
                                    _vbuf = AllocOp(kvblk_l2, [], [])
                                    _vbuf.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), col)
                                    ChannelGet('inKV_K', _kbuf, indices=[idx(_gi)])
                                    ChannelGet('inKV_V', _vbuf, indices=[idx(_gi)])
                                    for _lc, _cc in enumerate(_cus):
                                        _pk = ChannelPut('toK', _kbuf, indices=[idx(_cc)], offsets=[idx(0), idx(0), idx(_lc * KVPC_DH)], sizes=[idx(KVPC_DH // 8), idx(16), idx(8)], strides=[idx(8), idx(_gw), idx(1)])
                                        _pv = ChannelPut('toV', _vbuf, indices=[idx(_cc)], offsets=[idx(0), idx(0), idx(0), idx(_lc * KVPC_DH)], sizes=[idx(2), idx(KVPC_DH // 8), idx(8), idx(8)], strides=[idx(_gw * 8), idx(8), idx(_gw), idx(1)])
                                        if col == 3:
                                            _pk.operation.attributes['air.memtile_dma_channel_min'] = IntegerAttr.get(T.i32(), 1)
                                            _pv.operation.attributes['air.memtile_dma_channel_min'] = IntegerAttr.get(T.i32(), 1)
                                    DeallocOp(_kbuf)
                                    DeallocOp(_vbuf)
                                    yield_([])
                                return
                            _gated = _seg_arm_i is not None
                            if _gated:

                                def _rb_voc():
                                    yield_([])

                                def _rb_dec():
                                    _reblock_dec()
                                    yield_([])
                                index_switch([], _seg_arm_i, ARM_IDLE_CASES, case_body_builder=lambda op, i, cv: _rb_voc(), default_body_builder=lambda op: _rb_dec())
                            else:
                                _reblock_dec()

                            def _tok_L(Lh, t_iv):
                                delta = arith.index_cast(i32, t_iv) if CAUSAL else arith.ConstantOp(IntegerAttr.get(i32, BATCH - 1), None).result
                                return arith.addi(Lh, delta)

                            def _qk_body(sh, Lh, _c):
                                a_q = AllocOp(aq_l1, [], [])
                                a_m = AllocOp(m_l1, [], [])
                                a_cc = AllocOp(c_l1, [], [])
                                ChannelGet('toAttnQ', a_q, indices=[idx(_c)])
                                _rounds = _live_rounds(Lh) if DYNSEQ_TRIP else idx(ATTN_ROUNDS)
                                for _flat in for_(idx(0), arith.muli(idx(BATCH // ATTENTION_QUERY_TILE), _rounds), idx(1)):
                                    _t = arith.muli(arith.divui(_flat, _rounds), idx(ATTENTION_QUERY_TILE))
                                    _blk = arith.remui(_flat, _rounds)
                                    a_k = AllocOp(ak_l1, [], [])
                                    ChannelGet('toK', a_k, indices=[idx(_c)])
                                    _qk_args = [a_q, a_k, a_m, a_cc, sh, arith.index_cast(i32, _blk), _tok_L(Lh, _t), arith.index_cast(i32, _t)]
                                    if ATTENTION_QUERY_TILE > 1:
                                        _qk_args.append(arith.ConstantOp(IntegerAttr.get(i32, int(CAUSAL)), None).result)
                                    CallOp(attn_qk_blk_row, _qk_args)
                                    DeallocOp(a_k)
                                    yield_([])
                                DeallocOp(a_cc)
                                DeallocOp(a_m)
                                DeallocOp(a_q)

                            def _kv_body(sh, Lh, _c):
                                a_o = AllocOp(ao_l1, [], [])
                                a_y = AllocOp(y_l1, [], [])
                                a_l = AllocOp(lden_l1, [], [])
                                _rounds = _live_rounds(Lh) if DYNSEQ_TRIP else idx(ATTN_ROUNDS)
                                for _flat in for_(idx(0), arith.muli(idx(BATCH // ATTENTION_QUERY_TILE), _rounds), idx(1)):
                                    _t = arith.muli(arith.divui(_flat, _rounds), idx(ATTENTION_QUERY_TILE))
                                    _blk = arith.remui(_flat, _rounds)
                                    a_v = AllocOp(av_l1, [], [])
                                    ChannelGet('toV', a_v, indices=[idx(_c)])
                                    _kv_args = [sh, a_v, a_y, a_l, arith.index_cast(i32, _blk), _tok_L(Lh, _t)]
                                    if ATTENTION_QUERY_TILE > 1:
                                        _kv_args.append(arith.ConstantOp(IntegerAttr.get(i32, int(CAUSAL)), None).result)
                                    CallOp(attn_kv_blk, _kv_args)
                                    DeallocOp(a_v)
                                    _last = arith.cmpi(arith.CmpIPredicate.eq, _blk, arith.subi(_rounds, idx(1)))
                                    _done = IfOp(_last, [], has_else=False)
                                    with InsertionPoint(_done.thenRegion.blocks[0]):
                                        CallOp(attn_kv_fin_row, [a_y, a_l, a_o, arith.index_cast(i32, _t)])
                                        yield_([])
                                    yield_([])
                                _attn_o_put_block(a_o, _c)
                                DeallocOp(a_l)
                                DeallocOp(a_y)
                                DeallocOp(a_o)

                            def _attn_o_put_block(a_o, _c):
                                ChannelPut('attnO', a_o, indices=[idx(_c)], offsets=[idx(0)], sizes=[idx(BATCH * DQ_PADDED_PER_CU)], strides=[idx(1)])
                            return (a_sh, col, qk_row, L_c, _qk_body, _kv_body)
                        _cus = [_make_cu(c) for c in range(N_ATTN_CU)]
                        _sh = [t[0] for t in _cus]
                        _Lc = _cus[0][3]
                        _qkb = _cus[0][4]
                        _kvb = _cus[0][5]

                        def _attn_leaf(ty_arg, cu, sh, Lh, qk_ty):
                            _isqk = arith.cmpi(arith.CmpIPredicate.eq, ty_arg, idx(qk_ty))
                            _if = IfOp(_isqk, [], has_else=True)
                            with InsertionPoint(_if.thenRegion.blocks[0]):
                                _qkb(sh, Lh, cu)
                                yield_([])
                            with InsertionPoint(_if.elseRegion.blocks[0]):
                                _kvb(sh, Lh, cu)
                                yield_([])

                        def _attn_pairsel(ty_arg, shs, Lh, cu_lo, cu_hi):
                            _lo = arith.cmpi(arith.CmpIPredicate.slt, ty_arg, idx(2))
                            _ifp = IfOp(_lo, [], has_else=True)
                            with InsertionPoint(_ifp.thenRegion.blocks[0]):
                                _attn_leaf(ty_arg, cu_lo, shs[cu_lo], Lh, 0)
                                yield_([])
                            with InsertionPoint(_ifp.elseRegion.blocks[0]):
                                _attn_leaf(ty_arg, cu_hi, shs[cu_hi], Lh, 2)
                                yield_([])

                        def _attn_dec(tx_arg, ty_arg, shs, Lh):
                            _isc0 = arith.cmpi(arith.CmpIPredicate.eq, tx_arg, idx(0))
                            _ifc = IfOp(_isc0, [], has_else=True)
                            with InsertionPoint(_ifc.thenRegion.blocks[0]):
                                _attn_pairsel(ty_arg, shs, Lh, 0, 1)
                                yield_([])
                            with InsertionPoint(_ifc.elseRegion.blocks[0]):
                                _attn_pairsel(ty_arg, shs, Lh, 2, 3)
                                yield_([])
                        if _seg_arm_i is not None:

                            @herd(name='attn_blk', sizes=[2, 4], operands=[_sh[0].result, _sh[1].result, _sh[2].result, _sh[3].result, _Lc, _seg_arm])
                            def attn_blk(_tx, _ty, _sx, _sy, s0, s1, s2, s3, Lh, _arm):
                                shs = [s0, s1, s2, s3]

                                def _voc():
                                    yield_([])

                                def _dec():
                                    _attn_dec(_tx, _ty, shs, Lh)
                                    yield_([])
                                index_switch([], arith.index_cast(idx_t, _arm), ARM_IDLE_CASES, case_body_builder=lambda op, i, cv: _voc(), default_body_builder=lambda op: _dec())
                        else:

                            @herd(name='attn_blk', sizes=[2, 4], operands=[_sh[0].result, _sh[1].result, _sh[2].result, _sh[3].result, _Lc])
                            def attn_blk(_tx, _ty, _sx, _sy, s0, s1, s2, s3, Lh):
                                _attn_dec(_tx, _ty, [s0, s1, s2, s3], Lh)
                        attn_blk.attributes['x_loc'] = IntegerAttr.get(T.i64(), ATTN_CU_LOC[0][0])
                        attn_blk.attributes['y_loc'] = IntegerAttr.get(T.i64(), 2)

                    def _omtb_dec():
                        omtb = AllocOp(omt_l2, [], [])
                        omtb.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), 5)

                        def _o_gather(_toff):
                            for c in range(N_ATTN_CU):
                                ChannelGet('attnO', omtb, indices=[idx(c)], offsets=[idx(c * DQ_PER_CU) if _toff is None else arith.addi(_toff, idx(c * DQ_PER_CU))], sizes=[idx(DQ_PER_CU)], strides=[idx(1)])

                        def _o_gather_block():
                            if PARALLEL_ATTENTION:
                                parallel_attention.gather(omtb)
                                return
                            assert DH % 8 == 0
                            for c in range(N_ATTN_CU):
                                ChannelGet('attnO', omtb, indices=[idx(c)], offsets=[idx(0), idx(0), idx(0), idx(c * DQ_PER_CU)], sizes=[idx(BATCH), idx(DH // 8), idx(Q_HEADS_PER_CU), idx(8)], strides=[idx(DQ), idx(8), idx(DH), idx(1)])
                        if None:
                            for _t in for_(idx(0), idx(BATCH), idx(1)):
                                _o_gather(arith.muli(_t, idx(DQ)))
                                yield_([])
                        else:
                            _o_gather_block()
                        refeed(OPROJ_REFEED, lambda: _xnorm_put(omtb, N_ATTN_CU * DQ_PER_CU, ssa=True, indices=[idx(0)]))
                        DeallocOp(omtb)
                    if _seg_arm_i is not None:

                        def _o_voc():
                            yield_([])

                        def _o_dec():
                            _omtb_dec()
                            yield_([])
                        index_switch([], _seg_arm_i, ARM_IDLE_CASES, case_body_builder=lambda op, i, cv: _o_voc(), default_body_builder=lambda op: _o_dec())
                    else:
                        _omtb_dec()
                    _arm_glu = _seg_arm

                    @herd(name='glu', sizes=[1, 1], operands=[_arm_glu])
                    def glu_h(tx, ty, _sx, _sy, _arm):
                        _voc_arm = BATCH > 1 and GLU_DEST >= 0

                        def _get_slice(gx):
                            assert GLU_SLICE % PAYLOAD == 0
                            _go, _gs, _gt = outy_tokmajor(GLU_SLICE, rounds=GLU_SLICE // PAYLOAD)
                            ChannelGet('outY', gx, indices=[idx(0), idx(GLU_DEST)], offsets=[idx(v) for v in _go], sizes=[idx(v) for v in _gs], strides=[idx(v) for v in _gt])

                        def _glu_slice(gx, mk_gh):
                            _get_slice(gx)
                            gh = mk_gh()
                            for _t in range(BATCH):
                                CallOp(glu_row_aie, [gh, gx, arith.ConstantOp(IntegerAttr.get(i32, _t), None).result, _arm])
                            ChannelPut('gluOut', gh, offsets=[idx(0)], sizes=[idx(BATCH * GLU_HID)], strides=[idx(1)])
                            return gh

                        def _voc_slice(gx, mk_gh):
                            _get_slice(gx)
                            gh = mk_gh()
                            _half = BATCH * GLU_HID
                            for _h in range(2):
                                CallOp(glu_copy_aie, [gh, gx, arith.ConstantOp(IntegerAttr.get(i32, _h * _half), None).result, arith.ConstantOp(IntegerAttr.get(i32, _half), None).result, _arm])
                                ChannelPut('vocOut', gh, offsets=[idx(0)], sizes=[idx(_half)], strides=[idx(1)])
                            return gh
                        assert NGLU % 2 == 0, f'odd NGLU={NGLU} desyncs the GLU BD ring (GLU_PKTS={GLU_PKTS})'

                        def _dec():

                            def _slice():
                                gx = AllocOp(glu_x_l1, [], [])
                                gh = _glu_slice(gx, lambda: AllocOp(glu_hid_l1, [], []))
                                DeallocOp(gx)
                                DeallocOp(gh)
                            for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                _slice()
                                _slice()
                                yield_([])
                        if not _voc_arm:

                            def _dec_case():
                                _dec()
                                yield_([])
                            index_switch([], arith.index_cast(idx_t, _arm), ARM_IDLE_CASES, case_body_builder=lambda op, i, cv: yield_([]), default_body_builder=lambda op: _dec_case())
                        else:
                            _gx = [AllocOp(glu_x_l1, [], []) for _ in range(2)]
                            _gh = [AllocOp(glu_hid_l1, [], []) for _ in range(2)]

                            def _glu_dec_body():
                                for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                    _glu_slice(_gx[0], lambda: _gh[0])
                                    _glu_slice(_gx[1], lambda: _gh[1])
                                    yield_([])

                            def _glu_voc_body():
                                for _s in for_(idx(0), idx(VOC_ROTATIONS), idx(1)):
                                    _voc_slice(_gx[0], lambda: _gh[0])
                                    _voc_slice(_gx[1], lambda: _gh[1])
                                    yield_([])
                            _outer = None
                            _ctx = InsertionPoint(_outer.thenRegion.blocks[0]) if _outer is not None else contextlib.nullcontext()
                            with _ctx:
                                _ifa = IfOp(arith.cmpi(arith.CmpIPredicate.ne, _arm, arith.ConstantOp(IntegerAttr.get(i32, 0), None).result), [], has_else=True)
                                _ifa.operation.attributes['air.arm_select'] = UnitAttr.get()
                                with InsertionPoint(_ifa.thenRegion.blocks[0]):
                                    _glu_dec_body()
                                    yield_([])
                                with InsertionPoint(_ifa.elseRegion.blocks[0]):
                                    _glu_voc_body()
                                    yield_([])
                                if _outer is not None:
                                    yield_([])
                            for _b in _gx + _gh:
                                DeallocOp(_b)
                    glu_h.attributes['link_with'] = StringAttr.get('glu.o')
                    glu_h.attributes['x_loc'] = IntegerAttr.get(T.i64(), GLU_PCOL)
                    glu_h.attributes['y_loc'] = IntegerAttr.get(T.i64(), 3)
                    db = AllocOp(down_l2, [], [])
                    db.operation.attributes['air.memtile_col'] = IntegerAttr.get(T.i32(), DOWN_PCOL)
                    for _s in for_(idx(0), idx(NGLU), idx(1)):
                        soff = arith.muli(_s, idx(GLU_HID))
                        ChannelGet('gluOut', db, offsets=[idx(0), soff], sizes=[idx(BATCH), idx(GLU_HID)], strides=[idx(GLU_OUT), idx(1)])
                        yield_([])
                    refeed(DOWN_REFEED, lambda: _xnorm_put(db, GLU_OUT))
                    DeallocOp(db)

                    def _psw(ph, vals, ty):
                        if len(vals) == 1:
                            return vals[0]
                        return index_switch([ty], ph, list(range(len(vals) - 1)), case_body_builder=lambda op, i, cv: yield_([vals[i]]), default_body_builder=lambda op: yield_([vals[-1]]))

                    def _wscr_alloc():
                        scratch = AllocOp(wscr_mm_l1, [], [])
                        if PROJECTION_WEIGHT_PINGPONG:
                            # An opaque compute call may read its scratch on entry.
                            # Sinking it into the K loop prevents safe ping-pong.
                            scratch.operation.attributes['air.disable_alloc_sink'] = UnitAttr.get()
                        return scratch

                    def _core_blk_np(base_cx):

                        def body(tx, ty, _sx, _sy, _arm):
                            gcx = arith.addi(idx(base_cx), tx)
                            gcy = ty
                            i2c = [idx(v) for v in I2P]
                            j2c = [idx(v) for v in J2P]
                            pktc = [idx(d) for d in DEST]
                            c2 = idx(2)
                            c0i = arith.ConstantOp(IntegerAttr.get(i32, 0), None).result
                            _ws = _wscr_alloc()

                            def _wscr():
                                return _ws

                            def _mm(J2v):
                                J2x2 = arith.muli(J2v, c2)
                                a_acc = AllocOp(yacc_mm_l1, [], [])
                                a_ws = _wscr()
                                CallOp(mm_zero, [a_acc, _arm])
                                a_x_resident = None
                                if PROJECTION_WEIGHT_PINGPONG:
                                    # Only weights may alternate; preserve activation broadcast.
                                    a_x_resident = AllocOp(xblk_mm_l1, [], [])
                                    a_x_resident.operation.attributes['air.disable_alloc_sink'] = UnitAttr.get()
                                for _j in for_(idx(0), J2x2, idx(1)):
                                    a_x = a_x_resident if a_x_resident is not None else AllocOp(xblk_mm_l1, [], [])
                                    ChannelGet('inX', a_x, indices=[gcx, gcy])
                                    a_w = AllocOp(wblk_l1, [], [])
                                    ChannelGet('wL2ToL1', a_w, indices=[gcx, gcy])
                                    CallOp(mm_acc, [a_x, a_w, a_acc, a_ws])
                                    if a_x_resident is None:
                                        DeallocOp(a_x)
                                    DeallocOp(a_w)
                                    yield_([])
                                if a_x_resident is not None:
                                    DeallocOp(a_x_resident)
                                return a_acc
                            _proj = _mm
                            _tokstr = arith.ConstantOp(IntegerAttr.get(i32, PAIR_ROWS), None).result

                            def _emit(a_acc, destv):
                                if TOKEN_PACKETS:
                                    for token in for_(idx(0), idx(BATCH), idx(1)):
                                        yb = AllocOp(ypair_mm_l1, [], [])
                                        CallOp(mm_flush, [a_acc, yb, arith.index_cast(i32, token), _tokstr])
                                        ChannelPut('outA', yb, indices=[gcx, ty], offsets=[idx(14)], sizes=[idx(HDR + PAIR_PAY)], strides=[idx(1)], dest=destv)
                                        DeallocOp(yb)
                                        yield_([])
                                    DeallocOp(a_acc)
                                    return
                                yb = AllocOp(ypair_mm_l1, [], [])
                                if CIRCUIT_GATHER:
                                    CallOp(mm_flush, [a_acc, yb, arith.index_cast(i32, destv), _tokstr])
                                    _first = IfOp(arith.cmpi(arith.CmpIPredicate.eq, ty, idx(0)), [], has_else=True)
                                    with InsertionPoint(_first.thenRegion.blocks[0]):
                                        ChannelPut('outA', yb, indices=[gcx, ty], offsets=[idx(14)], sizes=[idx(HDR + PAIR_PAY * BATCH)], strides=[idx(1)])
                                        yield_([])
                                    with InsertionPoint(_first.elseRegion.blocks[0]):
                                        ChannelPut('outA', yb, indices=[gcx, ty], offsets=[idx(16)], sizes=[idx(PAIR_PAY * BATCH)], strides=[idx(1)])
                                        yield_([])
                                elif CIRCUIT_MAIN:
                                    CallOp(mm_flush, [a_acc, yb, arith.index_cast(i32, destv), _tokstr])
                                    ChannelPut('outA', yb, indices=[gcx, ty], offsets=[idx(14)], sizes=[idx(HDR + PAIR_PAY * BATCH)], strides=[idx(1)])
                                else:
                                    CallOp(mm_flush, [a_acc, yb, c0i, _tokstr])
                                    ChannelPut('outA', yb, indices=[gcx, ty], offsets=[idx(14)], sizes=[idx(HDR + PAIR_PAY * BATCH)], strides=[idx(1)], dest=destv)
                                DeallocOp(yb)
                                DeallocOp(a_acc)
                            _arm_i = arith.index_cast(idx_t, _arm)
                            _id4 = idx(VOCAB_DEST)

                            def _sel(voc_val, dec_thunk, ty_, ex=None):
                                return index_switch([ty_], _arm_i, [0], case_body_builder=lambda op, i, cv: yield_([voc_val]), default_body_builder=lambda op: yield_([dec_thunk()]))
                            nph_v = _sel(idx(1), lambda: idx(NPH), idx_t, [1] * 0)
                            for ph in for_(idx(0), nph_v, idx(1)):
                                I2v = _sel(idx(VOCAB_I2), lambda: _psw(ph, i2c, idx_t), idx_t, [])
                                J2v = _sel(idx(VOCAB_J2), lambda: _psw(ph, j2c, idx_t), idx_t, [])
                                pktv = _sel(_id4, lambda: _psw(ph, pktc, idx_t), idx_t, [])
                                for _v1 in for_(idx(0), I2v, idx(1)):
                                    for _e in range(PAIR_ROWS):
                                        _emit(_proj(J2v), pktv)
                                    yield_([])
                                yield_([])
                        return body

                    def _core_blk(base_cx):
                        return _core_blk_np(base_cx)
                    _arm_proj = _seg_arm
                    for blk in range(NCX // 2):
                        base_cx = blk * 2
                        _ops = [_arm_proj]
                        blk_h = herd(name=f'proj_blk{blk}', sizes=[2, 4], operands=_ops)(_core_blk(base_cx))
                        blk_h.attributes['link_with'] = StringAttr.get('proj_qmm.o')
                        blk_h.attributes['x_loc'] = IntegerAttr.get(T.i64(), PCOL[base_cx])
                        blk_h.attributes['y_loc'] = IntegerAttr.get(T.i64(), 2)
                    OPROJ_RNDS = PAIR_ROWS * I2P[1]
                    DOWN_RNDS = PAIR_ROWS * I2P[DOWN_PHASE]
                    _arm_rms = _seg_arm

                    def _rms_body(tx, ty, _sx, _sy, _arm):
                        _SINGLE_RMS = True
                        _rms_batched(_arm)
                        return

                    def _rms_batched_norm(xb, stg, scl, wbuf, nrefeed, _arm):
                        _nchunk = K // XCHUNK

                        def _i32(v):
                            return arith.ConstantOp(IntegerAttr.get(i32, v), None).result
                        for t in range(BATCH):
                            CallOp(rms_scale_row_aie, [scl, xb, _i32(t), _arm])
                        _n = nrefeed if isinstance(nrefeed, int) else None
                        _bound = idx(_n) if _n is not None else nrefeed
                        for _r in for_(idx(0), _bound, idx(1)):
                            for _c in for_(idx(0), idx(_nchunk), idx(1)):
                                CallOp(rms_chunk_aie, [stg, xb, wbuf, scl, _i32(BATCH), arith.index_cast(i32, _c), _i32(XCHUNK)])
                                ChannelPut('xnorm_fill' if NORMALIZATION_CACHE else 'xnorm', stg, offsets=[idx(0)], sizes=[idx(BATCH * XCHUNK)], strides=[idx(1)])
                                yield_([])
                            yield_([])

                    def _rms_batched(_arm):
                        assert RMS_DEST >= 0, 'DECODE_BATCH>1 needs the FULL4 residual path (RMS_DEST >= 0); the debug configs feed no o-proj'
                        assert K % XCHUNK == 0

                        def _i32(v):
                            return arith.ConstantOp(IntegerAttr.get(i32, v), None).result

                        def _cnt(voc, dec, ex=None):
                            return index_switch([idx_t], arith.index_cast(idx_t, _arm), [0], case_body_builder=lambda op, i, cv: yield_([idx(voc)]), default_body_builder=lambda op: yield_([idx(dec)]))

                        def _if_decode(body, decode_only=False):
                            _pred = arith.cmpi(arith.CmpIPredicate.eq, _arm, _i32(1)) if decode_only else arith.cmpi(arith.CmpIPredicate.ne, _arm, _i32(0))
                            _if = IfOp(_pred, [], has_else=False)
                            _if.operation.attributes['air.arm_select'] = UnitAttr.get()
                            with InsertionPoint(_if.thenRegion.blocks[0]):
                                body()
                                yield_([])
                        xb = AllocOp(rmsb_l1, [], [])
                        xr = None
                        sg2 = None
                        ChannelGet('rmsX', xb, indices=[idx(0)])
                        stg = AllocOp(rstg_l1, [], [])
                        scl = AllocOp(rscl_l1, [], [])
                        w = AllocOp(rms_l1, [], [])
                        w2 = AllocOp(rms_l1, [], [])

                        def _emit_norm(wbuf, nrefeed):
                            _rms_batched_norm(xb, stg, scl, wbuf, nrefeed, _arm)

                        def _accumulate(nrnds, stage, guard=None):
                            offsets, sizes, strides = outy_tokmajor(PAYLOAD)
                            for row in for_(idx(0), idx(nrnds), idx(1)):
                                ChannelGet('outY', stg, indices=[idx(0), idx(RMS_DEST)], offsets=[idx(v) for v in offsets], sizes=[idx(v) for v in sizes], strides=[idx(v) for v in strides])
                                off = arith.muli(arith.index_cast(i32, row), _i32(PAYLOAD))
                                for t in range(BATCH):
                                    CallOp(residual_acc_row_aie, [xb, stg, _i32(t), off, _i32(PAYLOAD)])
                                yield_([])

                        def _layer_out():
                            ChannelPut('layerOut', xb, offsets=[idx(0)], sizes=[idx(BATCH * K)], strides=[idx(1)])
                        ChannelGet('rmsW', w, indices=[idx(0)])
                        _emit_norm(w, 1 if NORMALIZATION_CACHE else _cnt(VOCAB_RNDS // NORMALIZATION_REPLAY, REFEED[0] // NORMALIZATION_REPLAY))
                        ChannelGet('rmsW2', w2, indices=[idx(0)])
                        _if_decode(lambda: _accumulate(OPROJ_RNDS, 1, guard=None))
                        _emit_norm(w2, _cnt(0, 1 if NORMALIZATION_CACHE else REFEED[GATEUP_PHASE] // NORMALIZATION_REPLAY))
                        _if_decode(lambda: _accumulate(DOWN_RNDS, 2), decode_only=bool(0))
                        _if_decode(_layer_out)
                        for _b in [w2] + ([xr] if xr is not None else []) + ([sg2] if sg2 is not None else []) + [w, scl, stg, xb]:
                            DeallocOp(_b)
                    rms_h = herd(name='rms', sizes=[1, 1], operands=[_arm_rms])(_rms_body)
                    rms_h.attributes['link_with'] = StringAttr.get('rms_residual.o')
                    rms_h.attributes['x_loc'] = IntegerAttr.get(T.i64(), RMS_PCOL)
                    rms_h.attributes['y_loc'] = IntegerAttr.get(T.i64(), 2)

                    if NORMALIZATION_CACHE:
                        # A free core keeps the full [chunk, token, element]
                        # activation. Its program orders variable-length phase
                        # replays; only the 8-KiB output chunk uses a DMA ring.
                        # 40-KiB slab + 8-KiB stage + 6,080-byte stack fit L1.
                        @herd(name='norm_cache', sizes=[1, 1], operands=[_arm_rms])
                        def cache_h(tx, ty, sx, sy, arm):
                            slab = AllocOp(rmsb_l1, [], [])
                            out = AllocOp(rstg_l1, [], [])
                            for buf in (slab, out):
                                buf.operation.attributes['air.disable_alloc_sink'] = UnitAttr.get()

                            def emit(count):
                                ChannelGet('xnorm_fill', slab)
                                for _r in for_(idx(0), count, idx(1)):
                                    for c in for_(idx(0), idx(K // XCHUNK), idx(1)):
                                        CallOp(cache_copy, [out, slab, arith.index_cast(i32, c)])
                                        ChannelPut('xnorm', out, offsets=[idx(0)], sizes=[idx(BATCH * XCHUNK)], strides=[idx(1)])
                                        yield_([])
                                    yield_([])
                            is_decode = arith.cmpi(arith.CmpIPredicate.ne, arm,
                                arith.ConstantOp(IntegerAttr.get(i32, 0), None).result)
                            first_count = arith.select(is_decode, idx(REFEED[0]), idx(VOCAB_RNDS))
                            emit(first_count)
                            branch = IfOp(is_decode, [], has_else=False)
                            branch.operation.attributes['air.arm_select'] = UnitAttr.get()
                            with InsertionPoint(branch.thenRegion.blocks[0]):
                                emit(idx(REFEED[GATEUP_PHASE]))
                                yield_([])
                            DeallocOp(out)
                            DeallocOp(slab)
                        cache_h.attributes['link_with'] = StringAttr.get('norm_cache.o')
                        cache_h.attributes['x_loc'] = IntegerAttr.get(T.i64(), RMS_PCOL)
                        cache_h.attributes['y_loc'] = IntegerAttr.get(T.i64(), 3)

            def _emit_wave_loop(lo, hi):
                if lo >= hi:
                    return
                for _iv in for_(idx(lo), idx(hi), idx(1)):
                    launch(sizes=[1, 1], operands=list(_fa) + [_iv], attributes={'air.preserve_shim_dma_order': UnitAttr.get()})(launch_body)
                    yield_([])
            _emit_wave_loop(UNI_WAVE_LO, UNI_WAVE_HI)
    return build()
