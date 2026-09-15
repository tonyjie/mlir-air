"""Explicit configuration and buffer ABI for the standalone NPU target.

Importing the upstream builder is confined to a save/restore environment adapter.
The runtime consumes the saved manifest rather than ambient DECODE_* settings.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[1]
REPO = HERE.parents[2]
FUSED = EXAMPLES/'fused_decode'
ABI_VERSION = 1


@dataclass(frozen=True)
class TargetConfig:
    role: str = "target"
    capacity: int = 512
    batch: int = 8
    layers: int = 36
    taps: bool = True
    causal: bool = True
    stack_bytes: int = 6080
    l2_pingpong: bool = True
    chain_pingpong: bool = True
    activation_memtile_col: int = 2
    gather_mode: str = "packet"
    token_packets: bool = False
    dynamic_context: bool = False
    projection_weight_pingpong: bool = False
    attention_query_tile: int = 1
    attention_layout: str = "pipeline"
    normalization_replay: int = 1
    normalization_cache: bool = False
    attention_pair_mode: int = 0
    vector_attention_norm: bool = False
    attention_optimization: str = "none"
    q4k_unpack_fma: bool = False
    q4k_unpack_unroll: int = 0

    def __post_init__(self):
        if self.attention_layout not in ('pipeline','fused-shards'):
            raise ValueError('Unknown attention layout')
        if self.attention_layout == 'fused-shards' and (self.batch != 8 or self.attention_query_tile != 4 or self.attention_pair_mode != 3):
            raise ValueError('Parallel attention requires B8, query tile 4, and both paired operations')
        if self.attention_optimization not in ('none','correction','pv','both'):
            raise ValueError('Unknown attention optimization')
        if self.attention_optimization != 'none' and (self.batch != 8 or self.attention_query_tile < 2 or self.attention_pair_mode != 3):
            raise ValueError('Attention candidates require paired B8 query tiles')
        if self.role not in ('target', 'drafter'):
            raise ValueError('role must be target or drafter')
        if self.role == 'drafter' and (self.layers != 5 or self.batch != 8 or self.causal):
            raise ValueError('The drafter requires five layers, batch 8, and bidirectional attention')
        if self.batch not in (1,8):
            raise ValueError('The current target supports batch 1 or 8; block 16 needs a new memory schedule')
        if self.capacity < 16 or self.capacity % 16:
            raise ValueError('capacity must be a positive multiple of the 16-key tile')
        if not 1 <= self.layers <= 36:
            raise ValueError('layers must be in [1,36]')
        if self.batch==8 and not 0 < self.stack_bytes <= 6080:
            raise ValueError('batch-8 RMS reserves 59,424 data bytes and 32 RTP/alignment bytes; stack must fit in 6,080 bytes')
        if self.token_packets and self.gather_mode != 'packet':
            raise ValueError('Token packets require packet gather mode')
        if self.gather_mode not in ('packet', 'circuit', 'circuit-main'):
            raise ValueError('Unknown projection gather mode')
        if self.activation_memtile_col not in (2,4):
            raise ValueError('activation relay supports memory-tile columns 2 and 4')
        if self.batch==1 and not self.causal:
            raise ValueError('batch-1 control uses causal attention')
        if self.attention_pair_mode not in (0, 1, 2, 3) or (self.attention_pair_mode and self.attention_query_tile < 2):
            raise ValueError('Attention pair mode needs a query tile of at least two')
        if self.normalization_replay not in (1, 2) or (self.batch == 1 and self.normalization_replay != 1):
            raise ValueError('Normalized activation replay is 1 or 2 for batch 8')
        if self.normalization_cache and (self.batch != 8 or self.normalization_replay != 1):
            raise ValueError('Core activation cache requires B8 and no memory-tile replay experiment')
        if self.attention_query_tile not in (1, 2, 4) or self.batch % self.attention_query_tile:
            raise ValueError('Attention query tile must divide the batch and be 1, 2, or 4')
        if self.projection_weight_pingpong and self.batch != 8:
            raise ValueError('Selective weight ping-pong is a batch-8 candidate')
        if self.q4k_unpack_unroll not in (0, 2, 4, 8, 16):
            raise ValueError('Q4 unpack unroll must be 0, 2, 4, 8, or 16')
        if self.batch != 8 and (self.q4k_unpack_fma or self.q4k_unpack_unroll):
            raise ValueError('Q4 unpack tuning applies only to the batch-8 matrix path')

    @property
    def build_lengths(self):
        last=self.capacity-self.batch+1
        return (last,last-1)


@contextmanager
def builder_environment(config, length):
    saved=dict(os.environ)
    try:
        for name in list(os.environ):
            if name.startswith(('DECODE_','RMS_','PROJ_','Q4K_','UNI_')):
                del os.environ[name]
        os.environ.update(DECODE_MODEL='qwen3-4b',VOCAB_CHUNK_I2='30',
                          DECODE_GOLDEN='1',DECODE_GOLDEN_L=str(length),
                          DECODE_UNI_DEC=str(config.layers),DECODE_STACK=str(config.stack_bytes),
                          DECODE_HIDDEN_TAPS=str(int(config.taps)),W_DUAL_CHAN='1',
                          NLAYERS='1',LM_HEAD='0',
                          DECODE_DYNSEQ=str(int(config.dynamic_context)))
        yield
    finally:
        os.environ.clear();os.environ.update(saved)


def load_geometry(config, length=None):
    length=config.build_lengths[0] if length is None else length
    if length not in config.build_lengths:
        raise ValueError('build length must be one of the explicit calibration pair')
    if str(FUSED) not in sys.path:
        sys.path.insert(0,str(FUSED))
    with builder_environment(config,length):
        spec=importlib.util.spec_from_file_location('_dflash_upstream_geometry',FUSED/'fused_decode.py')
        fd=importlib.util.module_from_spec(spec);spec.loader.exec_module(fd)
    assert fd.ATTN_MAXL==config.capacity
    assert (fd.K,fd.M,fd.DK_TOT_A,fd.DH,fd.N_ATTN_CU,fd.PAIR_ROWS)==(2560,6144,1024,128,4,1)
    return fd


def abi(config,fd):
    b=config.batch
    rope_offset=config.layers*fd.RMS_LAYER
    final_offset=rope_offset+config.layers*b*fd.ROPE_W_LEN
    decode_y=(fd.HOST_ROUNDS+fd.LAYER_RNDS)*fd.PAYLOAD*b
    return {
        'version':ABI_VERSION,'model':('z-lab/Qwen3-4B-DFlash-b16' if config.role=='drafter' else 'Qwen/Qwen3-4B'),'config':asdict(config),
        'quantization':'Q4NX packed q4k cascade, dual channel',
        'attention_context':'live blocks' if config.dynamic_context else 'compiled capacity',
        'attention_output':'cascade join then memory-tile gather' if config.attention_layout == 'fused-shards' else 'memory-tile gather',
        'activation':'BF16','projection_compute':'BFP16 mmul<8,8,8>' if b==8 else 'GEMV',
        'q4k_unpack_arithmetic':('single BF16 rounding after FP32 multiply-add' if config.q4k_unpack_fma
                                else 'BF16 product rounded before offset addition') if b==8 else 'fused GEMV',
        'rounding':'nearest-even explicitly set at kernel entry' if b==8 else 'current-main Qwen default',
        'geometry':{'hidden':fd.K,'vocab':fd.VOCAB_SIZE,'vocab_chunks':fd.UNI_LM,
                    'vocab_chunk_width':fd.VOCAB_SIZE_PADDED,'kv_groups':fd.NGRP,
                    'kv_region_width':fd.REGION_W,'rope_width':fd.ROPE_W_LEN},
        'elements':{'x':(config.layers+1 if config.taps else 1)*b*fd.K,
                    'w':config.layers*fd.W_LAYER+fd.UNI_LM*fd.VOCAB_W_BLOCKS*fd.BLOCK_BF16,
                    'r':final_offset+fd.K,'y':decode_y+fd.UNI_LM*b*fd.VOCAB_SIZE_PADDED,
                    'kv':config.layers*config.capacity*fd.KVSZ_TOK},
        'offsets':{'rope':rope_offset,'final_norm':final_offset,'logits':decode_y},
        'kv_layout':'[layer, K-groups then V-groups, position, region-width]',
        'logits_layout':'[vocabulary chunk, token, within-chunk vocabulary]',
        'hidden_layout':'[layer boundary, token, hidden]; boundary 0 is the input',
    }


def build_module(config, length):
    fd=load_geometry(config,length)
    if config.batch==1:
        with builder_environment(config,length):
            return fd.build_module(),abi(config,fd)
    import batch_graph
    b=config.batch
    def outy_tokmajor(row_stride,base=0,rounds=1):
        if config.token_packets:
            return ([0,0,0,base],[rounds,b,fd.N_PAIRS,fd.PAIR_PAY],[fd.PAYLOAD,row_stride,fd.PAIR_PAY,1])
        return ([0,0,base],[fd.N_PAIRS*rounds,b,fd.PAIR_PAY],[fd.PAIR_PAY,row_stride,1])
    extra={
        'BATCH':b,'CAUSAL':config.causal,'ARM_IDLE_CASES':[0],
        'PROJECTION_WEIGHT_PINGPONG':config.projection_weight_pingpong,
        'ATTENTION_QUERY_TILE':config.attention_query_tile,
        'PARALLEL_ATTENTION':config.attention_layout == 'fused-shards',
        'KV_QUERY_GROUPS':1 if config.attention_layout == 'fused-shards' else b//config.attention_query_tile,
        'NORMALIZATION_REPLAY':config.normalization_replay,
        'NORMALIZATION_CACHE':config.normalization_cache,
        'XMT_PCOL':config.activation_memtile_col,
        'CIRCUIT_GATHER':config.gather_mode == 'circuit',
        'CIRCUIT_MAIN':config.gather_mode != 'packet',
        'TOKEN_PACKETS':config.token_packets,
        'PACKET_BATCH':1 if config.token_packets else b,
        'GRP_ROWS_B':fd.HDR+fd.LEADS_PER_GRP*fd.PAIR_PAY*(1 if config.token_packets else b),
        'MAIN_ROWS_B':fd.HDR+fd.PAYLOAD*(1 if config.token_packets else b),'PAYLOAD_B':fd.PAYLOAD*b,
        'L1_VEC_BF16':32,'STG_W':max(2*fd.COL_BLOCK,fd.PAYLOAD),
        'XCHUNK':2*fd.COL_BLOCK,'XCHUNK_MUL':2,'VOCAB_DEST':fd.GLU_DEST,
        'VOC_SLICE_RNDS':fd.GLU_SLICE//fd.PAYLOAD,
        'VOC_ROTATIONS':fd.VOCAB_RNDS//(2*(fd.GLU_SLICE//fd.PAYLOAD)),
        '_ATTN_COLS':{col for col,_,_ in fd.ATTN_CU_LOC},
        '_XFEED_BD':([fd.COL_BLOCK//8,b,8],[8,2*fd.COL_BLOCK,1]),
        'outy_tokmajor':outy_tokmajor,
        'contextlib':__import__('contextlib'),
    }
    from fused_attention_graph import ParallelAttentionGraph
    ns=vars(fd).copy();ns.update(extra)
    ns['ATTENTION_ENV']=SimpleNamespace(**ns)
    ns['ParallelAttentionGraph']=ParallelAttentionGraph
    fn=FunctionType(batch_graph.build_module.__code__,ns)
    module=fn()
    if not config.chain_pingpong:
        # Disable only late twin synthesis, preserving v2 producer/consumer
        # chain locks. The earlier L2 pipeline pass is a separate control.
        from air.ir import IntegerAttr, MemRefType, UnitAttr
        def mark(operation):
            if operation.name == 'memref.alloc':
                space = MemRefType(operation.results[0].type).memory_space
                if isinstance(space, IntegerAttr) and IntegerAttr(space).value == 1:
                    operation.attributes['air.no_chain_pingpong'] = UnitAttr.get()
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        mark(child.operation)
        with module.context:
            mark(module.operation)
    return module,abi(config,fd)
