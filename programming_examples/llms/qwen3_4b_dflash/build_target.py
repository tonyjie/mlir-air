"""Compile isolated current-toolchain target artifacts and save their exact ABI.

Invoke through env.sh under /tmp/mlir-air-npu.lock. Each template keeps its own
compiler intermediates; sources and artifacts are hashed in manifest.json.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from target_config import HERE, EXAMPLES, FUSED, REPO, TargetConfig, build_module, load_geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--role',choices=('target','drafter'),default='target')
    ap.add_argument('--capacity',type=int,default=512)
    ap.add_argument('--batch',type=int,choices=(1,8),default=8)
    ap.add_argument('--layers',type=int,default=36)
    ap.add_argument('--taps',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--causal',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--stack-bytes',type=int)
    ap.add_argument('--l2-pingpong',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--chain-pingpong',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--activation-memtile-col',type=int,choices=(2,4),default=2)
    ap.add_argument('--gather-mode',choices=('packet','circuit','circuit-main'),default='packet')
    ap.add_argument('--token-packets',action='store_true')
    ap.add_argument('--dynamic-context',action='store_true')
    ap.add_argument('--projection-weight-pingpong',action='store_true')
    ap.add_argument('--attention-query-tile',type=int,choices=(1,2,4),default=1)
    ap.add_argument('--attention-layout',choices=('pipeline','fused-shards'),default='pipeline')
    ap.add_argument('--normalization-replay',type=int,choices=(1,2),default=1)
    ap.add_argument('--normalization-cache',action='store_true',help='Retain normalized activations on one spare NPU core')
    ap.add_argument('--vector-attention-norm',action='store_true')
    ap.add_argument('--attention-optimization',choices=('none','correction','pv','both'),default='none')
    ap.add_argument('--q4k-unpack-fma',action='store_true')
    ap.add_argument('--q4k-unpack-unroll',type=int,choices=(0,2,4,8,16),default=0)
    ap.add_argument('--attention-pair-mode',type=int,choices=(0,1,2,3),default=0,help='0=original, 1=QK pairs, 2=PV pairs, 3=both')
    args=ap.parse_args()
    cfg=TargetConfig(role=args.role,capacity=args.capacity,batch=args.batch,layers=args.layers,
                     taps=args.taps,causal=args.causal,l2_pingpong=args.l2_pingpong,
                     chain_pingpong=args.chain_pingpong,
                     activation_memtile_col=args.activation_memtile_col,
                     gather_mode=args.gather_mode,token_packets=args.token_packets,
                     dynamic_context=args.dynamic_context,
                     projection_weight_pingpong=args.projection_weight_pingpong,
                     attention_query_tile=args.attention_query_tile,
                     attention_layout=args.attention_layout,
                     normalization_replay=args.normalization_replay,
                     normalization_cache=args.normalization_cache,
                     attention_pair_mode=args.attention_pair_mode,
                     vector_attention_norm=args.vector_attention_norm,
                     attention_optimization=args.attention_optimization,
                      q4k_unpack_fma=args.q4k_unpack_fma,
                     q4k_unpack_unroll=args.q4k_unpack_unroll,
                     stack_bytes=args.stack_bytes or (6080 if args.batch==8 else 10240))
    dest=args.output.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    fd=load_geometry(cfg)
    inputs=list(HERE.glob('*.py'))+list((HERE/'kernels').glob('*'))
    inputs+=list(FUSED.glob('*.py'))+list((FUSED/'kernels').glob('*'))+list((FUSED/'models').glob('*'))
    manifest={'config':asdict(cfg),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
              'source_hashes':{str(p.relative_to(EXAMPLES)):sha(p) for p in sorted(inputs) if p.is_file()},
              'toolchain':{k:os.environ[k] for k in ('MLIR_AIR_INSTALL_DIR','MLIR_AIE_INSTALL_DIR','PEANO_INSTALL_DIR','LLVM_INSTALL_DIR')},
              'status':'building','commands':[],'artifacts':{}}
    air_opt=Path(shutil.which('air-opt')).resolve()
    provenance = Path(os.environ['MLIR_AIR_INSTALL_DIR']) / 'dflash-build.json'
    if not provenance.is_file():
        raise RuntimeError('Build this branch with setup_toolchain.py first: AIR provenance is missing')
    compiler_build = json.loads(provenance.read_text())
    if compiler_build['air_opt_sha256'] != sha(air_opt):
        raise RuntimeError('air-opt differs from the recorded reviewer compiler build')
    for name, expected in compiler_build['compiler_sources'].items():
        if sha(REPO / name) != expected:
            raise RuntimeError('AIR source differs from installed compiler: ' + name)
    manifest['air_build_manifest_sha256'] = sha(provenance)
    compiler_source=REPO/'mlir/lib/Conversion/AIRToAIEPass.cpp'
    manifest['compiler']={'air_opt':str(air_opt),'air_opt_sha256':sha(air_opt),
                          'air_to_aie_source_sha256':sha(compiler_source),
                          'air_to_aie_schedule_source_sha256':sha(compiler_source.with_name('AIRToAIESchedulingUtils.cpp')),
                          'air_schedule_source_sha256':sha(compiler_source.parent.parent/'Transform/AIRDependencyScheduleOpt.cpp'),
                          'dependency_util_source_sha256':sha(compiler_source.parent.parent/'Util/Dependency.cpp'),
                          'air_rt_to_npu_source_sha256':sha(compiler_source.with_name('AIRRtToNpuPass.cpp'))}
    import tarfile
    with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as archive:
        for p in sorted(inputs):
            if p.is_file():archive.add(p,arcname=str(p.relative_to(EXAMPLES)))
    write_json(dest/'manifest.json',manifest)
    clang=Path(os.environ['PEANO_INSTALL_DIR'])/'bin/clang++'
    flags=['-std=c++20','--target=aie2p-none-unknown-elf','-Wno-parentheses','-Wno-attributes',
           '-Wno-macro-redefined','-Wno-empty-body','-Wno-deprecated-declarations',
           '-DNDEBUG','-DMODEL_TYPE=QWEN3_4B','-D__AIE_API_AIE_ADF_HPP__',
           '-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16',
           '-DGLU_SLICE_EXPECTED='+str(fd.GLU_SLICE),
           '-I'+str(Path(os.environ['MLIR_AIE_INSTALL_DIR'])/'include'),
           '-I'+str(FUSED/'kernels'),'-I'+str(FUSED/'models'),
           '-I'+str(EXAMPLES),'-I'+str(HERE/'kernels')]
    if cfg.vector_attention_norm:
        flags+=['-DDFLASH_VECTOR_ATTN_NORM']
    if cfg.attention_optimization in ('correction','both'):
        flags+=['-DDFLASH_PACKED_SOFTMAX_CORRECTION']
    if cfg.attention_optimization in ('pv','both'):
        flags+=['-DDFLASH_FUSED_PV_RESCALE']
    if cfg.q4k_unpack_fma:
        flags+=['-DQ4K_UNPACK_FMA']
    if cfg.q4k_unpack_unroll:
        flags+=['-DQ4K_UNPACK_UNROLL='+str(cfg.q4k_unpack_unroll)]
    if cfg.batch==8:
        flags+=['-DPROJ_MM_BATCH=8','-DATTN_BATCH=8','-DDECODE_ROUND_NEAREST',
                '-DATTN_QUERY_TILE='+str(cfg.attention_query_tile),
                '-DATTN_PAIR_QK='+str(int(bool(cfg.attention_pair_mode & 1))),
                '-DATTN_PAIR_PV='+str(int(bool(cfg.attention_pair_mode & 2)))]
    objdir=dest/'objects';objdir.mkdir()
    mapping={'proj_qmm':'proj','rms_residual':'rms','glu':'glu','attn_qk':'attn_qk','attn_kv':'attn_kv'}
    try:
        # The ABI of Peano's lifetime intrinsics requires llvm-link <23.
        subprocess.run(['make','-C',str(FUSED),'preflight-llvm-link'],check=True)
        for name in ('proj_qmm','rms_residual','glu','rope') + (('attn_fused',) if cfg.attention_layout=='fused-shards' else ('attn_qk','attn_kv')) + (('norm_cache',) if cfg.normalization_cache else ()):
            src=HERE/'kernels'/(mapping.get(name,name)+'_batch.cc') if cfg.batch==8 and name!='rope' else FUSED/'kernels'/(name+'.cc')
            if name == 'norm_cache':
                src=HERE/'kernels/norm_cache.cc'
            attn=name.startswith('attn_')
            out=objdir/(name+('.ll' if attn else '.o'))
            cmd=[str(clang),*flags,'-O1' if attn else '-O2']
            cmd+=['-DDECODE_INLINE_ATTN','-S','-emit-llvm'] if attn else ['-c']
            cmd += [str(src),'-o',str(out)]
            manifest['commands'].append(cmd)
            print('compiling',name,flush=True)
            subprocess.run(cmd,check=True)
        from air.backend.xrt import XRTBackend
        for length in (cfg.build_lengths[:1] if cfg.dynamic_context else cfg.build_lengths):
            work=dest/f'compile_L{length}';work.mkdir()
            for p in objdir.iterdir():shutil.copy2(p,work/p.name)
            module,interface=build_module(cfg,length)
            assert module.operation.verify()
            (work/'input.mlir').write_text(str(module))
            old=Path.cwd()
            try:
                os.chdir(work)
                backend=XRTBackend(omit_while_true_loop=False,output_format='xclbin',
                    kernel_name='MLIR_AIE',stack_size=cfg.stack_bytes,
                    omit_pingpong='' if cfg.l2_pingpong else 'L2',
                    use_lock_race_condition_fix_v2=True,coalesce_shim_dma=bool(fd.COALESCE),
                    emit_txn_cpp=cfg.dynamic_context)
                print('compiling template',length,'capacity',cfg.capacity,flush=True)
                artifact=backend.compile(module,output_binary_name='decode',insts='decode.insts.bin')
            finally:
                os.chdir(old)
            files=({'xclbin':work/'decode.xclbin','txn.h':work/artifact.txn_header} if cfg.dynamic_context
                   else {suffix:work/f'decode.{suffix}' for suffix in ('xclbin','insts.bin')})
            prefix=f'decode_dyn_L{cfg.capacity}' if cfg.dynamic_context else f'decode_L{length}'
            for suffix,source in files.items():
                out=dest/f'{prefix}.{suffix}'
                shutil.copy2(source,out)
                manifest['artifacts'][out.name]=sha(out)
            manifest['abi']=interface
            write_json(dest/'manifest.json',manifest)
        manifest['status']='compiled'
        write_json(dest/'manifest.json',manifest)
        print('compiled',dest,flush=True)
    except BaseException as e:
        manifest['status']='failed'
        manifest['error']=repr(e)
        write_json(dest/'manifest.json',manifest)
        raise


if __name__=='__main__':
    main()
