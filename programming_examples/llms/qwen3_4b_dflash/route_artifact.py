"""Apply the checked post-routing assignments required by the selected B8 floorplan.

Uses aiecc's documented checkpoint/resume interface and original per-core
ELFs. Preserves instructions and records the exact IR edit and binary hashes.
This remains a guarded post-route step until the general router encodes these ownership constraints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pin_master(text, tile, port, expected, replacement):
    """Move one checked master output; reject ambiguous/broadcast AMSels.

    Identify the physical output port, not the incidental SSA value number.
    This diagnostic supports only the recorded Qwen batch-8 floorplan.
    """
    col,row=tile;bundle,channel=port
    pattern=rf'(%switchbox_{col}_{row} = aie.switchbox\([^\n]+\) \{{)(.*?)(\n    \}})'
    switch=re.search(pattern,text,re.S)
    if not switch:raise ValueError(('Missing switchbox',tile))
    body=switch.group(2)
    masters=list(re.finditer(rf'aie.masterset\({bundle} : {channel}, (%\w+)\)',body))
    if len(masters)!=1:raise ValueError(('Expected single-AMSel master',tile,port))
    ref=masters[0].group(1)
    users=[m for m in re.findall(r'aie.masterset\([^\n]+',body) if ref in re.findall(r'%\w+',m)]
    if len(users)!=1:raise ValueError(('AMSel has multiple outputs',tile,ref))
    match=re.search(rf'{re.escape(ref)} = aie.amsel<(\d+)> \((\d+)\)',body)
    if match is None or tuple(map(int,match.groups()))!=expected:
        raise ValueError(('Unexpected current assignment',tile,port))
    arbiter,msel=replacement
    if f'aie.amsel<{arbiter}> ({msel})' in body:
        raise ValueError(('Replacement is already allocated',tile,replacement))
    body=body[:match.start()]+f'{ref} = aie.amsel<{arbiter}> ({msel})'+body[match.end():]
    return text[:switch.start(2)]+body+text[switch.end(2):]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fix-shim-arbiter',action='store_true')
    ap.add_argument('--isolate-main-output',action='store_true')
    ap.add_argument('--isolate-normalization-replay',action='store_true')
    args=ap.parse_args()
    src=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((src/'manifest.json').read_text())
    if args.fix_shim_arbiter or args.isolate_main_output or args.isolate_normalization_replay:
        cfg=manifest['config']
        if (cfg['batch'],cfg['activation_memtile_col'],cfg['gather_mode']) != (8,4,'packet'):
            raise ValueError('Arbiter experiment requires the Qwen block-8 packet-gather floorplan')
        # Packet payload length does not identify a physical port. pin_master
        # still checks the exact source port, current arbiter and unique AMSel
        # before applying either change for every packet geometry.
    if args.isolate_normalization_replay and manifest['config'].get('normalization_replay',1) != 2:
        raise ValueError('Normalization routing probe requires replay factor two')
    manifest['status']='building';manifest['artifacts']={}
    manifest['routing_diagnostic']={'source':str(src),'source_manifest_sha256':sha(src/'manifest.json'),
        'fix_shim_arbiter':args.fix_shim_arbiter,'isolate_main_output':args.isolate_main_output,'isolate_normalization_replay':args.isolate_normalization_replay,'core_elf_hashes':{},'physical_ir_hashes':{}}
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest['routing_diagnostic']['routing_helper_sha256']=sha(Path(__file__))
    (out/'route_artifact_source.py').write_bytes(Path(__file__).read_bytes())
    save()
    dynamic=manifest['config'].get('dynamic_context',False)
    lengths=(manifest['config']['capacity']-manifest['config']['batch']+1,
             manifest['config']['capacity']-manifest['config']['batch'])
    if dynamic:lengths=lengths[:1]
    try:
        for length in lengths:
            old=src/f'compile_L{length}'/'air_project'
            work=out/f'compile_L{length}';work.mkdir()
            cp=work/'checkpoint'
            prefix=f"decode_dyn_L{manifest['config']['capacity']}" if dynamic else f'decode_L{length}'
            cmd=['aiecc',str(old/'npu.air.mlir'),'--get-xclbin','--cut=input_physical.mlir',
                 '--checkpoint='+str(cp),'--tmpdir='+str(work/'intermediates'),
                 '--output-dir='+str(work),'--xclbin-name='+str(out/f'{prefix}.xclbin'),
                 '--xclbin-kernel-name=MLIR_AIE','--xclbin-instance-name=MLIRAIE',
                 '--xclbin-kernel-id=0x901','--no-progress']
            subprocess.run(cmd,check=True)
            files=[p for p in cp.rglob('*.mlir') if p.is_file()]
            physical=[p for p in files if 'aie.switchbox' in p.read_text()]
            if len(physical)!=1:raise ValueError(('Unexpected checkpoint',files))
            path=physical[0];s=path.read_text()
            manifest['routing_diagnostic']['physical_ir_hashes'][str(length)+'_before']=sha(path)
            if args.fix_shim_arbiter:
                if manifest['config'].get('attention_layout') == 'fused-shards':
                    # Parallel-attention floorplan: separate gather 0/6 and
                    # RMS/gather-7. These replacements share only host drains,
                    # not inputs whose destinations wait on future phases.
                    s=pin_master(s, (2,0), ('North',0), (1,1), (4,1))
                    s=pin_master(s, (2,0), ('North',3), (0,1), (5,1))
                elif manifest['config'].get('normalization_cache',False):
                    # Cache core at (5,2): group 0/7 and group 1/6 otherwise
                    # share shim arbiters. Audited against the physical routes.
                    s=pin_master(s, (2,0), ('North',4), (2,0), (4,1))
                    s=pin_master(s, (2,0), ('North',5), (1,1), (3,1))
                    # QKV output must not share an arbiter with future RMS
                    # input packets whose destination is held by this layer.
                    s=pin_master(s, (2,0), ('East',0), (0,1), (5,1))
                else:
                    s=pin_master(s, (2,0), ('North',4), (1,1), (4,1))
            if args.isolate_main_output:
                if manifest['config'].get('attention_layout') == 'fused-shards':
                    # Both main-output routes originate at DMA0, so sharing
                    # their arbiter frees one for independent gather input 1.
                    s=pin_master(s, (2,1), ('South',0), (1,0), (0,2))
                    s=pin_master(s, (2,1), ('DMA',2), (2,1), (1,2))
                elif manifest['config'].get('normalization_cache',False):
                    # Keep main output independent of group-7 gather input.
                    s=pin_master(s, (2,1), ('North',5), (0,0), (1,2))
                elif manifest['config'].get('normalization_replay', 1) > 1:
                    # A separate normalization-replay DMA changes the router's
                    # allocation. Audited gather paths already use distinct
                    # shim arbiters; run this layout without fix-shim-arbiter.
                    # The main output (DMA1 -> North0) still shares arbiter 1
                    # with the third gather input, so isolate that output.
                    s=pin_master(s, (2,1), ('North',0), (1,0), (2,2))
                else:
                    s=pin_master(s, (2,1), ('North',1), (0,0), (1,2))
            if args.isolate_normalization_replay:
                # Circuit-fill layout: gather inputs use arbiters 2/0/5/4;
                # main output already uses arbiter 1. Move the replay output
                # off arbiter 0, which otherwise couples it to group-1 gather.
                s=pin_master(s, (2,1), ('South',2), (0,0), (3,1))
            # Attach identical prebuilt ELF files, avoiding all core recompilation.
            lines=s.splitlines(keepends=True);active=None;depth=0;count=0
            for i,line in enumerate(lines):
                m=re.search(r'aie.core\(%tile_(\d+)_(\d+)\)',line)
                if m:
                    assert active is None
                    active=tuple(map(int,m.groups()));depth=0;start=i
                if active is not None:
                    depth+=line.count('{')-line.count('}')
                    if depth==0:
                        c,r=active
                        elf=old/f'elfs_seg_core_{c}_{r}'/f'elfs_seg_core_{c}_{r}.elf'
                        assert elf.is_file(),elf
                        manifest['routing_diagnostic']['core_elf_hashes'][str(elf)]=sha(elf)
                        assert '} {' in line and 'elf_file' not in line
                        lines[i]=line.replace('} {','} {elf_file = "'+str(elf)+'", ',1)
                        for j in range(start+1,i): lines[j]=''
                        lines[start+1]='      aie.end\n'
                        count+=1;active=None
            expected_cores=(28 if manifest['config'].get('normalization_cache',False) else 27) + (2 if manifest['config'].get('attention_layout') == 'fused-shards' else 0)
            assert active is None and count==expected_cores,count
            path.write_text(''.join(lines))
            manifest['routing_diagnostic']['physical_ir_hashes'][str(length)+'_after']=sha(path)
            subprocess.run(['aiecc','--resume='+str(cp/'manifest.json')],check=True)
            sidecar='txn.h' if dynamic else 'insts.bin'
            inst=out/f'{prefix}.{sidecar}';shutil.copy2(src/inst.name,inst)
            for ext in ('xclbin',sidecar):
                artifact=out/f'{prefix}.{ext}'
                manifest['artifacts'][artifact.name]=sha(artifact)
            save()
        manifest['status']='compiled';save();print('compiled routing diagnostic',out,flush=True)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error));save();raise

if __name__=='__main__':main()
