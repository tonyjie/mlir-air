"""CPU vs host-NPU-path timing for the NON-ViT-layer parts of the vision encoder."""
import time, sys, numpy as np, torch
sys.path.insert(0,'.')
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
torch.set_grad_enabled(False)
p = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
vlm = p.model.vlm_with_expert.get_vlm_model()
vm, conn = vlm.vision_model, vlm.connector
dt = next(vm.parameters()).dtype
px = torch.zeros(1,3,512,512, dtype=dt)
hid = torch.randn(1,1024,768, dtype=dt)

def med(fn,n=20):
    fn(); ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e6)
    return np.median(ts)

print("=== CPU (lerobot torch) ===")
c_emb  = med(lambda: vm.embeddings(pixel_values=px, patch_attention_mask=torch.ones((1,32,32),dtype=torch.bool)))
c_pln  = med(lambda: vm.post_layernorm(hid))
c_conn = med(lambda: conn(hid))
print(f"  patch embed + pos (Conv2d)     {c_emb:9.1f} us")
print(f"  post_layernorm                 {c_pln:9.1f} us")
print(f"  connector (shuffle + proj)     {c_conn:9.1f} us")

print("\n=== NPU path, HOST-side parts (numpy) ===")
sys.path.insert(0,'..')
from vision_weights import load_vision_weights, SigLIPVisionConfig
from vision_cpu_helpers import im2col_patch_embed, pixel_shuffle
from ml_dtypes import bfloat16
cfg=SigLIPVisionConfig(); w=load_vision_weights("lerobot/smolvla_base", dtype=bfloat16, config=cfg)
pv = np.zeros((3,512,512), np.float32)
pw, pb, pe = w.patch_w.astype(np.float32), w.patch_b.astype(np.float32), w.pos_embed.astype(np.float32)
# split im2col into extraction vs matmul
C,H,W_=pv.shape; g=H//16
def extract():
    cols=np.empty((1024, 768), np.float32)
    for ph in range(g):
        for pw_ in range(g):
            cols[ph*g+pw_] = pv[:, ph*16:(ph+1)*16, pw_*16:(pw_+1)*16].reshape(-1)
    return cols
cols = extract()
h_ext = med(extract, 10)
h_mm  = med(lambda: cols @ pw + pb, 10)
h_all = med(lambda: im2col_patch_embed(pv, pw, pb, pe, 16), 10)
post = np.random.randn(1024,768).astype(np.float32)
h_ps  = med(lambda: pixel_shuffle(post), 20)
print(f"  im2col patch extraction        {h_ext:9.1f} us")
print(f"  patch matmul 1024x768x768      {h_mm:9.1f} us   <-- ON HOST, NPU does this shape in 578 us")
print(f"  (whole im2col_patch_embed)     {h_all:9.1f} us")
print(f"  pixel_shuffle (pure reshape)   {h_ps:9.1f} us")
