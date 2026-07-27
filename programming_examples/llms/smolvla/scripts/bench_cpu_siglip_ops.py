"""Per-operation CPU timing of ONE SigLIP encoder layer, to sit beside the NPU breakdown."""
import time, numpy as np, torch, sys
from collections import defaultdict
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

torch.set_grad_enabled(False)
p = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
vm = p.model.vlm_with_expert.get_vlm_model().vision_model
layer = vm.encoder.layers[0]
dt = next(vm.parameters()).dtype
print(f"threads={torch.get_num_threads()}  dtype={dt}")

x = torch.randn(1, 1024, 768, dtype=dt)

acc = defaultdict(list)
def timed(name, fn, *a):
    t0 = time.perf_counter(); r = fn(*a); acc[name].append((time.perf_counter()-t0)*1e6); return r

N = 30
for i in range(N):
    h  = timed("LayerNorm1",  layer.layer_norm1, x)
    q  = timed("q_proj (+bias)", layer.self_attn.q_proj, h)
    k  = timed("k_proj (+bias)", layer.self_attn.k_proj, h)
    v  = timed("v_proj (+bias)", layer.self_attn.v_proj, h)
    # attention core: reshape to heads + SDPA (no mask, scale 1/8)
    def core(q=q,k=k,v=v):
        B,S,_ = q.shape; H,D = 12,64
        qq = q.view(B,S,H,D).transpose(1,2); kk = k.view(B,S,H,D).transpose(1,2); vv = v.view(B,S,H,D).transpose(1,2)
        o = torch.nn.functional.scaled_dot_product_attention(qq,kk,vv, is_causal=False)
        return o.transpose(1,2).reshape(B,S,H*D)
    a  = timed("attention core", core)
    o  = timed("out_proj (+bias)", layer.self_attn.out_proj, a)
    x1 = timed("residual add 1", lambda: x + o)
    h2 = timed("LayerNorm2",  layer.layer_norm2, x1)
    f1 = timed("fc1 (+bias)", layer.mlp.fc1, h2)
    g  = timed("GELU-tanh",   layer.mlp.activation_fn, f1)
    f2 = timed("fc2 (+bias)", layer.mlp.fc2, g)
    _  = timed("residual add 2", lambda: x1 + f2)

# whole layer for cross-check
wl=[]
for _ in range(10):
    t0=time.perf_counter(); layer(x, attention_mask=None); wl.append((time.perf_counter()-t0)*1e6)

print(f"\n{'op':22s} {'CPU µs (median)':>16s} {'share':>8s}")
tot = sum(np.median(v) for v in acc.values())
for kname, v in acc.items():
    m = np.median(v); print(f"{kname:22s} {m:16.1f} {100*m/tot:7.1f}%")
print(f"{'Σ parts':22s} {tot:16.1f}")
print(f"{'whole layer (check)':22s} {np.median(wl):16.1f}")
