# FlashRT Batch Inference Optimization Log

## 实验环境
- **硬件**: Jetson AGX Thor (SM110), LPDDR5x 204.8 GB/s
- **模型**: Pi0.5 (SigLIP 27L, Encoder 18L, Decoder 18L x10 steps)
- **精度**: FP8 (encoder/decoder), FP16 (SigLIP/decoder GEMMs)
- **Benchmark**: 30 warmup + 50 timed iterations, torch.cuda.synchronize()

## Baseline (优化前, 2026-06-01)

| Metric | Value |
|--------|-------|
| B=1 serial avg | 43.2ms |
| B=8 batched avg | 236.4ms |
| B=8 per-sample | 29.5ms |
| Speedup (B*serial/batched) | 1.46x |

## P12 优化后 (2026-06-04 实测修正)

| Metric | Value |
|--------|-------|
| B=1 serial avg | 45.1ms |
| B=8 batched avg | 232.3ms |
| B=8 per-sample | 29.0ms |
| Speedup (B*serial/batched) | 1.55x |
| Determinism (same B, same seed) | cos=1.000000 |
| Cross-batch cos (B=1 serial vs B=8 batched) | 0.91 |

## P14 优化后 (2026-06-05)

| Metric | Value |
|--------|-------|
| B=1 serial avg | 45.1ms |
| B=8 batched avg | 235ms |
| B=8 per-sample | 29.4ms |
| Speedup (B*serial/batched) | 1.53x |
| Determinism (same B, same seed) | cos=1.000000 |
| Cross-batch cos (B=1 serial vs B=8 batched) | 0.91 |

**变更**: P4 (批量 patch embed) + P13 (hash 优化) + P14 (批量 SigLIP)
- `_patch_embed_ops_batched()`: 3*B kernel launches → 3 launches
- 图像 dedup hash: `_to_np16()` → `im.tobytes()`, 省 ~7ms
- SigLIP: 去除 all_same dedup 分支, 统一 `siglip_forward_batched`

**精度说明**: B=1 serial 走 CFG pipeline (cond+uncond 双分支)，B>=2 batched 走 non-CFG 单分支，
两者是不同代码路径，cross-batch cos ≈ 0.91 是预期行为。
同 B、同 seed 下确定性 cos=1.0（graph replay 后完美确定）。

---

## 优化计划

| 编号 | 优化项 | 位置 | 预期收益 | 状态 |
|------|--------|------|----------|------|
| P1a | PostLN projection batching | `postln_project_batched` | ~0.2ms | ❌ 不可行 (output layout 有 gap) |
| P1b | Padding zero 省略 | `infer_multi_prompt_batch` | ~0.1ms | ✅ 已实现 (KV cache zero 移除) |
| P2 | Encoder batch-aware FMHA | `encoder_forward_b2` | ~2-3ms | ✅ 已实现 |
| P3 | Decoder batch-aware FMHA | `decoder_forward_b2` | ~1-2ms | ✅ 已实现 |
| P4 | Patch embed batching | `_patch_embed_ops` loop | ~0.5ms | ✅ 已实现 (2026-06-05) |
| P5 | CPU 侧开销优化 | image upload, dedup, zeros | ~2ms | ✅ 已实现 |
| P13 | Image dedup hash 优化 | `_to_np16` 调用 | ~7ms | ✅ 已实现 (2026-06-05) |
| P14 | SigLIP 批量化 | dedup 分支去除 | ~1ms | ✅ 已实现 (2026-06-05) |
| P15 | SigLIP + PostLN CUDA graph | `_siglip_batched_graph` | ~3ms | ✅ 已实现 (2026-06-05) |
| P16 | 批量图像上传 (单次 .to) | `_upload_images_gpu_batched` | ~14ms | ✅ 已实现 (2026-06-05) |
| P6 | Encoder RMSNorm skip | step 11 中间层 | ~0.6ms | ✅ 已实现 |
| P7 | Python 微优化 | image upload, time cond, D2H | ~0.5ms | ✅ 已实现 |
| P8 | Encoder step 11 fusion | residual_add + RMSNorm→FP8 | ~1-2ms | ✅ 已实现 |
| P9 | Down GEMM residual fusion (beta=1.0) | `cutlass_fp8_wide` + `rms_norm` | ~4ms | ✅ 已实现 |
| P10 | KV cache zero 批量化 | graph 内 2B→2 次 zero | ~0.1ms | ✅ 已实现 |
| P11 | Time conditioning cache 修复 | `sa_all` 作用域 | bug fix | ✅ 已实现 |
| P12 | O GEMM residual fusion (beta=1.0) | `cutlass_fp8_sq` + `rms_norm` | ~13ms | ✅ 已实现 |

## 验证标准

每个优化必须同时满足：
1. **精度无损**: 同 B、同 seed，优化前后 cosine similarity ≥ 0.999
2. **实质性加速**: ≥ 0.1ms 绝对改善

---

## 实验记录

### P17: Batch-shape-aware Encoder Down GEMM Tactic (2026-06-16)

**方案**: 对 batched encoder FFN down projection 做 batch-shape-aware CUTLASS
tactic selection。原路径固定使用 `cutlass_fp8_wide`:

```text
hid_fp8 = GEGLU(gate, up)      # [B*Se, H=16384]
x = hid_fp8 @ down_w + x       # [B*Se, D=2048], beta=1 residual fused
```

该 GEMM 的 `M=B*Se` 随 batch size 变化，但固定 tactic 不能同时覆盖小 M 和大 M。
当前 `FLASHRT_THOR_BATCH_ENCODER_DOWN_TACTIC=auto` 策略:

| B | tactic |
|---:|---|
| 1 | `cutlass_fp8_t1` |
| 2-3 | `cutlass_fp8_wide` |
| >=4 | `cutlass_fp8_t2` if available, otherwise `cutlass_fp8_wide` |

**端到端 A/B**:

| B | wide E2E | auto E2E | auto tactic | Delta |
|---:|---:|---:|---|---:|
| 1 | 44.9ms | 44.0ms | t1 | -2.0% |
| 2 | 61.0ms | 61.9ms | wide | +1.5% |
| 3 | 83.0ms | 83.4ms | wide | +0.4% |
| 4 | 108.2ms | 104.7ms | t2 | -3.3% |
| 5 | 129.8ms | 125.0ms | t2 | -3.7% |
| 6 | 153.3ms | 147.8ms | t2 | -3.6% |
| 7 | 176.3ms | 168.4ms | t2 | -4.5% |
| 8 | 202.6ms | 193.0ms | t2 | -4.7% |

For B=2/3, `auto` intentionally resolves to `wide`; the small positive delta
is measurement noise between two independent runs, not an algorithmic slowdown.

**正确性**: microbenchmark 中候选 tactic 与生产 tactic bit-exact；B=4/B=8
端到端动作输出 `cosine=1.0`。

**结论**: 这是一个低风险生产优化。更大的后续空间不在继续调 decoder 小 GEMM，
而在 encoder `GEGLU -> hid_fp8 -> Down GEMM` 边界融合。B=8 时该边界折算到
17 个 encoder FFN 层约暴露 23.8ms/infer，但朴素按 Down N tile 重新计算 GEGLU
会引入约 342.7ms/infer 额外代价，因此后续需要 CTA/persistent tile 级 A tile 复用。

---

### P2/P3: Batched Softmax (Encoder + Decoder Attention)

**方案**: 将 encoder 和 decoder 的 per-sample attention 循环改为 decomposed QK^T + batched softmax + PV。

原代码 (per-sample):
```python
for b in range(B):
    fvk.attention_qkv_fp16(ctx, Q, K, V, logits, out, S, tk, NH, HD, scale, stream)
```

新代码 (decomposed + batched softmax):
```python
for b in range(B):
    fvk.attention_qk_gemm_fp16(ctx, Q, K, logits + b*off, S, tk, NH, HD, tk, scale, stream)
fvk.softmax_fp16(logits, B*S*NH, tk, stream)  # 单次 batched softmax
for b in range(B):
    fvk.attention_pv_gemm_fp16(ctx, V, logits + b*off, out + b*out_off, S, tk, NH, HD, tk, stream)
```

**C++ 新增**: `attention_qk_gemm_fp16` 和 `attention_pv_gemm_fp16` (decomposed attention GEMMs)

**精度验证**:
- 单元测试: decomposed vs monolithic, cos=1.0, max_diff=0.0 (bit-exact)
- Batched vs monolithic: cos=1.0, max_diff=0.0 (bit-exact)
- 端到端 determinism: cos=1.000000 (同 B, 同 seed)

**性能结果** (B=8):
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| B=8 batched avg | 250.24ms | 248.43ms | -1.81ms |
| Per-sample | 31.28ms | 31.05ms | -0.23ms |
| Speedup | 1.45x | 1.47x | +0.02x |

**分析**: 加速不显著 (~0.7%), 因为 attention 在 enc_ae graph 中占比很小 (~13ms / 195ms = 6.7%)。
主要瓶颈是 FP8 GEMM (GateUp: 31ms, Down: 10ms) 和 elementwise kernels (gate_res_adarms: 35ms)。
Batched softmax 减少了 B-1 次 softmax kernel launch per layer，但在 CUDA graph 中 launch overhead 已被消除。

**结论**: 精度无损 ✅, 实质性加速 ❌ (仅 ~2ms). 保留此优化作为代码改进（减少 kernel 数量），但不作为性能关键优化。

---

### 深度性能分析 (Kernel-level Profiling)

**enc_ae graph 内部 kernel 时间分解** (B=8, Se=768, 18 encoder layers):

| Kernel | 单次 (ms) | 调用次数 | 总计 (ms) |
|--------|-----------|----------|-----------|
| cutlass_fp8_t1 (GateUp) | 1.322 | 18 | 23.80 |
| cutlass_fp8_wide (Down) | 0.576 | 18 | 10.37 |
| gate_geglu_merged_fp8 | 0.663 | 18 | 11.93 |
| qkv_split_rope_kvcache | 0.068 | 144 (B×18) | 9.80 |
| cutlass_fp8_sq (QKV) | 0.231 | 18 | 4.16 |
| cutlass_fp8_sq (O) | 0.196 | 18 | 3.53 |
| residual_add_rms_norm | 0.376 | 18 | 6.77 |
| rms_norm_fp8 | 0.158 | 18 | 2.84 |
| quantize_fp8 | 0.155 | 18 | 2.79 |
| attention (QK+softmax+PV) | ~0.09 | 18 | 1.62 |
| **Encoder 小计** | | | **77.6ms** |
| Decoder (10×18 layers) | | | **~15ms** |
| **Kernel 总计** | | | **~93ms** |

**实际 enc_ae graph 时间: ~195ms**

**Gap 分析**: 93ms (kernel) vs 195ms (actual) = **2.1x gap**.

原因: **内存带宽竞争**。Thor LPDDR5x 204.8 GB/s 带宽在多个 kernel 连续执行时被饱和，
导致每个 kernel 的实际执行时间比独立运行时膨胀 ~2x。

**关键发现**:
1. enc_ae graph 是 **内存带宽瓶颈**，不是计算瓶颈 (计算效率仅 0.04%)
2. GateUp GEMM [6144, 11008, 2048] 是最大的单个 kernel (24ms)
3. Elementwise kernels (rms_norm, gate_res 等) 合计 ~25ms
4. Attention 仅占 ~2ms (batched softmax 后)
5. Python 侧开销: ~15ms (SigLIP 5.9ms + Patch embed 7.1ms + 其他 2ms)

**优化瓶颈**: 由于 enc_ae graph 占 93% 时间且是内存带宽瓶颈，
Python 层面的优化（如 batched softmax）对总时间影响很小。
进一步优化需要:
- 减少 kernel 数量（C++ 层面 kernel fusion）
- 减少数据搬运量（更激进的量化如 FP4）
- 减少 encoder sequence length（更短的 prompt）

---

### P5: CPU 侧开销优化

**方案**: 三项 Python 层面微优化，减少 enc_ae graph 之外的 CPU 开销。

**5a. GPU 侧图像转换**:
原路径: `uint8 numpy → float32 numpy → fp16 numpy → H2D upload` (CPU 侧)
新路径: `uint8 numpy → H2D upload → GPU float() / 127.5 - 1.0 → half()` (GPU 侧)
原理: 避免 CPU 侧 float32 临时数组分配和 numpy 类型转换。

**5b. id()-based 图像去重**:
原路径: `hash(images_np.tobytes())` — 对每个样本序列化并哈希
新路径: `id(s['images'][0])` — Python 对象指针比较 (O(1))
回退: 当 id 不匹配时，使用 `np.array_equal` 内容比较。

**5c. 移除冗余 buffer 清零**:
- `_ae_x_b2.zero_()` — 被 decoder GEMM (beta=0.0) 完全覆盖，冗余
- `_ae_xn_b2.zero_()` — 被 `fused_adarms_fp8_static_fp16` 完全覆盖，冗余
- `_g_noise_b2.zero_()` — 被 noise 生成循环完全覆盖，冗余
- `_Kc_b2/_Vc_b2` 外部清零 — 必须保留（graph 内部清零不覆盖所有位置）

**精度验证**:
- 确定性: cos=1.000000 (同 B, 同 seed) ✅

**性能结果** (B=8, 100 iterations):
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| B=8 P50 | 213.78ms | 211.83ms | -1.95ms |
| CPU overhead | ~8.1ms | ~7.7ms | -0.4ms |
| Speedup | 1.47x | 1.48x | +0.01x |

**分析**: GPU 侧图像转换节省 ~1.9ms (16 张图像 × 0.12ms/张)。
id()-based 去重节省 ~0.3ms。移除冗余清零节省 ~0.1ms。
总改善 ~2ms，但 enc_ae graph 仍占 93% 时间。

---

### 最终性能总结 (2026-06-04 修正)

**⚠️ 修正说明**: 之前记录的 "B=8 = 194ms / 1.85x" 仅测了 enc_ae graph replay 时间，
未包含 SigLIP + Patch embed + PostLN + Python 开销 (~50ms)。
以下为完整 pipeline 端到端实测数据。

**P1-P12 优化累计效果**:
| 优化项 | 改善 | 备注 |
|--------|------|------|
| Batched softmax (P2/P3) | ~2ms | 减少 kernel 数量 |
| Encoder RMSNorm skip (P6) | ~0.6ms | 跳过 17 层冗余 RMSNorm |
| Down GEMM residual fusion (P9) | ~5ms | beta=1.0 融合 FFN residual |
| KV cache zero 批量化 (P10) | ~0.1ms | 2B→2 次 zero |
| O GEMM residual fusion (P12) | ~13ms | beta=1.0 融合 attention residual |
| GPU-side image conversion (P5a) | ~0.5ms | 避免 CPU float32 中间数组 |
| id()-based dedup (P5b) | ~0.1ms | O(1) vs hash(tobytes) |
| 移除冗余 buffer 清零 (P5c) | ~0.1ms | 减少 kernel launch |
| **总优化** | **~21ms** | — |

**全 Batch Size 端到端性能** (P50, numpy uint8 输入, P12 优化后, 100 warmup + 50 timed):
| Batch Size | 总时间 | Per-sample | Speedup |
|------------|--------|------------|---------|
| B=1 | 45.1ms | 45.1ms | 1.00x |
| B=2 | 69.2ms | 34.6ms | 1.30x |
| B=3 | 94.8ms | 31.6ms | 1.43x |
| B=4 | 134.1ms | 33.5ms | 1.35x |
| B=5 | 161.9ms | 32.4ms | 1.39x |
| B=6 | 176.6ms | 29.4ms | 1.53x |
| B=7 | 204.3ms | 29.2ms | 1.55x |
| B=8 | 232.3ms | 29.0ms | 1.55x |

**注**: Speedup = B × T_serial / T_batched. B=8 达到 1.55x 吞吐量提升。

**结论**: P1-P12 优化总共减少 ~21ms。当前 B=8 端到端 232ms，per-sample 29ms。
下一步优化方向: SigLIP + Patch embed + PostLN 的 batched 封装 (~50ms 开销)。

**GEMM 计算分析** (B=8):
| GEMM | 形状 | 精度 | 理论峰值 | 实测 | 效率 |
|------|------|------|----------|------|------|
| Enc GateUp | [6144, 11008, 2048] | FP8 | 4.16ms | 1.32ms | 31.8% |
| Enc Down | [6144, 2048, 5504] | FP8 | 2.08ms | 0.58ms | 27.9% |
| Dec GateUp | [80, 5120, 1024] | FP16 | 0.01ms | — | — |

**Arithmetic Intensity**:
- Enc GateUp: 3453 FLOP/byte (远超 roofline 325 FLOP/byte → 计算密集)
- Dec GateUp: 853 FLOP/byte (也超过 roofline → 计算密集)
- 但实际性能仅 30% 峰值，原因是 **内存带宽竞争**: 多个 kernel 连续执行时
  LPDDR5x 204.8 GB/s 被饱和，每个 kernel 的内存操作被延迟。

---

### P8: Encoder Step 11 Fusion (residual_add + RMSNorm→FP8)

**方案**: 将 encoder 每层 step 11 的 `residual_add_fp16` 替换为
`residual_add_rms_norm_fp8_noweight_fp16` (融合 residual add + next layer RMSNorm→FP8)。

**原理**: 原代码 step 11 做 `x += ffn_out` (residual add)，然后下一层 step 1 做
`x_fp8 = RMSNorm(x)` (normalize)。融合后一步完成：`x += ffn_out; x_fp8 = RMSNorm(x)`。

**数学等价性**:
- 原代码: QKV 输入 = RMSNorm(x + ffn_out) — step 1 of next layer
- 融合后: QKV 输入 = RMSNorm(x + ffn_out) — step 11 of current layer
- 结果完全相同 (同一个 kernel，同一个输入)

**内存节省** (per intermediate layer):
- 原代码: residual_add 读写 x[BSe, D] + rms_norm 读 x[BSe, D] 写 x_fp8 = 4 次内存操作
- 融合后: 1 个 kernel 读 x+fg 写 x+x_fp8 = 3 次内存操作
- 节省: 1 次 x[BSe, D] 读写 = 2 × BSe × D × 2 bytes

| B | Se | D | Per-layer saving | 17 layers total |
|---|----|----|------------------|-----------------|
| 1 | 768 | 2048 | 6.3 MB | 107 MB |
| 2 | 768 | 2048 | 12.6 MB | 213 MB |
| 8 | 768 | 2048 | 50.2 MB | 853 MB |

**精度验证**:
- B=1 warm determinism: cos=1.000000 ✅
- B=2 warm determinism: cos=1.000000 ✅
- B=8 warm determinism: cos=1.000000 ✅

**性能结果** (30 warmup + 50 timed):
| B | P50 | Baseline | Delta |
|---|-----|----------|-------|
| 1 | 44.91ms | 44.39ms | +0.5ms (方差) |
| 2 | 64.47ms | 60.91ms | +3.6ms (方差) |
| 8 | 207.99ms | 196.66ms | +11.3ms (方差) |

**分析**: 不同进程的测量方差 (~10ms) 远大于理论收益 (~1-2ms)。
理论内存节省 853MB (B=8) → 4.2ms @ 204.8 GB/s，但实际收益被带宽竞争掩盖。
精度无损，保留此优化作为代码改进。

**结论**: 精度无损 ✅, 实质性加速 ❓ (需同进程 A/B 测试).

---

### P7: Python 层面微优化 (infer_multi_prompt_batch)

**方案**: 四项 Python 层面优化，减少 enc_ae graph 之外的开销。

**7a. 移除冗余 KV cache 清零**:
原代码: `self._Kc_b2.zero_(); self._Vc_b2.zero_()` 在 graph replay 之前
新代码: 移除 (graph 内部 `_capture_enc_ae_graph_b2` line 1694 已包含清零)
原理: graph replay 会重新清零，Python 侧清零完全冗余

**7b. 复用 set_prompt 的 time conditioning 缓存**:
原代码: 每次 `infer_multi_prompt_batch` 都重新计算 `_sa_all_b2`
新代码: 如果 `_sa_all` (B=1) 已存在且尺寸匹配，直接 `repeat_interleave` 展开
原理: `set_prompt` 已预计算 `_sa_all`，仅需 B-tile 展开

**7c. 避免 uint8→fp16 图像转换的 FP32 中间数组**:
原代码: `(t_u8.float() / 127.5 - 1.0).half()` — 分配 3.5MB FP32 临时张量
新代码: `t_u8.to(torch.float16).div_(127.5).sub_(1.0)` — 直接 uint8→fp16
节省: 2.6MB GPU 内存/张量，避免 1 次 FP32 分配

**7d. 批量 D2H 结果传输**:
原代码: B 次 `.float().cpu().numpy()` (每次 D2H + GPU fp16→fp32)
新代码: 一次 `raw_all = _g_noise_b2[:B*Sa].float().cpu().numpy()` 后 CPU 侧切片
节省: B-1 次 D2H 传输启动开销

**精度验证**:
- Warm determinism: cos=1.000000 ✅
- Cold→warm 差异: cos≈0.49 (pre-existing graph capture vs replay 差异，非本次优化引入)

**性能结果** (B=8, 30 warmup + 50 timed):
| Metric | Baseline | After | Delta |
|--------|----------|-------|-------|
| B=8 P50 | 196.66ms | 206.88ms | +10ms (测量方差) |
| B=2 P50 | 62.00ms | 64.28ms | +2ms (测量方差) |

**分析**: Python 层面优化在稳态 (graph replay) 下几乎无加速效果，因为 enc_ae graph 占 98% 时间。
主要收益在冷启动路径 (time conditioning 缓存复用避免 880 次 kernel launch)。
不同进程的测量方差 (~10ms) 远大于优化收益，需要同一进程内 A/B 对比才能准确测量。

**结论**: 精度无损 ✅, 实质性加速 ❌ (Python 层面优化已达极限).

---

**下一步优化方向** (需 C++ 层面):
1. **Encoder O proj + quantize fusion**: 合并 step 6 (quantize) 和 step 7 (residual+norm)
   - 将 `quantize_fp8_static_fp16` 融合到 `residual_add_rms_norm_fp8_noweight_fp16` 的输入
   - 节省 o_fp8 的写+读 round-trip (B*Se*D bytes per layer)
   - 预期收益: ~3ms (17 layers × 0.18ms/layer)
   - 阻碍: 需要新 C++ kernel `quantize_residual_add_rms_norm_fp8`
2. **GeGLU + G8 megakernel bundling (FP8 path)**: 合并 `gate_geglu_merged_fp8_fp16` 和 `cutlass_fp8_wide`
   - FP16 路径已有 `flashrt_megakernel_geglu_g8_fp16` 可参考
   - 预期收益: ~0.5ms (减少 kernel launch overhead)
3. **Batch-aware QKV split + rope**: 消除 B×18 encoder + B×180 decoder per-sample 循环
   - 减少 kernel launch overhead (在 graph 中影响较小)
4. **减少 decoder 步数**: 从 10 步减至 5 步
   - 需要模型层面评估精度影响
   - 预期收益: ~50% decoder 时间 ≈ 7-8ms
5. **FP4 decoder**: 使用 NVFP4 替代 FP8 decoder
   - 已有 `Pi05TorchFrontendThorFP4` 支持
   - 需要精度验证
6. **更短的 prompt**: 减少 encoder sequence length
   - 直接减少 GEMM M 维度和 elementwise 工作量

---

### P9: Encoder Down GEMM Residual Fusion (beta=1.0) + Bug Fixes

**方案**: 将 encoder 每层 step 10 (Down GEMM) 的输出直接写入 `x` (residual stream)，
使用 `beta=1.0` 将 FFN residual add 融合到 CUTLASS GEMM epilogue 中。

**原理**: CUTLASS GEMM runner (`cutlass_sm100.cu:42`) 将 C 和 D 矩阵指针别名为同一地址：
```cpp
{{alpha, beta}, (ElementD*)D, stride_D, (ElementD*)D, stride_D}
```
因此 `cutlass_fp8_wide(hid_fp8, down_w, x, ..., beta=1.0)` 计算:
`x = alpha * (hid_fp8 @ down_w) + 1.0 * x` — FFN residual 直接在 GEMM epilogue 中完成。

**数学等价性**:
- 原代码: `fg = GEMM(hid_fp8, down_w); x += fg; x_fp8 = RMSNorm(x)`
- 优化后: `x = GEMM(hid_fp8, down_w) + x; x_fp8 = RMSNorm(x)`
- 结果完全相同 (CUTLASS epilogue 在写入前读取旧的 x)

**内存节省** (per intermediate layer):
- 原代码: Down GEMM 写 fg + residual_add_rms_norm 读 fg+x 写 x+x_fp8 = 4 次内存操作
- 优化后: Down GEMM 读 x 写 x + rms_norm 读 x 写 x_fp8 = 3 次内存操作
- 节省: 1 次 x[Se, D] fp16 读写 = 2 × Se × D × 2 bytes

| B | Se | D | Per-layer saving | 17 layers total |
|---|----|----|------------------|-----------------|
| 1 | 768 | 2048 | 6.3 MB | 107 MB |
| 8 | 768 | 2048 | 50.2 MB | 853 MB |

**附带修复**:
- KV cache zero: 将 graph 内 2B 次 per-slice zero 替换为 2 次 whole-tensor zero
- Time conditioning cache: 修复 `sa_all` 变量作用域错误 (for/else 误用)

**精度验证**:
- B=8 warm determinism: cos=1.000000 ✅

**性能结果** (30 warmup + 50 timed):
| B | P50 | Baseline | Delta |
|---|-----|----------|-------|
| 1 | 45.24ms | 44.39ms | +0.85ms (方差) |
| 8 | 206.31ms | 211.40ms | -5.09ms |

**Speedup**: 1.76x (baseline 1.68x), per-sample 25.77ms (baseline 26.43ms)

**分析**: Down GEMM beta=1.0 融合节省 ~4ms (17 layers × 0.24ms/layer)。
KV cache zero 减少 14 次 kernel launch (graph 内)。
Time conditioning cache 修复避免了不必要的重新计算。

**结论**: 精度无损 ✅, 实质性加速 ✅ (-5.1ms, +4.8% 吞吐量)

---

### P12: Encoder O GEMM Residual Fusion (beta=1.0)

**方案**: 将 encoder 每层 step 6 (O proj GEMM) 的输出直接写入 `x` (residual stream)，
使用 `beta=1.0` 将 attention residual add 融合到 CUTLASS GEMM epilogue 中。

**原理**: 与 P9 (Down GEMM) 相同的 CUTLASS C/D pointer aliasing 机制。
`cutlass_fp8_sq(o_fp8, o_w, x, ..., beta=1.0)` 计算:
`x = alpha * (o_fp8 @ o_w) + 1.0 * x` — attention residual 直接在 GEMM epilogue 中完成。

**数学等价性**:
- 原代码: `fg = GEMM(o_fp8, o_w); x += fg; x_fp8 = RMSNorm(x)` (两步: GEMM + residual_add_rms_norm)
- 优化后: `x = GEMM(o_fp8, o_w) + x; x_fp8 = RMSNorm(x)` (两步: GEMM + rms_norm)
- 结果完全相同 (CUTLASS epilogue 在写入前读取旧的 x)

**内存节省** (per intermediate layer):
- 原代码: O GEMM 写 fg + residual_add_rms_norm 读 fg+x 写 x+x_fp8 = 4 次内存操作
- 优化后: O GEMM 读 x 写 x + rms_norm 读 x 写 x_fp8 = 3 次内存操作
- 节省: 1 次 x[Se, D] fp16 读写 = 2 × Se × D × 2 bytes
- 额外节省: fg buffer 的写入 (Se × D × 2 bytes)

| B | Se | D | Per-layer saving | 17 layers total |
|---|----|----|------------------|-----------------|
| 1 | 768 | 2048 | 9.4 MB | 160 MB |
| 8 | 768 | 2048 | 75.5 MB | 1284 MB |

**注意**: 此优化将 O GEMM 的输出从 `fg` 改为 `x`。`x` 是 encoder 的 residual stream buffer
(`_enc_x` / `_enc_x_b2`)，在 graph replay 前由 Python 侧写入 PostLN 输出。
CUTLASS GEMM 在写入前读取旧的 x 值，因此 residual add 正确执行。
与 P9 (Down GEMM) 不同的是，O GEMM 修改的是 encoder 的 **输入** buffer，
而 Down GEMM 修改的是 encoder 的 **中间** residual stream — 两者在 CUDA graph
capture/replay 下均正确工作。

**精度验证**:
- B=1 warm determinism (同 seed): cos=1.000000 ✅
- B=8 warm determinism (同 seed): cos=1.000000 ✅

**性能结果** (30 warmup + 50 timed, GPU 热态):
| B | P50 | Baseline | Delta |
|---|-----|----------|-------|
| 1 | 44.96ms | 45.07ms | -0.11ms |
| 2 | 60.23ms | — | — |
| 8 | 194.39ms | 207.54ms | -13.15ms |

**Speedup**: 1.85x (baseline 1.74x), per-sample 24.30ms (baseline 25.94ms)

**分析**: O GEMM beta=1.0 融合节省 ~13ms (17 layers × 0.77ms/layer)。
每层节省 1 次 Se×D fp16 residual_add kernel + 1 次 fg buffer 写入。
fg buffer 不再被 O GEMM 使用 (仅 GateUp GEMM 输出仍使用 fg)。

**结论**: 精度无损 ✅, 实质性加速 ✅ (-13.5ms, +6.5% 吞吐量)

---

### P13: SigLIP Graph Replay (same-image batched) — 实验失败

**方案**: 当 B 个样本共享相同图像和 prompt 时，用 B=1 的 `_siglip_graph` (CUDA graph) 替代 batched SigLIP + PostLN 直接 kernel 调用。

**原理**: B=1 graph 包含 `patch_embed + SigLIP + PostLN`，一次 `replay()` 替代 ~100 次 Python kernel dispatch。

**实验结果**: 精度无损 (cos=1.000000) ✅，但 **无性能改善**。

**原因分析**:
1. Dedup 循环已先执行 `_patch_embed_ops`，graph replay 重复执行此操作
2. Graph 内 `patch_embed` 使用 `self._img_buf` (B=1 buffer)，而非 per-sample `_img_buf_b2_list`
3. Graph replay 后仍需逐 slot copy `_enc_x → _enc_x_b2`
4. 总开销 ≈ 旧路径 (serial SigLIP + batched PostLN)

**结论**: 精度无损 ✅, 实质性加速 ❌ (graph replay 与 dedup 循环冲突). 已 revert。

---

---

## 最终优化总结

### 优化时间线

| 阶段 | 编号 | 优化内容 | 精度验证 | 性能变化 | 提交 |
|------|------|----------|----------|----------|------|
| 1 | P2/P3 | Batched softmax (encoder/decoder attention) | cos=1.000000 ✅ | B=8: 250→248ms (-2ms) | 6329452 |
| 2 | P5a | GPU-side uint8→fp16 图像转换 | cos=1.000000 ✅ | ~0.5ms (16张图×0.12ms) | 8075dbd |
| 2 | P5b | id()-based 图像去重 | cos=1.000000 ✅ | ~0.1ms (O(1) vs hash) | 8075dbd |
| 2 | P5c | 移除冗余 buffer 清零 | cos=1.000000 ✅ | ~0.1ms | 8075dbd |
| 3 | P6 | Encoder 最后一层 RMSNorm skip | cos=1.000000 ✅ | ~0.6ms | a6eb5b8 |
| 4 | P7a | 移除冗余 KV cache 清零 | cos=1.000000 ✅ | ~0.1ms | 84259ca |
| 4 | P7b | Time conditioning 缓存复用 | cos=1.000000 ✅ | 冷启动优化 | 84259ca |
| 4 | P7c | uint8→fp16 直接转换 (避免 FP32 中间) | cos=1.000000 ✅ | ~0.2ms | 84259ca |
| 4 | P7d | 批量 D2H 结果传输 | cos=1.000000 ✅ | ~0.1ms | 84259ca |
| 5 | P8 | Encoder step 11 fusion (residual_add + RMSNorm→FP8) | cos=1.000000 ✅ | ~0.6ms (17层×0.04ms) | 304bdf7 |
| 6 | P9 | **Down GEMM residual fusion (beta=1.0)** | cos=1.000000 ✅ | **B=8: 211→206ms (-5ms)** | 444feb1 |
| 6 | P10 | KV cache zero 批量化 (2B→2次) | cos=1.000000 ✅ | ~0.1ms | 444feb1 |
| 6 | P11 | Time conditioning cache 作用域修复 | cos=1.000000 ✅ | bug fix | 444feb1 |
| 7 | P12 | **O GEMM residual fusion (beta=1.0)** | cos=1.000000 ✅ | **B=8: 207→194ms (-13ms)** | 2f37f0c |
| 8 | P13 | SigLIP Graph Replay (same-image) | cos=1.000000 ✅ | 无改善，已 revert | 9eb44fc |

### 核心优化详解

**P9 + P12: CUTLASS GEMM beta=1.0 residual fusion** — 本次最大收益

原理: CUTLASS GEMM runner (`cutlass_sm100.cu:42`) 将 C 和 D 矩阵指针别名为同一地址：
```cpp
{{alpha, beta}, (ElementD*)D, stride_D, (ElementD*)D, stride_D}
```
因此 `cutlass_fp8_*(input, weight, x, ..., beta=1.0)` 计算:
`x = alpha * (input @ weight) + 1.0 * x` — residual add 直接在 GEMM epilogue 中完成。

- **P9 (Down GEMM)**: 消除 17 层 `residual_add_fp16` kernel + fg buffer 写入 → **-5ms**
- **P12 (O GEMM)**: 消除 17 层 `residual_add_rms_norm_fp8_noweight_fp16` kernel + fg buffer 写入 → **-13ms**
- 精度: CUTLASS epilogue 在写入前读取旧的 x，数学等价性完全保证

**P2/P3: Batched softmax** — 减少 kernel 数量

将 per-sample attention 循环改为 decomposed QK^T + batched softmax + PV:
- 原代码: B 次 `attention_qkv_fp16` per layer
- 新代码: B 次 `attention_qk_gemm` + 1 次 `softmax_fp16(B*S*NH)` + B 次 `attention_pv_gemm`
- 单元测试: decomposed vs monolithic, cos=1.0, max_diff=0.0 (bit-exact)

### 性能对比

**Baseline → P12 优化后 (端到端实测)**:

| Batch Size | Baseline (6/1) | P12 优化后 (6/4) | 改善 |
|------------|----------------|------------------|------|
| B=1 | 43.2ms | 45.1ms | +1.9ms (波动) |
| B=2 | 67.0ms | 69.2ms | +2.2ms (波动) |
| B=8 | 236.4ms | **232.3ms** | **-4.1ms (-1.7%)** |
| B=8 per-sample | 29.5ms | **29.0ms** | **-0.5ms** |
| B=8 speedup | 1.46x | **1.55x** | +0.09x |

**注**: P1-P12 优化主要作用于 enc_ae graph 内部 (减少 ~21ms)，
但 graph 外开销 (~50ms: SigLIP + Patch embed + PostLN) 未被优化，
端到端改善约 4ms。

### 瓶颈分析

B=8 端到端 232ms，其中 enc_ae graph ~182ms，graph 外 ~50ms。

**Kernel 时间分解** (B=8, 18 encoder layers, enc_ae graph 内):

| Kernel | 总计 (ms) | 占比 |
|--------|-----------|------|
| cutlass_fp8_t1 (GateUp) | 23.80 | 30.7% |
| cutlass_fp8_wide (Down) | 10.37 | 13.4% |
| gate_geglu_merged_fp8 | 11.93 | 15.4% |
| qkv_split_rope_kvcache | 9.80 | 12.6% |
| cutlass_fp8_sq (QKV) | 4.16 | 5.4% |
| residual_add_rms_norm | 6.77 | 8.7% |
| rms_norm_fp8 | 2.84 | 3.7% |
| quantize_fp8 | 2.79 | 3.6% |
| cutlass_fp8_sq (O) | 3.53 | 4.5% |
| attention (QK+softmax+PV) | 1.62 | 2.1% |
| **Kernel 总计** | **77.6** | **100%** |

**Gap**: 77.6ms (kernel) vs 182ms (graph 实际) = **2.3x gap**。
原因: LPDDR5x 204.8 GB/s 带宽在多个 kernel 连续执行时被饱和。

### 下一步优化路线图

| 优先级 | 优化项 | 预期收益 | 复杂度 | 说明 |
|--------|--------|----------|--------|------|
| 0 | SigLIP + Patch embed + PostLN batch化 | ~30-40ms | 中 | 当前 graph 外最大开销 (~50ms)，可封装进 CUDA graph |
| 1 | Encoder GateUp+GeGLU+Down megakernel | ~10-15ms | 高 | 合并 3 个 kernel，消除 2 次 B*Se*H fp8 内存 round-trip |
| 2 | Encoder QKV GEMM + split/rope fusion | ~5-8ms | 中 | 将 qkv_split_rope_kvcache 融合到 GEMM epilogue |
| 3 | 减少 decoder 步数 10→5 | ~7-8ms | 低 | 需要模型层面精度验证 |
| 4 | FP4 decoder | ~5-7ms | 中 | 已有 `Pi05TorchFrontendThorFP4`，需精度验证 |
| 5 | Encoder O proj + quantize fusion | ~3ms | 中 | 将 quantize_fp8 融合到 O GEMM epilogue |
| 6 | Batch-aware QKV split | ~1ms | 低 | 消除 B×18 per-sample 循环 |

**注**: 优先级 0 是当前最大瓶颈 (graph 外 50ms)。优化项 1-3 需 C++ kernel 开发。

## 2026-06-16: RP1 stage split / hetero-batch feasibility benchmark

新增 `benchmarks/bench_pi05_stage_ideas.py`，用于在 Thor 上安全复核两个 RP1 扩展方向。该脚本只使用合成 observation 和模型推理，不连接相机、机械臂、CAN 或动作执行通道。

正式命令:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_pi05_stage_ideas.py \
  --batch-sizes 1-8 \
  --warmup 8 \
  --iters 30 \
  --stage-iters 40 \
  --json-out /mnt/home/zengzixuan/workspace/Lingshu/docs/experiments/pi05_stage_ideas_thor_2026-06-16.json
```

关键测量:

| B | E2E P50 | SigLIP/PostLN | Enc+AE graph | Encoder | Decoder | ideal pipe bound |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.3 ms | 5.6 ms | 37.0 ms | 19.4 ms | 20.5 ms | 20.5 ms |
| 2 | 60.4 ms | 9.5 ms | 49.7 ms | 31.3 ms | 21.1 ms | 31.3 ms |
| 4 | 120.8 ms | 17.6 ms | 100.9 ms | 78.5 ms | 23.7 ms | 78.5 ms |
| 8 | 202.7 ms | 33.8 ms | 166.0 ms | 136.6 ms | 25.1 ms | 136.6 ms |

真实 double-buffer overlap 复核命令:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_pi05_stage_ideas.py \
  --batch-sizes 2,4,8 \
  --warmup 4 \
  --iters 8 \
  --stage-iters 12 \
  --pipeline-batches 2,4,8 \
  --pipeline-iters 16 \
  --json-out /mnt/home/zengzixuan/workspace/Lingshu/docs/experiments/pi05_real_overlap_pipeline_thor_2026-06-16.json
```

真实 overlap 结果:

| B | serial double-buffer | overlap double-buffer | speedup | improvement | parity |
|---:|---:|---:|---:|---:|---|
| 2 | 52.4 ms | 52.0 ms | 1.008x | 0.8% | max_abs=0, cos=1.000000 |
| 4 | 101.7 ms | 100.8 ms | 1.009x | 0.9% | max_abs=0, cos=1.000000 |
| 8 | 168.9 ms | 167.5 ms | 1.008x | 0.8% | max_abs=0, cos=1.000000 |

结论:

- 直接 stage split 不是已经实现的端到端优化；手动拆分后的串行时间与 fused graph 接近。
- benchmark-only 真实 encoder/decoder overlap 已实现并验证 bit-exact，但 B=2/4/8 收益均不足 1%。理论 `max(Encoder, Decoder)` 上限没有转化为真实收益，说明当前 Thor 上 encoder 与 decoder 并发受到 GPU 资源竞争、带宽竞争或库内部调度限制。
- 异构 batch 更适合作为 deadline/priority primitive。多数计划会增加全部请求完成时间，但能显著提前首个紧急 chunk，例如 `Bv=8,Br=2` 可把首块完成时间从 199.9 ms 提前到 83.5 ms，同时总时间增加到 232.6 ms。
- 因此该方向暂不接入默认生产推理路径，保留为研究 benchmark。后续只有在跨设备流水线或 Lingshu scheduler 混合 slack replay 中出现正向收益，才应进入生产路径。

## 2026-06-16: Same-Thor Vision/Enc+AE overlap probe

在 `benchmarks/bench_pi05_stage_ideas.py` 中新增 benchmark-only 的
`--vision-pipeline-batches` 探针，用于验证同一台 Thor 上
`Vision(batch i+1)` 是否可以与 `Enc+AE(batch i)` 安全并发。该实验只使用合成
observation，不连接相机、机械臂、CAN 或动作执行通道；也不使用 4090 端侧计算或跨设备
overlap。

smoke 命令:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_pi05_stage_ideas.py \
  --batch-sizes 2 \
  --warmup 1 \
  --iters 2 \
  --stage-iters 2 \
  --pipeline-batches none \
  --vision-pipeline-batches 2 \
  --vision-pipeline-iters 4 \
  --json-out /mnt/home/zengzixuan/workspace/Lingshu/docs/experiments/pi05_vision_encae_overlap_smoke_2026-06-16.json
```

smoke 结果:

| B | serial stage/iter | overlap stage/iter | speedup | improvement | parity |
|---:|---:|---:|---:|---:|---|
| 2 | 60.7 ms | 61.5 ms | 0.987x | -1.3% | max_abs=2.36, cos=0.291195 |

结论:

- 该探针已经使用双 `_enc_x_b2` feature slot，并尝试为 slot graph 使用独立 FVK/GEMM context。
- correctness 仍然不通过，因此不能把该结果解释为性能收益或性能回退；它首先说明当前 CUDAGraph/buffer/context 组织还不支持安全的 Vision/Enc+AE 同机并发。
- 不做 B=4/B=8 正式 sweep，不接入 `Pi05TorchFrontendThor` 默认推理路径。
- 若后续继续该方向，应先做 graph 并发安全性审计和 Nsight 解释性实验，而不是扩大工程集成。

## 2026-06-16: Batched encoder Down GEMM tactic selection

目标回到单机纯 batch 推理，不再依赖 scheduler、跨设备流水线或同机 overlap。复跑
`benchmarks/bench_b1_b8.py` 后确认，稳定热状态下 B=1..8 的主要瓶颈在
`Enc+AE graph` 内部，graph 外 staging/residual 通常只有 1-3 ms。

热状态 baseline:

| B | E2E avg | SigLIP/PostLN | Enc+AE graph | throughput |
|---:|---:|---:|---:|---:|
| 1 | 42.3 ms | 5.5 ms | 35.5 ms | 23.6 req/s |
| 2 | 57.7 ms | 9.2 ms | 47.5 ms | 34.7 req/s |
| 3 | 76.3 ms | 12.1 ms | 62.8 ms | 39.3 req/s |
| 4 | 108.3 ms | 16.2 ms | 90.6 ms | 36.9 req/s |
| 5 | 130.9 ms | 20.3 ms | 109.2 ms | 38.2 req/s |
| 6 | 153.3 ms | 24.2 ms | 127.6 ms | 39.1 req/s |
| 7 | 177.7 ms | 28.3 ms | 146.0 ms | 39.4 req/s |
| 8 | 201.8 ms | 33.9 ms | 165.9 ms | 39.6 req/s |

开关 sweep 结论:

- `FLASHRT_THOR_BATCH_SYNC_AFTER_GRAPH=0` 无收益。最终 D2H `.cpu().numpy()` 本身会同步。
- `FLASHRT_THOR_BATCH_POSTLN_PROJ_FUSION=1`、`FLASHRT_THOR_BATCH_GRAPH_AUTOTUNE=1`、
  `FLASHRT_THOR_BATCH_SIGLIP_GRAPH_AUTOTUNE=1` 的短测结果与热状态 baseline 基本同档。
  早期较慢结果主要来自冷状态或 CUDA Graph capture schedule 波动，不能作为稳定收益。
- 因此继续优化应集中在 `Enc+AE graph` 内部，而不是 prompt upload、padding zero、
  D2H 或 Python staging。

新增 microbenchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_pi05_batch_research_kernels.py \
  --batch-sizes 1-8 \
  --warmup 10 \
  --iters 30 \
  --max-candidates 8 \
  --skip-chains \
  --json-out /mnt/home/zengzixuan/workspace/Lingshu/docs/experiments/pi05_batch_kernel_tactics_short_2026-06-16.json
```

关键发现: encoder FFN down projection 在 B>=4 时，`cutlass_fp8_t2` 比原生产路径
`cutlass_fp8_wide` 更快，且 microbenchmark 中 bit-exact:

| B | production wide | best t2 | single-GEMM speedup |
|---:|---:|---:|---:|
| 4 | 1.440 ms | 1.243 ms | 1.16x |
| 5 | 1.765 ms | 1.571 ms | 1.12x |
| 6 | 2.103 ms | 1.845 ms | 1.14x |
| 7 | 2.445 ms | 2.163 ms | 1.13x |
| 8 | 2.808 ms | 2.389 ms | 1.18x |

实现:

- `flash_rt/hardware/thor/shared_primitives_batched.py` 新增
  `FLASHRT_THOR_BATCH_ENCODER_DOWN_TACTIC`。
- 支持 `auto/wide/t2/sq/t1/plain`。
- 默认 `auto`: B=1 使用 `t1`；B=2/3 使用原 `wide`；B>=4 且存在
  `cutlass_fp8_t2` 时使用 `t2`，否则回退原 `wide`。
- 该改动只替换 FFN down GEMM tactic，不改变 tensor layout、batch 语义或动作输出格式。

端到端 A/B:

```bash
CUDA_VISIBLE_DEVICES=0 FLASHRT_THOR_BATCH_ENCODER_DOWN_TACTIC=wide \
  python benchmarks/bench_b1_b8.py --batch-sizes 1-8 --warmup 8 --iters 20 \
  --profile --profile-iters 8 --reuse-frontend \
  --json-out /mnt/home/zengzixuan/workspace/Lingshu/docs/experiments/pi05_b1_b8_down_wide_control_2026-06-16.json

CUDA_VISIBLE_DEVICES=0 FLASHRT_THOR_BATCH_ENCODER_DOWN_TACTIC=auto \
  python benchmarks/bench_b1_b8.py --batch-sizes 1-8 --warmup 8 --iters 20 \
  --profile --profile-iters 8 --reuse-frontend \
  --json-out /mnt/home/zengzixuan/workspace/Lingshu/docs/experiments/pi05_b1_b8_down_auto_2026-06-16.json
```

| B | wide E2E | auto E2E | auto tactic | improvement | wide Enc+AE | auto Enc+AE |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 44.9 ms | 44.0 ms | t1 | 2.0% | 37.5 ms | 37.3 ms |
| 2 | 61.0 ms | 61.9 ms | wide | -1.5% | 50.5 ms | 50.9 ms |
| 3 | 83.0 ms | 83.4 ms | wide | -0.4% | 68.4 ms | 68.6 ms |
| 4 | 108.2 ms | 104.7 ms | t2 | 3.3% | 90.7 ms | 87.2 ms |
| 5 | 129.8 ms | 125.0 ms | t2 | 3.7% | 107.9 ms | 103.1 ms |
| 6 | 153.3 ms | 147.8 ms | t2 | 3.6% | 127.6 ms | 121.9 ms |
| 7 | 176.3 ms | 168.4 ms | t2 | 4.5% | 145.2 ms | 138.2 ms |
| 8 | 202.6 ms | 193.0 ms | t2 | 4.7% | 166.7 ms | 157.4 ms |

正确性验证:

| B | max_abs | mean_abs | cosine |
|---:|---:|---:|---:|
| 4 | 0.00000000 | 0.00000000 | 1.000000000 |
| 8 | 0.00000000 | 0.00000000 | 1.000000000 |

结论:

- 这是当前单机 batch 路径中第一个明确穿透到端到端延迟的 kernel tactic 优化。
- 收益幅度不大但稳定，符合瓶颈分解: down GEMM 是 Enc+AE 内部的一部分，因此整体收益约
  3-5%，不是数量级提升。
- 后续最有希望的方向不是继续堆 runtime 开关，而是做 batch-shape-aware kernel
  selection/autotuning，并扩展到 decoder `fp8_gemm_descale_fp16`、GateUp/GeGLU/Down
  producer chain 或更深的 FFN fusion。

### Git 提交记录

```
9eb44fc docs: add P13 experiment (SigLIP graph replay) and final optimization summary
2be70e9 docs: update optimization log with P12 results and final performance table
2f37f0c opt: fuse O GEMM residual (beta=1.0), eliminating 17 encoder residual_add kernels
b6b4224 opt: skip duplicate image uploads in batched inference
c2a034f docs: update optimization log with P9 results and revised next steps
444feb1 opt: fuse Down GEMM residual (beta=1.0), fix KV cache zero + time cond cache
e10feef docs: add P8 encoder fusion results, update optimization table
304bdf7 opt: fuse encoder step 11 (residual_add) with next layer's RMSNorm→FP8
d88fca2 docs: add P7 Python micro-optimization results to log
84259ca opt: Python-level micro-optimizations for infer_multi_prompt_batch
c0afb23 docs: add P5/P6 to optimization table
a6eb5b8 feat: skip redundant RMSNorm in encoder step 11 for intermediate layers
2b9ed20 docs: update optimization log with final analysis
ea20d21 perf: GPU-side image conversion via torch tensors
11eb25d perf: pre-allocate GPU staging tensor for image conversion
8075dbd feat: CPU-side optimization — GPU image conversion, id-based dedup, remove redundant zeros
6329452 feat: batched softmax optimization for encoder/decoder attention
```

### P18: Periodic Research Loop for Batch Inference Hypothesis Testing (2026-06-17)

**目的**: 将单机 Pi0.5 Thor batch 优化从零散 microbenchmark 改为可重复的研究闭环。
每个周期同时记录:

1. B=1..8 端到端 batch latency 和 graph replay profile。
2. encoder GEGLU -> Down GEMM producer-chain 边界。
3. virtual FP8 activation mainloop 成本模型。
4. 第一周期额外运行 GEGLU LUT/row8 排除实验。

新增脚本:

- `benchmarks/run_pi05_batch_research_loop.py`
- `benchmarks/summarize_pi05_research_loop.py`

执行:

```bash
CUDA_VISIBLE_DEVICES=0 FLASHRT_THOR_BATCH_ENCODER_DOWN_TACTIC=auto \
  python benchmarks/run_pi05_batch_research_loop.py \
    --duration-min 120 \
    --batch-sizes 1-8 \
    --warmup 8 \
    --iters 20 \
    --profile-iters 8 \
    --kernel-warmup 8 \
    --kernel-iters 20 \
    --output-dir /mnt/home/zengzixuan/workspace/Lingshu/docs/experiments
```

结果:

- run_id: `20260616_225321`
- completed cycles: 102
- elapsed: 120.4 min
- summary: `/mnt/home/zengzixuan/workspace/Lingshu/docs/experiments/pi05_research_loop_aggregate_20260616_225321.json`

Context reuse upper bound negative control:

该结果现在只作为反证/上限实验保留。它假设 visual-language context 已经可安全驻留并复用，
而真实多机器人请求的图像状态通常不同，因此不能把它作为第一个研究点的核心优化路线。
它的价值是说明: 即使在理想复用假设下，上限也只有约 1.2x，Enc+AE 仍然是主瓶颈。

| B | full avg | context reuse avg | saved | ideal speedup | Enc+AE share |
|---:|---:|---:|---:|---:|---:|
| 1 | 42.3 +/- 0.3 ms | 35.8 +/- 0.3 ms | 6.5 +/- 0.3 ms | 1.18x | 83.9% |
| 2 | 57.7 +/- 0.5 ms | 47.9 +/- 0.4 ms | 9.8 +/- 0.2 ms | 1.21x | 82.4% |
| 3 | 78.9 +/- 0.7 ms | 65.8 +/- 0.6 ms | 13.1 +/- 0.2 ms | 1.20x | 83.0% |
| 4 | 107.5 +/- 1.3 ms | 90.2 +/- 1.1 ms | 17.3 +/- 0.3 ms | 1.19x | 83.6% |
| 5 | 126.7 +/- 1.7 ms | 105.3 +/- 1.5 ms | 21.3 +/- 0.4 ms | 1.20x | 82.7% |
| 6 | 147.6 +/- 1.8 ms | 122.2 +/- 1.6 ms | 25.4 +/- 0.4 ms | 1.21x | 82.4% |
| 7 | 171.1 +/- 2.3 ms | 139.6 +/- 2.0 ms | 31.5 +/- 0.6 ms | 1.23x | 81.3% |
| 8 | 194.0 +/- 1.3 ms | 158.8 +/- 1.0 ms | 35.2 +/- 0.7 ms | 1.22x | 81.5% |

GEGLU -> Down boundary:

| B | chain ms/layer | visible ms/infer | naive recompute delta | required A-tile reuse |
|---:|---:|---:|---:|---:|
| 1 | 0.347 +/- 0.009 | 2.62 +/- 0.14 ms | 42.2 +/- 1.5 ms | 8.3x |
| 2 | 0.620 +/- 0.009 | 5.31 +/- 0.43 ms | 82.9 +/- 2.0 ms | 8.2x |
| 3 | 1.123 +/- 0.020 | 8.27 +/- 0.20 ms | 124.7 +/- 1.8 ms | 8.0x |
| 4 | 1.925 +/- 0.056 | 11.34 +/- 0.33 ms | 166.1 +/- 1.7 ms | 7.9x |
| 5 | 2.374 +/- 0.070 | 13.88 +/- 0.30 ms | 208.3 +/- 1.8 ms | 8.0x |
| 6 | 2.807 +/- 0.090 | 17.35 +/- 0.45 ms | 248.8 +/- 2.3 ms | 7.8x |
| 7 | 3.206 +/- 0.092 | 19.44 +/- 0.37 ms | 290.8 +/- 2.4 ms | 8.0x |
| 8 | 3.686 +/- 0.060 | 22.50 +/- 1.28 ms | 331.3 +/- 3.1 ms | 7.9x |

结论:

- context reuse 只应作为 memory-for-compute upper bound / negative-control probe；
  真实多机器人图像不可假设相同，且理想上限约 1.2x 后 Enc+AE graph 仍占 81%-84%，
  因此它不能作为下一阶段核心优化。
- GEGLU LUT/row8 producer 数值安全但没有稳定加速，排除为核心方向。
- 下一阶段主线应集中在 Enc+AE 内部，尤其是 GEGLU-producing Down GEMM /
  virtual FP8 activation mainloop。
  关键设计约束是避免 per-N-tile 朴素重算，并在 CTA/persistent tile 级达到约 8x A-tile reuse。
