# 碩士論文研究計畫：基於 Mamba2-Transformer 混合式骨幹與光流導引之漸進式高解析度視訊幀插補

## 1. 研究概述 (Research Overview)

本研究提出一個**漸進式 (Progressive)** 的研究路徑，旨在解決視訊幀插補 (VFI) 領域的兩大難題：**長序列全域感知 (Global Context)** 與 **大動作模糊 (Large Motion Blur)**，並最終實現 **4K 超高解析度紋理保持**。

不同於直接訓練一個龐大的複雜模型，本研究將分為三個階段進行驗證與優化，逐步構建出最終的完整架構。

### 核心差異化定位 (Differentiation from Existing Work)

| 方法 | 發表 | Inter-frame Modeling | 局部精細度 | 複雜度 |
| :--- | :--- | :--- | :--- | :--- |
| **RIFE** (Huang et al.) | ECCV 2022 | Convolution (IFBlocks) | 高 | 低 |
| **EMA-VFI** (Zhang et al.) | CVPR 2023 | Local Attention + CNN Hybrid | 高 | 中 |
| **AMT** (Li et al.) | CVPR 2023 | All-pairs Correlation + Multi-field Refinement | 高 | 中 |
| **SGM-VFI** (Liu et al.) | CVPR 2024 | Sparse Global Matching + Local Flow | 高 (全域匹配) | 高 |
| **VFIMamba** (Zhang et al.) | NeurIPS 2024 | Pure SSM / Mamba1 (Mixed-SSM Block) | 依賴 CAB | 低 (線性) |
| **LC-Mamba** (Jeong & Rhee) | CVPR 2025 | Windowed SSM (Hilbert Curve) | 改善歷史衰減 | 低 (線性) |
| **BiM-VFI** (Seo et al.) | CVPR 2025 | Bidirectional Motion Field | 高 | 中 |
| **Ours (Thesis)** | -- | **Mamba2 (SSD) + Gated Window Attention Hybrid** | **Transformer 顯式建模** | 低~中 (線性主導) |

**核心創新點**：

1. **Mamba2 (SSD) 取代 Mamba1**：Mamba2 基於 State Space Duality 理論 (Dao & Gu, ICML 2024)，揭示 SSM 與 linear attention 之間的數學對偶性，使用 SSD 演算法，state dimension 從 16 擴大到 64~128，訓練速度比 Mamba1 快 2-8 倍，且可利用 tensor core 矩陣乘法。VFIMamba 使用的是 Mamba1 的 S6 scan；本研究升級為 Mamba2 的 SSD，理論上能在相同複雜度下獲得更強的表達能力。

2. **Gated Window Attention**：借鑒 Qwen 團隊 NeurIPS 2025 最佳論文 "Gated Attention" 的核心發現，在 Window Self-Attention 的 SDPA 輸出後加入 head-specific sigmoid gate。此 gating 機制引入非線性、促進稀疏性、消除 attention sink，提升訓練穩定性與長序列外推性能。在視覺任務中，這有助於讓每個 attention head 專注於不同的空間區域。

3. **SSM + Attention 混合式 VFI**：VFIMamba 的 inter-frame modeling 完全依賴 SSM，局部精細紋理僅透過 CAB 補充。MambaVision (CVPR 2025, NVIDIA) 已證實在最後幾層加入 Self-Attention 能顯著提升長距離空間建模。MaTVLM (ICCV 2025) 驗證了以 Mamba2 層替換部分 Transformer 層的混合策略。本研究在 VFI 場景中結合兩者，設計 **Local-Global Synergistic Block (LGS Block)**。

---

## 2. 背景技術與 Preliminary 相關研究整理

以下整理與碩論 Chapter 3 (Background and Preliminary) 各節相關的近期關鍵研究。

### 2.1 Attention Mechanisms (3.1)

#### 2.1.1 Scaled Dot-Product Attention / Multi-Head Attention (3.1.1, 3.1.2)

- **Gated Attention** (Qiu et al., NeurIPS 2025 Best Paper): 在 SDPA 輸出後加入 sigmoid gate `Y' = sigma(X * W_theta) * Y`。核心發現：(1) 引入非線性打破 softmax attention 的 low-rank 限制；(2) query-dependent 稀疏 gating 自動抑制 attention sink 與 massive activation；(3) 提升訓練穩定性，容許更大的 learning rate。已整合進 Qwen3-Next 模型。在我們的 Window Attention 分支中可直接應用。
- **FlashAttention-2/3** (Dao, 2023/2024): IO-aware exact attention 實作，透過 tiling 與 kernel fusion 實現顯著加速。Window Attention 可直接使用 FlashAttention kernel。
- **MambaIRv2 Attentive State-Space Equation** (Guo et al., CVPR 2025): 修改 Mamba 的 output matrix C 為 global query，使 SSM 具備非因果的 attention-like 全域查詢能力。值得在 SSM 分支考慮引入。

#### 2.1.2 Channel Attention / Spatial Attention (3.1.3, 3.1.4)

- **CAB (Channel Attention Block)**: VFIMamba 與 MambaIR 系列標準組件，SE-style squeeze-and-excitation。
- **CBAM / Spatial Attention**: 結合 channel 與 spatial 注意力。在 LGS Block 的 fusion 階段可考慮引入 spatial attention 引導融合權重。

### 2.2 Transformer Architecture (3.2)

- **Swin Transformer** (Liu et al., ICCV 2021): Shifted window attention 建立跨窗口連接，是 Window Attention 的標準實踐。
- **Window Attention in VFI**: EMA-VFI 使用 inter-frame attention，AMT 使用 all-pairs correlation volume。本研究 LGS Block 採用 8x8 非重疊窗口 + shifted window (每隔一個 LGS Block 做 shift)，平衡效率與跨窗口資訊流動。

### 2.3 State Space Models (3.3)

#### 2.3.1 SSM Formulation and Discretization (3.3.1)

- **S4** (Gu et al., ICLR 2022): 結構化狀態空間模型，線性複雜度序列建模。
- **Mamba / S6** (Gu & Dao, 2023): 引入 selective scan 機制，使 SSM 參數成為 input-dependent，打破 LTI 限制。
- **Mamba2 / SSD** (Dao & Gu, ICML 2024): 揭示 SSM 與線性注意力之間的結構對偶性 (Structured State Space Duality)。核心改進：(a) scalar-times-identity state matrix 等價於 1-semiseparable causal mask 的 masked attention；(b) state dimension N 從 16 擴大到 64/128；(c) 使用 matmul 作為基本運算 (利用 tensor core，比非 matmul 快 16 倍)；(d) 訓練速度比 Mamba1 快 2-8 倍；(e) Hybrid Mamba2+Attention 模型 (如 mamba2attn-2.7b) 已驗證比純 SSD 或純 Transformer 更好。

#### 2.3.2 Selective Scan Mechanism (3.3.2)

- Mamba1 使用 parallel associative scan，受限於 non-matmul FLOPs。
- Mamba2 使用 chunkwise 演算法，將序列分成 chunks，chunk 內用矩陣乘法 (SSD dual form)，chunk 間用遞推 (SSM recurrence form)，實現最佳的硬體效率。

#### 2.3.3 2D-Selective-Scan / SS2D (3.3.3)

- **VMamba SS2D** (Liu et al., 2024): 4 方向 Z-shaped scan (水平正反 + 垂直正反)，是 VFIMamba 的基礎。
- **VFIMamba Interleaved Rearrangement**: 先交錯排列雙幀特徵再掃描，優於 sequential 拼接。
- **MaIR NSS** (Li et al., CVPR 2025): Nested S-shaped Scan，使用 stripe-based 區域 + S-shaped 路徑，同時保持 locality 與 continuity，在 image restoration 14 個 dataset SOTA。此掃描策略值得在 Mamba2 分支中嘗試。
- **LC-Mamba Hilbert Curve Scan** (CVPR 2025): Shifted window + Hilbert curve，解決歷史衰減問題。
- **EAMamba MHSSM** (2025): Multi-Head Selective Scan，高效聚合多方向掃描序列，避免計算量隨掃描方向數量線性增加。

### 2.4 Motion Modeling (3.4)

#### 2.4.1 Optical Flow and Backward Warping (3.4.1)

- **RIFE IFNet** (Huang et al., ECCV 2022): 端到端中間光流估計，coarse-to-fine IFBlocks (H/4 -> H/2 -> H)。
- **IFRNet** (Kong et al., CVPR 2022): 單一 encoder-decoder 中聯合精煉光流與特徵。
- **BiM-VFI** (Seo et al., CVPR 2025): Bidirectional Motion field 描述非均勻運動 (加速/減速/變向)，Content-Aware Upsampling + VFI-centric flow distillation。LPIPS 改善 26%，STLPIPS 改善 45%。
- **OCAI** (Jeong et al., CVPR 2024): Occlusion and Consistency Aware Interpolation，提升遮擋區域的光流估計品質。
- **FlowFormer++** (Shi et al., 2023): Masked Cost Volume Autoencoding 預訓練光流估計，更高效的記憶體使用，可作為 flow supervision 的 teacher model。
- **Depth-Aware VFI** (Yan et al., CVPR 2025): 結合 differential curves 與深度感知處理模糊視訊幀插補。

#### 2.4.2 4D Cost Volume (3.4.2)

- **RAFT** (Teed & Deng, ECCV 2020): All-pairs correlation volume 開創性工作，建立雙幀所有像素對的 4D 相關體。
- **AMT** (Li et al., CVPR 2023): All-Pairs Multi-Field Transforms，在 VFI 中引入 bidirectional correlation volume + multi-field refinement。
- **SGM-VFI** (Liu et al., CVPR 2024): Sparse Global Matching，先用局部資訊估計初始光流，再用全域匹配修正，專注 large motion SOTA。
- **EDNet** (Zhang et al., CVPR 2021): Efficient Disparity Estimation，Cost Volume Combination + Attention-based Spatial Residual，展示如何在 cost volume 上高效應用注意力機制。

---

## 3. 進階技術評估 (Advanced Techniques Evaluation)

### 3.1 mHC: Manifold-Constrained Hyper-Connections (DeepSeek, Dec 2025)

**概述**：mHC 擴展傳統殘差連接，將多條平行 residual stream 的混合矩陣投影到 Birkhoff Polytope (雙隨機矩陣流形) 上，透過 Sinkhorn-Knopp 演算法強制約束。

**效果**：在 27B 模型上控制信號放大至 1.6 倍 (對比 HC 的 3000 倍爆炸)，BIG-Bench Hard +2.1%，訓練額外開銷僅 6.7%。

**對本研究的適用性評估**：

| 面向 | 評估 |
| :--- | :--- |
| **正面** | mHC 作為 drop-in 替換標準殘差連接，可直接應用於 LGS Block 堆疊中的跨層資訊流。在多尺度 backbone 中，不同 scale 間的特徵傳遞可受益於更穩定的梯度流動。 |
| **疑慮** | mHC 主要驗證於大規模 LLM (3B~27B 參數)，本研究的 VFI 模型相對較小 (~10-30M 參數)，depth 也較淺 (< 50 layers)。小模型中 residual stream 爆炸的問題不如 LLM 嚴重。 |
| **實作複雜度** | Sinkhorn-Knopp 需要額外的反覆迭代 (預設 20 次)；需要 kernel fusion 與 recomputing 策略。在 V100 上可能造成額外延遲。 |
| **建議** | 作為 Phase 1 的 **Optional Ablation** (Exp-1h)：替換 LGS Block 之間的殘差連接為 mHC，觀察收斂速度或精度變化。若提升有限，則不加入最終架構，但可在論文中作為 study 討論。不建議作為核心創新。 |

### 3.2 Gated Attention (Qwen, NeurIPS 2025 Best Paper)

**概述**：在 SDPA 輸出後加入 head-specific sigmoid gate。

**對本研究的適用性評估**：

| 面向 | 評估 |
| :--- | :--- |
| **正面** | 實作極其簡單 (單一線性投影 + sigmoid)，幾乎零額外參數，可直接應用於 Window Attention 分支。消除 attention sink 對 VFI 的空間注意力分布有正面影響。 |
| **與 VFI 的適配** | VFI 中 Window Attention 處理的是雙幀特徵，attention distribution 需要精確對應空間位移。Gated Attention 允許 head 輸出接近零向量，避免噪聲區域被強制分配 attention weight。 |
| **建議** | **強烈建議整合**進 LGS Block 的 Attention 分支，作為核心設計之一。NeurIPS 2025 最佳論文且已被 Qwen3-Next 採用，在論文中引用具說服力。 |

### 3.3 MaTVLM 的 Mamba2 初始化策略 (ICCV 2025)

**概述**：MaTVLM 利用 attention 與 Mamba2 的內在對偶關係，用 pre-trained attention weights 初始化 Mamba2 layers，加速收斂。實現 3.6x 推理加速、27.5% 記憶體減少。

**對本研究的啟發**：在 Phase 1 中，若先訓練一個 pure Window Attention 版本 (Exp-1b)，可嘗試用其 attention weights 初始化 Mamba2 分支，可能加速 hybrid model 的收斂。

### 3.4 DepthFlow (BrokenSource)

**概述**：DepthFlow 是一個 image-to-video 的 3D parallax 動畫工具，利用 depth map + GLSL shader 產生視差效果。

**對本研究的適用性評估**：可作為碩論中 **應用展示 (Application Demo)** 的一環，展示 VFI 在創意內容生成中的應用。不建議作為核心方法的一部分，但可在 Chapter 5 (Applications & Future Work) 中提及。

### 3.5 Diffusion-Based VFI 趨勢 (Related but Orthogonal)

以下為近期 diffusion-based VFI 方法，與本研究的確定性方法路線不同，但需在 Related Work 中討論：

- **EDEN** (2025): Enhanced Diffusion for large-motion VFI，LPIPS 減少約 10%。
- **Hierarchical Flow Diffusion** (CVPR 2025): 在 flow 空間做 hierarchical diffusion，比其他 diffusion VFI 快 10 倍以上。
- **ToonCrafter** (2024): Generative Cartoon Interpolation。

本研究不使用 diffusion model，但可在 perceptual metrics (LPIPS, STLPIPS) 上與 diffusion methods 比較。

---

## 4. 階段性研究規劃 (Phased Research Roadmap)

### 第一階段：混合式骨幹驗證 (Phase 1: Hybrid Backbone Baseline)

**目標**：驗證 **Mamba2 (SSD)** 與 **Gated Window Attention** 結合的有效性。

#### 4.1.1 核心架構：Local-Global Synergistic Block (LGS Block)

```
Input Features (B, C, H, W)
    |
    +-- [Branch A] Mamba2 Path (SSD)
    |     --> 2D-to-1D Scan (interleaved rearrangement, 4-direction or NSS)
    |     --> Mamba2 SSD Layer (d_state=64, d_conv=4, expand=2)
    |     --> Global long-range dependency (O(n), tensor core acceleration)
    |
    +-- [Branch B] Gated Window Attention Path
    |     --> Window Partition (8x8, shifted every other block)
    |     --> Multi-Head Self-Attention (FlashAttention kernel)
    |     --> Sigmoid Gate: Y' = sigma(X * W_g) * Y
    |     --> Local fine-grained texture alignment
    |
    +-- Fusion: Concat + 1x1 Conv + Residual
    +-- Channel Attention Block (CAB)
Output Features (B, C, H, W)
```

**設計決策依據**：

- **Mamba2 分支**：使用 `mamba_ssm.Mamba2` (d_state=64, d_conv=4, expand=2)，比 VFIMamba 的 S6 更高效。掃描策略沿用 interleaved rearrangement，可額外嘗試 MaIR 的 NSS。
- **Gated Window Attention 分支**：8x8 window size，shifted window (如 Swin)。核心加入 Qwen-style SDPA output gating (head-specific sigmoid)。
- **融合策略**：Concat 後 1x1 Conv 壓縮回原通道數，residual + CAB。
- **Multi-scale 配置**：低解析度層 SSM 佔比多 (全域感知)，高解析度層 Attention 佔比多 (局部紋理)。

#### 4.1.2 整體 Pipeline

```
I_0, I_1 (B, 3, H, W)
    |
    v
[1] CNN Feature Extractor (Stem)
    | --> shallow features (B, C, H/2, W/2)
    v
[2] Multi-scale LGS Backbone
    | Scale 1 (H/4, W/4): N_1 x LGS Block (Mamba2 heavy)
    | Scale 2 (H/8, W/8): N_2 x LGS Block (balanced)
    | Scale 3 (H/16, W/16): N_3 x LGS Block (Attention heavy)
    v
[3] Frame Generation (Motion Estimator + Refiner)
    | --> Optical Flow F_{t->0}, F_{t->1}
    | --> Fusion Map M, Residual R
    v
[4] Backward Warping + Fusion --> I_t (B, 3, H, W)
```

#### 4.1.3 訓練配置

| 項目 | 設定 |
| :--- | :--- |
| 訓練資料 | Vimeo90K Triplet (73171 triplets) |
| Optimizer | AdamW, lr=1e-4, cosine decay -> 1e-5 |
| Batch Size | 24 (6 per GPU x 4 GPUs) |
| Epochs | 300 |
| Crop Size | 224 x 224 (random crop + flip + rotate) |
| Loss | L1 + Perceptual (VGG-19 relu2_2, relu3_3, relu4_3) |
| Hardware | 4x V100 32GB (DDP) |
| Precision | AMP, Mamba2 核心參數保持 FP32 |

#### 4.1.4 Phase 1 及格標準

| Benchmark | 及格線 | 目標 | RIFE | VFIMamba |
| :--- | :--- | :--- | :--- | :--- |
| Vimeo90K (PSNR) | >= 35.5 dB | >= 36.0 dB | 35.62 | 36.64 |
| UCF101 (PSNR) | >= 35.0 dB | >= 35.2 dB | 35.28 | 35.47 |
| SNU-FILM Hard | >= 29.5 dB | >= 30.0 dB | -- | -- |

#### 4.1.5 Phase 1 Ablation Study

| 編號 | 配置 | 驗證目標 |
| :--- | :--- | :--- |
| Exp-1a | Pure Mamba2 (SSD only, 無 Attention) | Mamba2 baseline |
| Exp-1b | Pure Gated Window Attention (無 SSM) | Attention baseline |
| Exp-1c | **Hybrid LGS Block (Mamba2 + Gated Attn)** | 混合架構優勢 |
| Exp-1d | 不同融合策略 (Add vs. Concat vs. Gating) | 最佳融合方式 |
| Exp-1e | 不同 Mamba2/Attn 比例 (2:1 vs 1:1 vs 1:2) | 最佳配比 |
| Exp-1f | 有/無 CAB | channel attention 貢獻 |
| Exp-1g | Window Attn 有/無 Gated Attention | Gating 效果 |
| Exp-1h | (Optional) mHC 殘差連接 | mHC 小模型效果 |
| Exp-1i | (Optional) MaIR NSS vs. 4-dir scan | 掃描策略比較 |

---

### 第二階段：光流導引增強 (Phase 2: Motion-Aware Guidance)

**目標**：引入 **光流估計模組**，解決大動作場景偽影問題。

#### 4.2.1 光流模組設計

```
I_0, I_1 (B, 3, H, W) + timestep t
    |
    v
[IFBlock-1] (H/4) --> coarse flow F^0 + mask M^0
    v
[IFBlock-2] (H/2) --> refined flow F^1 = F^0 + dF^1
    v
[IFBlock-3] (H)   --> final flow F^2, final mask M
```

**關鍵改進**：

1. **Feature Pre-warping**：光流 warp 淺層特徵到中間時刻 t，再送入 LGS Backbone。
2. **Flow-aware Backbone**：光流資訊作為 conditioning 注入 LGS Block。
3. **Multi-field Refinement (Optional)**：參考 AMT 多組 fine-grained flows。
4. **Occlusion-aware (Optional)**：參考 OCAI (CVPR 2024)。

#### 4.2.2 Phase 2 Ablation Study

| 編號 | 配置 | 驗證目標 |
| :--- | :--- | :--- |
| Exp-2a | 無光流 (Phase 1 baseline) | 對照組 |
| Exp-2b | Image-level warping | 基線方法 |
| Exp-2c | **Feature-level warping** | 核心方法 |
| Exp-2d | Flow conditioning | 資訊注入 |
| Exp-2e | 2c + FlowFormer++ distillation | 光流品質 |
| Exp-2f | (Optional) Multi-field refinement | 遮擋處理 |

---

### 第三階段：高解析度紋理注入 (Phase 3: High-Fidelity Synthesis with X4K)
**目標**：透過 **X4K1000FPS** 資料集與特殊訓練策略，賦予模型處理 4K 影像的能力，從「插補準確」提升至「紋理逼真」。

#### 3.1 實驗配置與預期貢獻矩陣 (Experimental Protocol & Contributions)

| 階段 (Stage) | 資料集 (Dataset) | 關鍵操作細節 (Key Operations) | 預期帶來的論文貢獻 (Contribution) |
| :--- | :--- | :--- | :--- |
| **訓練 (Train)** | **Vimeo90K + X4K1000FPS** | **混合比例**：建議 **1:1** 或 **2:1** (Vimeo佔多)。<br>**Patch Size**：**256x256**。<br>**Augmentation**：對 X4K 做**時間跳幀 (Stride 8~32)**。 | 1. **多尺度適應性**：證明模型能同時處理低畫質與 4K 畫質。<br>2. **細節增強**：Mamba 模組能從 X4K 學到更好的紋理修復能力。 |
| **評估 1 (基礎)** | Vimeo90K-Test, UCF101, Middlebury | (維持不變) | **確保基礎分數 (PSNR/SSIM) 不掉**：證明混合訓練沒有副作用，維持模型在標準解析度下的通用性。 |
| **評估 2 (亮點)** | **SNU-FILM (Hard/Extreme)** | (維持不變) | **光流 Block 的主戰場**：加上 X4K 的大位移訓練 (Stride 32) 有助於顯著減少大動作下的邊緣模糊與鬼影。 |
| **評估 3 (效率/畫質)** | **X4K1000FPS-Test** | (維持不變) | **超高解析度優勢**：由於訓練集包含 X4K，理論上此處分數 (PSNR/SSIM/LPIPS) 將大幅超越僅用 Vimeo 訓練的 SOTA 模型。 |

#### 3.2 執行策略備註
*   **混合訓練策略**：將這兩個性質差異巨大的資料集放在一起訓練，利用 Random Crop 解決解析度落差，並利用 Temporal Subsampling 解決 X4K 原生動作過小的問題。
*   **IO 優化**：X4K 訓練時務必預先處理資料 (Resize/Crop/LMDB)，否則 GPU 會因為等待硬碟讀取而閒置。

---

## 5. 論文架構與貢獻對應表

| 章節 | 內容 | 貢獻 |
| :--- | :--- | :--- |
| **Method A** | LGS Block: Mamba2 + Gated Window Attention | 首次在 VFI 中使用 Mamba2 (SSD) + Gated Attention |
| **Method B** | Flow-Guided Feature Warping | Feature-level pre-warping for hybrid backbone |
| **Method C** | Curriculum Mixed Training | 4K curriculum learning + temporal subsampling |
| **Experiments** | Comprehensive Ablation | Mamba2 vs Mamba1, Gated vs Non-gated, 融合策略 |

---

## 6. 執行計畫 (14 Weeks)

- **Week 1-2**: Infrastructure, LGS Block (Mamba2 SSD + Gated Window Attn), dry run
- **Week 3-4**: Phase 1 ablation (Exp-1a~1i), 300 epoch training, benchmarks
- **Week 5-7**: Phase 2 flow integration, feature pre-warping, ablation (Exp-2a~2f)
- **Week 8-11**: Phase 3 X4K preprocessing, curriculum training, ablation (Exp-3a~3e)
- **Week 12-14**: Paper writing, ERF visualization, Mamba2 vs Mamba1 speed comparison

---

## 7. 實作備忘錄

### 7.1 核心依賴
- **mamba-ssm >= 2.0**: Mamba2 SSD (`from mamba_ssm import Mamba2`)
- **PyTorch >= 2.0**: FlashAttention for Gated Window Attention
- **einops, timm**: Tensor manipulation
- **AMP 精度**：Mamba2 參數保持 FP32 (對遞迴動態敏感)

### 7.2 重要 GitHub Repos

| 專案 | 連結 | 用途 |
| :--- | :--- | :--- |
| Mamba (official) | `state-spaces/mamba` | Mamba2 SSD (modules/mamba2.py) |
| VFIMamba | `MCG-NJU/VFIMamba` | SSM VFI, MSB, curriculum learning |
| SGM-VFI | `MCG-NJU/SGM-VFI` | Sparse Global Matching |
| AMT | `MCG-NKU/AMT` | All-pairs correlation |
| EMA-VFI | `MCG-NJU/EMA-VFI` | Hybrid CNN+Transformer VFI |
| RIFE | `hzwer/ECCV2022-RIFE` | IFNet, training pipeline |
| MambaIR/v2 | `csguoh/MambaIR` | RSSB, ASE |
| MambaVision | `NVlabs/MambaVision` | Hybrid design |
| MaIR | `XLearning-SCU/2025-CVPR-MaIR` | NSS scan |
| MaTVLM | `hustvl/MaTVLM` | Mamba2-Transformer hybrid |
| Gated Attention | `qiuzh20/gated_attention` | SDPA gating |
| mHC | `tokenbender/mHC-manifold-constrained-hyper-connections` | mHC module |
| Awesome-VFI | `CMLab-Korea/Awesome-Video-Frame-Interpolation` | VFI 文獻列表 |
| DepthFlow | `BrokenSource/DepthFlow` | Depth parallax (應用展示) |

### 7.3 Preliminary 章節核心引用

| 章節 | 核心引用 |
| :--- | :--- |
| 3.1.1 SDPA | Vaswani+ (2017), Qiu+ (NeurIPS 2025, Gated Attn) |
| 3.1.2 MHA | Vaswani+ (2017), Dao (FlashAttention, 2023) |
| 3.1.3 Channel Attn | Hu+ (SE-Net, CVPR 2018), Guo+ (MambaIR, ECCV 2024) |
| 3.1.4 Spatial Attn | Woo+ (CBAM, ECCV 2018) |
| 3.2 Transformer | Dosovitskiy+ (ViT, 2021), Liu+ (Swin, ICCV 2021) |
| 3.3.1 SSM | Gu+ (S4, 2022), Gu & Dao (Mamba, 2023) |
| 3.3.2 Selective Scan | Gu & Dao (Mamba, 2023), Dao & Gu (Mamba2/SSD, ICML 2024) |
| 3.3.3 2D-SS | Liu+ (VMamba, 2024), Li+ (MaIR/NSS, CVPR 2025) |
| 3.4.1 Optical Flow | Teed & Deng (RAFT, 2020), Huang+ (RIFE, 2022) |
| 3.4.2 4D Cost Volume | Teed & Deng (RAFT), Li+ (AMT, CVPR 2023), Zhang+ (EDNet, 2021) |

---

*文件更新日期：2026-02-02*
*版次：v4.0 (Mamba2 + Gated Attention + Comprehensive Literature)*