---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: 13b7f5127f83c972d4bf7ee4581bff74
    PropagateID: 13b7f5127f83c972d4bf7ee4581bff74
    ReservedCode1: 304402203de1717618b4d933927c3a34fa4013a676988ea39c6094c015dd771e24a055b50220549100697594013733a8b5efb1bb680bb08ea3a603fb25bf8c0b4cbb0af78b6c
    ReservedCode2: 30450221009ec3683a1e04a36462cb10857fcae055579f7826a19f64ee9e80efcfcbb89bfc02202039936477e9689a5a2bac88304be119521c8f8a943accee9c27feb08405b713
---

# EcoCompute Demo 使用示例

基于实测数据：RTX 5090 · Qwen2-7B

---

## 场景 1：你的模型在列表里 → 使用 "Tested models" Tab

**背景**：你正在用 RTX 5090 运行 Qwen2-7B，想知道是否应该用 NF4 量化。

**步骤**：

1. 打开页面，默认进入 **"Tested models"** Tab
2. GPU 选择 **"RTX 5090 · Blackwell · 32GB"**
3. Model 选择 **"Qwen2-7B (7B)"**

**结果**：

| 精度 | 能耗/1M tokens | 变化 |
|------|---------------|------|
| FP16 | 5,509 kJ | baseline |
| NF4  | 4,878 kJ | **−11%** |

**Verdict**：
> ↓ NF4 saves 11% energy — quantization is worth it for this model.
>
> Ready to quantize? `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')`

**图表**：Crossover curve 显示 7B 模型位置在零线下方（绿色 = 节能区域）

---

## 场景 2：你的模型不在列表里 → 使用 "Your model" Tab

**背景**：你计划用 A100 运行一个 13B 的自定义模型，想预估 NF4 的能耗影响。

**步骤**：

1. 点击 **"Your model"** Tab
2. Model size 输入 **13**
3. 点击 Advanced 展开，选择：
   - GPU architecture: **Ampere** (A100)
   - Precision: **NF4**

**结果**：

- 估算能耗变化：**−18%**（基于拟合曲线）
- Confidence: high (interpolated from A800 measurements)
- Crossover point: ≈6B

**Verdict**：
> ↓ Your 13B model is above the crossover on Ampere — NF4 saves ~18% energy.

**图表**：Fitted crossover curve 显示 13B 模型位置在零线下方

---

## 场景 3：发现小模型量化反而费电

**背景**：你好奇为什么 TinyLlama-1.1B 在 RTX 5090 上量化反而费电。

**步骤**：

1. "Tested models" Tab → GPU: RTX 5090 → Model: TinyLlama-1.1B

**结果**：

| 精度 | 能耗/1M tokens | 变化 |
|------|---------------|------|
| FP16 | 1,659 kJ | baseline |
| NF4  | 2,098 kJ | **+26%** |

**Verdict**：
> ↑ NF4 adds 26% energy — quantization actually costs more for this model.
>
> Stick with FP16 — no quantization config needed.

**原因**：这就是 **Crossover 效应**——小模型的量化开销（反量化计算）超过内存带宽节省。

---

## 场景 4：对比不同 GPU 架构

**背景**：你有两个 GPU，想知道在哪上面量化更划算。

**步骤**：

在 "Tested models" Tab 切换 GPU，对比同一模型（如 Qwen2.5-3B）：

| GPU | FP16 | NF4 | 变化 |
|-----|------|-----|------|
| RTX 5090 | 3,383 kJ | 3,780 kJ | +12% (费电) |
| T4 | 11,268 kJ | 11,112 kJ | −1% (基本持平) |
| A800 | — | — | (未测3B) |

**洞察**：量化效果高度依赖 GPU 架构，不是所有场景都适合量化。

---

## 场景 5：使用自然语言查询

**背景**：你想快速查询 "13B on H100, NF4" 的情况。

**步骤**：

1. "Your model" Tab
2. 在 Ask 输入框输入：`13B on H100, NF4`

系统会自动解析并填充表单。

---

## 核心洞察总结

| 场景 | 结论 |
|------|------|
| **≥7B on RTX 5090** | NF4 节能 ~11%，推荐量化 |
| **1.1B on RTX 5090** | NF4 费电 +26%，保持 FP16 |
| **13B on A100** | NF4 节能 ~18%，推荐量化 |
| **任意小模型** | 低于 Crossover 点，量化可能费电 |

---

## 决策流程图

```
你的模型是多大？
├── < 5B → 用 "Tested models" 或 "Your model" 查 Crossover
│   ├── 在 Crossover 上方 → 保持 FP16
│   └── 在 Crossover 下方 → NF4 节能
│
└── ≥ 5B → 很可能 NF4 节能
    └── 确认：用 "Your model" 获取精确估算
```

---

## 下一步

如果 EcoCompute 告诉你该量化，下一步是使用量化框架执行：

**bitsandbytes 示例**：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4'
)

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    quantization_config=quantization_config
)
```

**vLLM 示例**：
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="your-model-name",
    quantization="fp8",  # 或 "awq", "gptq"
)
```

---

*数据来源：Hongping Zhang, "Weight-Only Quantization Does Not Always Save Energy"*
*实测数据：NVML 10Hz 采样，10 次重复，CV<3%*