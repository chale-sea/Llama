#!/usr/bin/env bash
# =============================================================================
# 第 1 步 · 冒烟闸门（30 分钟，必须先过这一关）
# =============================================================================
#
# 目的：在投入 10000 条数据和几小时训练之前，先验证三件事能不能跑通：
#   a) ms-swift 认识 Qwen3.5 的 GatedDeltaNet 架构
#   b) T4 (fp16) + QLoRA 4bit 组合不会 OOM、不会报 kernel 不支持
#   c) 训完能存出 adapter
#
# 只要这三件事有一件不通，就不要再往下做数据 —— 先换路（见 TRAINING.md 第 3 节备选）。
#
# 在 Colab / Kaggle notebook 里用 %%bash cell 执行，或：
#   !bash /kaggle/working/sft/01_smoke_test.sh
#
# 前置：必须先跑过 00_prep_ms_swift.py，产出 /kaggle/working/data/ms_swift_train.jsonl
# =============================================================================
set -euo pipefail

# ── 环境变量 ─────────────────────────────────────────────────────────────────
# USE_MCORE_GDN=0：GDN 默认走 Megatron 实现（要 megatron-core>=0.16），
#                  Kaggle 单卡上没必要装 Megatron，切回 transformers 实现。
export USE_MCORE_GDN=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/kaggle/temp/hf            # 权重缓存放临时盘，别占 20GB 的 working
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

WORK=/kaggle/temp/smoke
rm -rf "$WORK" && mkdir -p "$WORK"

echo "==================== 1/4 安装依赖 ===================="
pip install -q -U "ms-swift>=4.1" transformers accelerate peft bitsandbytes

echo "==================== 2/4 准备 100 条冒烟数据 ===================="
python - <<'PY'
import json, random
from pathlib import Path

src = Path("/kaggle/working/data/ms_swift_train.jsonl")
rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
random.seed(42)
random.shuffle(rows)
sample = rows[:100]
out = Path("/kaggle/temp/smoke/smoke_train.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sample), encoding="utf-8")
print(f"[OK] 冒烟样本 {len(sample)} 条 -> {out}")
PY

echo "==================== 3/4 QLoRA 4bit + fp16 训练 ===================="
# 关键参数说明：
#   --torch_dtype float16   T4/P100 无 bf16 硬件支持，只能用 fp16
#   --quantization_bit 4    QLoRA；若报 GDN 层不支持量化，删掉这行退化为 fp16 LoRA
#   --group_by_length true  GDN 的 transformers 实现不支持 packing，官方推荐用它代替
#   --max_steps 20          冒烟只跑 20 步
#   --add_non_thinking_prefix true  让模型学会「直接答，不吐思考块」
#   --loss_scale ignore_empty_think 只对答案部分计损失
swift sft \
  --model Qwen/Qwen3.5-4B \
  --tuner_type lora \
  --dataset /kaggle/temp/smoke/smoke_train.jsonl \
  --torch_dtype float16 \
  --quantization_bit 4 \
  --num_train_epochs 1 \
  --max_steps 20 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --group_by_length true \
  --max_length 1024 \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --output_dir "$WORK/out" \
  --logging_steps 1 \
  --save_steps 20 \
  --save_total_limit 1 \
  --dataloader_num_workers 2 \
  --dataset_num_proc 2

echo "==================== 4/4 冒烟推理，看输出是不是纯 JSON ===================="
ADAPTER=$(find "$WORK/out" -name "checkpoint-*" -type d | head -1)
echo "[INFO] adapter = $ADAPTER"

swift infer \
  --adapters "$ADAPTER" \
  --load_data_args false \
  --val_dataset /kaggle/temp/smoke/smoke_train.jsonl \
  --max_new_tokens 64 \
  --temperature 0.0 \
  --stream false \
  --enable_thinking false 2>&1 | tail -30 || true

cat <<'EOF'

────────────────────────────────────────────────────────────────────────
冒烟结论怎么读：

  ✅ 训练 20 步跑完 + 有 checkpoint 目录 + 推理输出是一行 JSON
     → 链路通，可以去做 10000 条数据和正式训练

  ❌ 报 "not supported" / "unexpected key" 之类
     → ms-swift 或 transformers 版本太旧：pip install -U 'ms-swift>=4.1' transformers

  ❌ 报 OOM
     → 删掉 --quantization_bit 4 用纯 fp16 LoRA，或把 max_length 降到 512

  ❌ 量化相关报错（bnb 不认识 GDN 的 linear）
     → 同上去掉 --quantization_bit 4

  ❌ 全套都不通
     → 换 Qwen3-4B（标准 Transformer，工具链成熟）过渡，架构新颖度不是重点，
       「10000 条蒸馏数据把窄域 NLU 做到 95%+」才是重点
────────────────────────────────────────────────────────────────────────
EOF
