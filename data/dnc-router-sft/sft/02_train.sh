#!/usr/bin/env bash
# =============================================================================
# 第 2 步 · 正式训练（Qwen3.5-4B QLoRA，Colab / Kaggle 单卡 T4）
# =============================================================================
#
# 执行方式：**不要用交互式 cell 挂机**。
#   Kaggle：右上角 Save Version → Save & Run All (Commit)，后台跑，关浏览器也继续
#   Colab ：免费版没有后台执行，必须保持浏览器标签页打开（约 2-3 小时）
#
# 命令：!bash /kaggle/working/sft/02_train.sh
# =============================================================================
set -euo pipefail

# ── 环境变量 ─────────────────────────────────────────────────────────────────
export USE_MCORE_GDN=0                       # GDN 走 transformers 实现，避开 Megatron
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/kaggle/temp/hf
export TOKENIZERS_PARALLELISM=false

OUT=/kaggle/working/out/dnc-router-v2-4b     # adapter 很小（几十 MB），放 working 便于持久化
mkdir -p "$OUT"

echo "==================== 安装依赖 ===================="
pip install -q -U "ms-swift>=4.1" transformers accelerate peft bitsandbytes

# ── 超参选择理由（改之前先读） ───────────────────────────────────────────────
# lora_rank 32 / alpha 64
#   官方 Qwen3.5-4B 示例用的是 rank 8 / alpha 32（通用指令微调，求泛化）。
#   我们是「窄域结构化抽取」，目标行为单一且要学死，rank 提到 32 更合适；
#   rank 64 在单卡 T4 上显存吃紧且容易过拟合小数据，不建议。
# per_device_train_batch_size 2 × gradient_accumulation_steps 4 = 有效 batch 8
#   T4 16GB 跑 4B QLoRA 的稳妥点。若 OOM 先把 batch 降到 1。
# num_train_epochs 3
#   10000 条、窄任务，3 epoch 是常规起点；看 val_loss 抬头就降到 2。
# learning_rate 1e-4 + cosine
#   与官方示例一致，LoRA 常用量级。
# max_length 1024
#   我们的样本（system ~250 token + 指令 + JSON）实测远小于此，1024 足够。
# save_steps 200 + save_total_limit 2
#   Kaggle session 会断，必须留 checkpoint 供 resume；只留 2 个防爆盘。

swift sft \
  --model Qwen/Qwen3.5-4B \
  --tuner_type lora \
  --dataset /kaggle/working/data/ms_swift_train.jsonl \
  --val_dataset /kaggle/working/data/ms_swift_val.jsonl \
  --torch_dtype float16 \
  --quantization_bit 4 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --group_by_length true \
  --max_length 1024 \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --output_dir "$OUT" \
  --logging_steps 10 \
  --save_steps 200 \
  --save_total_limit 2 \
  --eval_steps 200 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4

echo
echo "==================== 训练完成 ===================="
find "$OUT" -maxdepth 3 -name "adapter_*" -o -maxdepth 3 -name "*.safetensors" | head -20
echo
echo "[NEXT] 记下最终 checkpoint 目录，第 3 步导出用："
find "$OUT" -maxdepth 2 -name "checkpoint-*" -type d | sort -V | tail -3

# ── 如果 session 断了要续训 ──────────────────────────────────────────────────
# 在 swift sft 命令末尾追加：
#   --resume_from_checkpoint /kaggle/working/out/dnc-router-v2-4b/vX-YYYYMMDD-HHMMSS/checkpoint-XXXX
# 依赖 working 目录持久化 —— 所以务必用 "Save & Run All (Commit)" 跑，别用交互式。
