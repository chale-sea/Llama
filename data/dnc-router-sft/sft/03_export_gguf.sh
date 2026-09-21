#!/usr/bin/env bash
# =============================================================================
# 第 3 步 · 合并 LoRA → 转 GGUF → 量化 Q4_K_M
# =============================================================================
#
# ⚠️ 磁盘是这个脚本最大的坑：
#   /kaggle/working 只有 20GB，而 合并后 fp16 权重 ~8.5GB + f16 GGUF ~8.5GB
#   加起来就把 working 撑爆了。
#   → 所有中间产物一律写 /kaggle/temp（~60GB），最后只把 q4_k_m.gguf 拷回 working。
#
# 建议单独开一个 session 跑这一步（别和训练挤在同一个 9 小时窗口）。
#
# 用法：!bash /kaggle/working/sft/03_export_gguf.sh <adapter目录>
# =============================================================================
set -euo pipefail

ADAPTER="${1:-}"
if [[ -z "$ADAPTER" || ! -d "$ADAPTER" ]]; then
  echo "[FATAL] 用法：bash 03_export_gguf.sh /kaggle/working/out/dnc-router-v2-4b/vX-.../checkpoint-XXXX"
  echo "        可从 02_train.sh 结尾打印的列表里取。"
  exit 1
fi

export USE_MCORE_GDN=0
export HF_HOME=/kaggle/temp/hf

TEMP=/kaggle/temp/export
MERGED="$TEMP/merged"
GGUF_F16="$TEMP/dnc-router-qwen3.5-4b-f16.gguf"
GGUF_Q4="$TEMP/dnc-router-qwen3.5-4b-q4_k_m.gguf"
mkdir -p "$TEMP"

echo "==================== 1/4 合并 LoRA 到底座（fp16） ===================="
pip install -q -U "ms-swift>=4.1" transformers accelerate peft

swift export \
  --model Qwen/Qwen3.5-4B \
  --adapters "$ADAPTER" \
  --merge_lora true \
  --torch_dtype float16 \
  --output_dir "$MERGED"

echo "[INFO] 合并产物："
du -sh "$MERGED" 2>/dev/null || true
ls "$MERGED" | head -20

echo "==================== 2/4 编译 llama.cpp（CPU 工具即可） ===================="
cd "$TEMP"
if [[ ! -d llama.cpp ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
# -DGGML_CUDA=OFF 省掉 CUDA 编译时间（我们只要 convert / quantize 两个 CPU 工具）
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > "$TEMP/cmake.log" 2>&1
cmake --build build --config Release -j"$(nproc)" >> "$TEMP/cmake.log" 2>&1 || {
  echo "[FATAL] llama.cpp 编译失败，最后 40 行日志："
  tail -40 "$TEMP/cmake.log"
  exit 1
}
pip install -q -r requirements/requirements-convert_hf_to_gguf.txt

echo "==================== 3/4 转 GGUF（f16） ===================="
# 注意：Qwen3.5 是 GatedDeltaNet 混合架构，llama.cpp 必须足够新才认识。
#       如果这里报 unknown architecture / architecture not supported，
#       说明 clone 到的版本还没跟上 —— 见 TRAINING.md 第 3 节的处理办法。
python convert_hf_to_gguf.py "$MERGED" --outfile "$GGUF_F16" --outtype f16

echo "==================== 4/4 量化到 Q4_K_M ===================="
./build/bin/llama-quantize "$GGUF_F16" "$GGUF_Q4" Q4_K_M

echo
echo "==================== 落盘 ===================="
ls -lh "$GGUF_Q4"
cp "$GGUF_Q4" /kaggle/working/
# 顺手把 adapter 也打包（很小，几十 MB），方便以后换量化精度重导
tar -czf /kaggle/working/dnc-router-v2-4b-adapter.tar.gz -C "$(dirname "$ADAPTER")" "$(basename "$ADAPTER")" || true

echo
echo "[OK] 已在 /kaggle/working 下生成："
ls -lh /kaggle/working/*.gguf /kaggle/working/*.tar.gz 2>/dev/null || true
echo
cat <<'EOF'
下一步（二选一取回文件）：

  A) Kaggle 原生方式（推荐，简单）
     右上角 Save Version → Save & Run All (Commit)
     跑完后在 notebook 右侧 "Output" 面板里直接下载 .gguf
     注意：Output 有 500 个文件的上限，确认 working 里别堆中间产物。

  B) 推 HuggingFace Hub（要 Token，适合反复迭代）
     export HF_TOKEN=hf_xxx
     huggingface-cli upload <你的用户名>/dnc-router-v2-4b \
        /kaggle/working/dnc-router-qwen3.5-4b-q4_k_m.gguf . --repo-type model
     本地再：huggingface-cli download <用户名>/dnc-router-v2-4b \\
        dnc-router-qwen3.5-4b-q4_k_m.gguf --local-dir E:/code/dnc/agent/ml

  C) 顺便把 working 目录做成 Kaggle Dataset（跨 session 持久）
     Output 面板 → "Create Dataset" → adapter/gguf 即可在下次 session 当输入挂载
EOF
