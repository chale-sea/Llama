dnc-router · qwen3.5-4b  线上微调上传包
========================================

本包用于在 Kaggle / Colab 上微调 Qwen3.5-4B 为 DevOps 意图路由模型。
本地对应位置：E:\code\dnc\agent\ml\qwen3.5-4b\

包内结构
--------
  dataset/
    devops_router_v2_train.json     9000 条训练集（15.7MB）
    devops_router_v2_holdout.json   1000 条评测集（1.75MB，本地评测用，云端不参与训练）
    dataset_info.json               LLaMA-Factory 格式的数据集注册清单
  sft/
    00_env_check.py                 环境体检（GPU / bf16 / 磁盘 / 网络 / 依赖版本）
    00_prep_ms_swift.py             sharegpt -> ms-swift messages JSONL（含二次去重）
    01_smoke_test.sh                冒烟闸门：验证 GDN 架构能否被 QLoRA 训练（先过这关）
    02_train.sh                     正式训练（Qwen3.5-4B QLoRA，rank 32 / alpha 64）
    03_export_gguf.sh               合并 LoRA -> f16 GGUF -> Q4_K_M 量化

数据分布（五类均衡，与 Java 端 CloudEdgeIntentRouter 的消费分支严格对齐）
------------------------------------------------------------------------
  build          3000    打包出包（target: jenkins）
  review         3000    代码检视（target: gitlab）
  query          2000    知识库查询（target: knowledge）
  rejected       1000    安全拒绝（target: security）—— 提示注入 / 越权 / 危险命令
  chat_fallback  1000    脱域闲聊（target: none）
  合计          10000    训练 9000 + 评测 1000（按类分层切分）

怎么用（两条命令的差别只在于"摆目录"）
--------------------------------------
【Kaggle】把本 zip 作为 Dataset 上传（Private），在 notebook 里：
    !mkdir -p /kaggle/working/sft /kaggle/temp
    !unzip -o -q /kaggle/input/dnc-router-sft/dnc-router-sft.zip -d /kaggle/temp/unpack
    !cp -r /kaggle/temp/unpack/dnc-router-sft/sft/. /kaggle/working/sft/
  之后所有脚本命令一致。

【Colab】上传到 /content 后：
    !mkdir -p /kaggle/working /kaggle/temp /kaggle/input
    !unzip -o -q dnc-router-sft.zip -d /kaggle/input/
  Colab 的 notebook 默认 root 权限，可以直接在根下建 /kaggle/... 目录，
  于是 5 个脚本一行都不用改，两家平台命令完全一致。

详细步骤见：doc/guide/SFT_微调全流程实战SOP手册_qwen3.5-4b.md

注意
----
* Internet 默认关闭，需要在 notebook 的 Settings 里手动打开（Kaggle 需手机验证）。
* /kaggle/working 只有 20GB，导出环节的中间产物（合并权重 ~8.5GB、f16 GGUF ~8.5GB）
  一律写 /kaggle/temp（约 60GB），只把最终的 q4_k_m.gguf 拷回 working。
* 长任务请用 Save Version -> Save & Run All (Commit) 后台跑，别用交互式 cell 挂机。
