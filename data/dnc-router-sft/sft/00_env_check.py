#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle 环境体检脚本（第 0 步，先跑这个）
========================================

在 Colab / Kaggle notebook 里第一个 cell 执行：

    !python /kaggle/working/sft/00_env_check.py

检查项：
1. GPU 型号与显存（T4 / P100 / 有没有真的分配到卡）
2. bf16 硬件支持 —— T4/P100 都不支持，训练必须走 fp16
3. 磁盘：/kaggle/working（20GB 持久化）vs /kaggle/temp（~60GB 不持久）
4. 网络是否打开（拉 Qwen 权重必需）
5. transformers / ms-swift 版本，是否认识 qwen3_5 架构

@author vigor  2026/04/20
"""

import os
import shutil
import subprocess
import sys

PASS, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def df(path: str) -> str:
    if not os.path.isdir(path):
        return "目录不存在"
    try:
        total, used, free = shutil.disk_usage(path)
        return f"总 {total/2**30:.1f}GB / 已用 {used/2**30:.1f}GB / **可用 {free/2**30:.1f}GB**"
    except Exception as e:  # noqa: BLE001
        return f"<读取失败: {e}>"


def main() -> int:
    fatal = 0

    print("=" * 68)
    print("1) GPU")
    print("=" * 68)
    try:
        import torch

        print(f"  torch            : {torch.__version__}")
        print(f"  cuda available   : {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print(f"  {FAIL} 没有可用 GPU —— 去 notebook 右侧 Settings → Accelerator 选 GPU")
            fatal += 1
        else:
            print(f"  device count     : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(
                    f"  GPU[{i}]           : {p.name} | VRAM {p.total_memory/2**30:.1f}GB "
                    f"| SM {p.major}.{p.minor}"
                )
            # bf16 支持判定：SM >= 8.0 (Ampere) 才有原生 bf16
            major = torch.cuda.get_device_properties(0).major
            if major >= 8:
                print(f"  {PASS} 支持 bf16（SM {major}.x）")
            else:
                print(
                    f"  {WARN} 不支持原生 bf16（SM {major}.x 是 Turing/Pascal）\n"
                    f"         → 训练必须用 --torch_dtype float16，不要用 bfloat16"
                )
    except ImportError:
        print(f"  {FAIL} 没有 torch —— 这个 notebook 的镜像不对")
        fatal += 1

    print()
    print("=" * 68)
    print("2) 磁盘账本（最容易踩的坑）")
    print("=" * 68)
    print(f"  /kaggle/working  : {df('/kaggle/working')}   ← 持久化，上限 20GB")
    print(f"  /kaggle/temp     : {df('/kaggle/temp')}   ← 不持久，用完即弃")
    print(f"  /kaggle/input    : {df('/kaggle/input')}   ← 只读挂载，不占额度")
    print()
    print("  提醒：合并后的 fp16 权重 ~8.5GB + f16 GGUF ~8.5GB，放 working 会爆；")
    print("        所有中间产物一律写 /kaggle/temp，只把最终 q4_k_m.gguf 拷回 working。")

    print()
    print("=" * 68)
    print("3) 网络")
    print("=" * 68)
    code = sh("curl -s -o /dev/null -w '%{http_code}' -m 10 https://huggingface.co")
    if code == "200":
        print(f"  {PASS} 外网可达（huggingface.co -> 200）")
    else:
        print(
            f"  {FAIL} 外网不可达（http_code={code}）\n"
            f"         → notebook 右侧 Settings → Internet → 打开（需要先手机验证账号）"
        )
        fatal += 1

    print()
    print("=" * 68)
    print("4) 关键依赖")
    print("=" * 68)
    for pkg in ("transformers", "peft", "accelerate", "bitsandbytes", "ms_swift", "swift"):
        try:
            mod = __import__(pkg)
            print(f"  {pkg:<16}: {getattr(mod, '__version__', 'n/a')}")
        except ImportError:
            print(f"  {pkg:<16}: 未安装")

    print()
    try:
        import transformers

        print(f"  transformers 版本: {transformers.__version__}")
        print(
            "  提示：若 ms-swift 报 'qwen3_5 not supported'，先升级：\n"
            "        pip install -U 'ms-swift>=4.1' transformers"
        )
    except Exception:  # noqa: BLE001
        pass

    print()
    print("=" * 68)
    if fatal:
        print(f"体检结论：有 {fatal} 项硬性阻塞，先修掉再往下走。")
        return 1
    print("体检结论：环境可用于后续步骤。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
