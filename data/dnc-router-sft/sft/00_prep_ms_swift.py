#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 gen_dataset.py 产出的 sharegpt 数据集转成 ms-swift 的 messages 格式
=====================================================================

为什么需要这一步：
  gen_dataset.py 输出的是 **LLaMA-Factory 风格**的 sharegpt：
      {"system": "...", "conversations": [{"from": "human", "value": ".."},
                                          {"from": "gpt",   "value": ".."}]}

  而 Qwen3.5 的**第一方**微调框架是 ms-swift（官方有 Qwen3.5 Best Practices 文档，
  GatedDeltaNet 的 GDN 实现由它维护）。ms-swift 最稳的输入是 messages 风格 JSONL：
      {"messages": [{"role": "system", "content": ".."},
                    {"role": "user", "content": ".."},
                    {"role": "assistant", "content": ".."}]}

本脚本只做格式搬运，不改任何内容 —— 保证「训练数据 / Modelfile / Java 端」
三处 system prompt 逐字一致（P0 要求）。

用法（在 Colab / Kaggle notebook cell 里）：
    !python /kaggle/working/sft/00_prep_ms_swift.py \
        --in-dir  /kaggle/input/dnc-router-sft/dataset \
        --out-dir /kaggle/working/data

@author vigor  2026/04/20
"""

import argparse
import json
import sys
from pathlib import Path

ROLE_MAP = {
    "system": "system",
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant",
}
SYSTEM_FALLBACK = (
    "你是一名 DevOps 极速意图识别分发助手。请分析用户指令，以紧凑单行 JSON 格式提取核心意图 "
    "(action)、操作目标 (target) 与业务参数 (params)。只输出紧凑单行合法 JSON，"
    "禁止输出任何其他解释文字。"
)


def normalize_one(item: dict, default_system: str) -> dict | None:
    """
    把一条样本转成 ms-swift 的 messages 风格；无法识别则返回 None。

    同时支持工程里出现过的两种风格：
      A) messages 风格（devops_router_v2_*.json，本工程当前使用）
         {"messages":[{"role":"system"..},{"role":"user"..},{"role":"assistant"..}], "images":[]}
      B) sharegpt / from-value 风格（LLaMA-Factory 老格式，作兼容保留）
         {"system":"..","conversations":[{"from":"human","value":".."},...]}
    """
    system = item.get("system") or default_system

    # ── A) 已经是 messages 风格：只做规整与校验 ────────────────────────────
    if isinstance(item.get("messages"), list) and item["messages"]:
        msgs: list[dict] = []
        for turn in item["messages"]:
            if not isinstance(turn, dict):
                return None
            role = ROLE_MAP.get(str(turn.get("role", "")).lower())
            content = turn.get("content")
            if role is None or not isinstance(content, str) or not content.strip():
                return None
            # system 统一用 default_system，保证与训练数据逐字一致
            if role == "system":
                continue
            msgs.append({"role": role, "content": content.strip()})
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
            return None
        return {"messages": [{"role": "system", "content": system}, *msgs]}

    # ── B) from/value 风格：逐轮映射 ───────────────────────────────────────
    conv = item.get("conversations")
    if not isinstance(conv, list) or not conv:
        return None

    msgs = []
    for turn in conv:
        if not isinstance(turn, dict):
            return None
        role = ROLE_MAP.get(str(turn.get("from", "")).lower())
        content = turn.get("value")
        if role is None or not isinstance(content, str) or not content.strip():
            return None
        msgs.append({"role": role, "content": content.strip()})

    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None
    return {"messages": [{"role": "system", "content": system}, *msgs]}


def convert(src: Path, dst: Path, default_system: str) -> tuple[int, int]:
    raw = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(raw, dict):  # 兼容 {"data": [...]}
        raw = raw.get("data", [])
    if not isinstance(raw, list):
        raise ValueError(f"{src.name}: 顶层不是数组，无法识别")

    kept, dropped, seen = [], 0, set()
    for item in raw:
        one = normalize_one(item, default_system)
        if one is None:
            dropped += 1
            continue
        key = json.dumps(one["messages"], ensure_ascii=False, sort_keys=True)
        if key in seen:  # 二次去重，防生成阶段的句式坍缩
            dropped += 1
            continue
        seen.add(key)
        kept.append(one)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for one in kept:
            f.write(json.dumps(one, ensure_ascii=False) + "\n")
    return len(kept), dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="含 devops_router_v2_*.json 的目录")
    ap.add_argument("--out-dir", required=True, help="输出 messages 风格 jsonl 的目录")
    ap.add_argument("--train-name", default="devops_router_v2_train.json")
    ap.add_argument("--holdout-name", default="devops_router_v2_holdout.json")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    default_system = SYSTEM_FALLBACK

    # 优先用训练集里第一条样本的 system，保证与训练分布逐字一致
    train_src = in_dir / args.train_name
    if train_src.exists():
        try:
            probe = json.loads(train_src.read_text(encoding="utf-8"))
            if isinstance(probe, list) and probe:
                first = probe[0]
                found = first.get("system")
                if not isinstance(found, str):  # messages 风格：system 在 messages[0]
                    for turn in first.get("messages", []):
                        if isinstance(turn, dict) and turn.get("role") == "system":
                            found = turn.get("content")
                            break
                if isinstance(found, str) and found.strip():
                    default_system = found
                    print(f"[INFO] 采用数据集内 system prompt（{len(default_system)} 字符）")
        except Exception:  # noqa: BLE001
            pass

    ok_any = False
    for src_name, dst_name in (
        (args.train_name, "ms_swift_train.jsonl"),
        (args.holdout_name, "ms_swift_val.jsonl"),
    ):
        src = in_dir / src_name
        if not src.exists():
            print(f"[SKIP] 找不到 {src}", file=sys.stderr)
            continue
        kept, dropped = convert(src, out_dir / dst_name, default_system)
        print(f"[DONE] {src.name} -> {dst_name} | 保留 {kept} | 丢弃 {dropped}")
        ok_any = True

    if not ok_any:
        print("[FATAL] 没有任何输入文件被处理，检查 --in-dir 是否挂载正确", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
