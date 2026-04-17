#!/usr/bin/env python3
"""
快速配置 API Keys 到 params.json
用法：python setup_keys.py --deepseek_key "your_key_here"
"""
import json, argparse
from pathlib import Path

PARAMS_PATH = Path(__file__).parent / "videotrans" / "params.json"

KEY_MAP = {
    "deepseek_key":     "DeepSeek API Key (推荐翻译)",
    "chatgpt_key":      "OpenAI API Key",
    "chatgpt_api":      "OpenAI API Base URL (自定义时填写)",
    "claude_key":       "Claude API Key",
    "gemini_key":       "Gemini API Key",
    "openairecognapi_key": "OpenAI Whisper ASR API Key",
    "openairecognapi_url": "OpenAI ASR API Base URL (自定义时填写)",
}

def main():
    parser = argparse.ArgumentParser(description="配置 pyVideoTrans API Keys")
    for k, desc in KEY_MAP.items():
        parser.add_argument(f"--{k}", type=str, default=None, help=desc)
    parser.add_argument("--show", action="store_true", help="显示当前所有 key 配置（隐藏实际值）")
    args = parser.parse_args()

    data = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))

    if args.show:
        print("\n当前 API Key 配置状态：")
        for k, desc in KEY_MAP.items():
            val = data.get(k, "")
            status = "✅ 已配置" if val else "❌ 未配置"
            print(f"  {status}  {k:30s}  ({desc})")
        return

    updated = []
    for k in KEY_MAP:
        val = getattr(args, k, None)
        if val is not None:
            data[k] = val
            updated.append(k)

    if not updated:
        parser.print_help()
        return

    PARAMS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已更新 {len(updated)} 个配置项：{', '.join(updated)}")

if __name__ == "__main__":
    main()
