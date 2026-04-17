"""
火山声音复刻 2.0 凭证自检脚本
用法:
    .venv/bin/python tools/probe_doubao_clone.py

读取 videotrans/params.json 里的 volcenginetts_appid / volcenginetts_access,
用一次无害的 status 查询验证:
  - 返回 403 + "license not found"   → 产品未开通
  - 返回 200 + status/BaseResp        → 已开通 (speaker 不存在是正常)
  - 返回 其它 4xx                     → 鉴权 / 参数 错
"""
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
params = json.loads((ROOT / "videotrans" / "params.json").read_text(encoding="utf-8"))

APPID = params.get("volcenginetts_appid", "")
TOKEN = params.get("volcenginetts_access", "")
POOL = params.get("volcenginetts_clone_speaker_pool", [])
RESOURCE_ID = params.get("volcenginetts_clone_resource_id", "volc.megatts.voiceclone")

if not APPID or not TOKEN:
    raise SystemExit("volcenginetts_appid / volcenginetts_access 未填写")
if not POOL:
    raise SystemExit("volcenginetts_clone_speaker_pool 未填写")

headers = {
    "Authorization": f"Bearer;{TOKEN}",
    "Resource-Id": RESOURCE_ID,
    "Content-Type": "application/json",
}
# 用池子里第一个真实 speaker_id 探测
body = {"appid": APPID, "speaker_id": POOL[0]}
print(f"探测: AppID={APPID}, speaker_id={POOL[0]}, Resource-Id={RESOURCE_ID}")

resp = requests.post(
    "https://openspeech.bytedance.com/api/v1/mega_tts/status",
    json=body,
    headers=headers,
    timeout=15,
)
print(f"HTTP {resp.status_code}")
print(resp.text[:400])

if resp.status_code == 403 and "license" in resp.text.lower():
    print("\n❌ 声音复刻 2.0 产品未开通 → 去 https://console.volcengine.com/speech/service/10007 开通")
elif resp.status_code == 200:
    print("\n✅ 产品已开通，凭证有效")
elif resp.status_code in (401, 403):
    print("\n❌ 鉴权失败 → 检查 AppID / Access Token 是否复制完整")
else:
    print(f"\n⚠️  未预料的响应，把上面 body 发给我")
