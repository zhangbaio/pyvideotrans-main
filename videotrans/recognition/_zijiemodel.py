# zh_recogn 识别
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

import requests
from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang
from videotrans.recognition._base import BaseRecogn
from videotrans.util import tools

RESOURCE_ID = "volc.bigasr.auc_turbo"

_error={
"20000003":"静音音频",



"45000010":"鉴权失败：无效的 X-Api-Key。新版控制台请在第一框填写 APP Key，并清空第二框；旧版控制台请填写 APP ID 和 Access Token。",
"45000001":"请求参数缺失必需字段 / 字段值无效",
"45000002":"空音频",
"45000151":"音频格式不正确",

"550XXXX":"服务内部处理错误",
"55000031":"服务器繁忙"
}

@dataclass
class ZijieRecogn(BaseRecogn):

    def __post_init__(self):
        super().__post_init__()

    def _build_headers(self, *, task_id, appid, token):
        headers = {
            "X-Api-Resource-Id": RESOURCE_ID,
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1",
        }
        if token:
            headers["X-Api-App-Key"] = appid
            headers["X-Api-Access-Key"] = token
        else:
            headers["X-Api-Key"] = appid
        return headers

    def _raise_api_error(self, response):
        code = response.headers.get("X-Api-Status-Code", "")
        message = response.headers.get("X-Api-Message", "")
        logid = response.headers.get("X-Tt-Logid", "")
        details = []
        if code:
            details.append(f"code={code}")
        if message:
            details.append(f"message={message}")
        if logid:
            details.append(f"logid={logid}")
        if response.status_code:
            details.append(f"http={response.status_code}")

        err = _error.get(str(code), "")
        if response.status_code == 403:
            err = err or "VolcEngine STT authorization failed. Please check API Key/AppID, Access Token, and resource permission."
        if not err:
            err = "VolcEngine STT request failed"
        if details:
            err = f"{err} ({', '.join(details)})"
        try:
            body = response.text
            if body:
                err += f"\n{body[:500]}"
        except Exception:
            pass
        raise RuntimeError(err)

    def _exec(self) -> Union[List[Dict], None]:
        if self._exit():  return

        submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
        task_id = str(uuid.uuid4())
        appid = params.get('zijierecognmodel_appid', '').strip()
        token = params.get('zijierecognmodel_token', '').strip()
        if not appid and token:
            appid, token = token, ''
        if appid.isdigit() and not token:
            raise RuntimeError("检测到旧版数字 APP ID，请同时填写旧版控制台里的 Access Token，不要填写 Secret Key。")
        headers = self._build_headers(task_id=task_id, appid=appid, token=token)
        request = {
            "user": {
                "uid": appid
            },
            "audio": {"data": self._audio_to_base64(self.audio_file)},
            "request": {
                "model_name": "bigmodel",
                "model_version": "400",
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": True,
                "show_utterances": True,
                # "vad_segment":True,
                # "end_window_size":300,
                "enable_speaker_info": True
            }
        }
        # print(request)

        response = requests.post(submit_url, json=request, headers=headers)
        logger.info(f'{response=}')
        logger.info(f'{response.headers=}')
        code = response.headers.get('X-Api-Status-Code')
        if response.status_code >= 400 or not code:
            self._raise_api_error(response)
        if str(code) != "20000000":
            self._raise_api_error(response)

        res = response.json()
        seg_list = res.get('result', {}).get('utterances')
        if not seg_list:
            raise RuntimeError(f'返回数据中无识别结果:{response=}')

        srt_list = []
        speaker_list=[]
        srt_strings=""

        for it in seg_list:
            if not it.get('text','').strip():
                continue
            speaker_list.append(f'spk{it.get("additions", {}).get("speaker", 0)}')
            startraw = tools.ms_to_time_string(ms=it['start_time'])
            endraw = tools.ms_to_time_string(ms=it['end_time'])
            tmp={
                "line": len(srt_list) + 1,
                "start_time": it['start_time'],
                "end_time": it['end_time'],
                "startraw": startraw,
                "endraw": endraw,
                "text": it['text'].strip()
            }
            srt_list.append(tmp)
            srt_strings+=f"{tmp['line']}\n{startraw} --> {endraw}\n{tmp['text']}\n\n"

        self._signal(
            text=srt_strings,
            type='replace_subtitle'
        )
        if speaker_list:
            Path(f'{self.cache_folder}/speaker.json').write_text(json.dumps(speaker_list), encoding='utf-8')
        return srt_list


