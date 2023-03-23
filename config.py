#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pydantic import BaseModel, Extra
from nonebot import get_driver
from pathlib import Path

resource_path = Path(__file__).parent / "resources"


class Config(BaseModel, extra=Extra.ignore):
    # ChatGLM模型路径
    chatglm_model_path: str = ""
    # OpenAI的API_KEY
    openai_api_key: str = ""
    # 申请源1.0API的账号名与手机号
    yuan_account: str = ""
    yuan_phone: str = ""
    # 个性化
    yuan_nick_name: str = "莲莲"
    yuan_persona_path: str = resource_path / "bot_info.txt"
    yuan_example_path: str = resource_path / "bot_example.txt"
    # 模型设置
    yuan_engine: str = "dialog"
    yuan_max_tokens: int = 100
    yuan_topK: int = 3
    yuan_topP: int = 0.9
    yuan_input_prefix: str = f"问：“"
    yuan_input_suffix: str = "”"
    yuan_output_prefix: str = f"{yuan_nick_name}答：“"
    yuan_output_suffix: str = "”"
    yuan_append_output_prefix_to_query: bool = True
    yuan_frequencyPenalty: float = 1.2
    yuan_responsePenalty: float = 1.2
    yuan_noRepeatNgramSize: float = 2


driver = get_driver()
config = Config.parse_obj(driver.config)
