import asyncio
import os
import platform
import sys
import time
import traceback
import nonebot
from nonebot import on_command, on_keyword
from nonebot.adapters.onebot.v11 import Bot, Event, MessageSegment, Message, MessageEvent, GroupMessageEvent
from nonebot.log import logger
from nonebot.params import CommandArg
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .inspurai import Yuan, set_yuan_account, Example
from .config import config
from .common import *


dir_path = Path(__file__).parent / "resources"
IMG_PATH = str((dir_path / "meme").absolute()) + "/"
RECORD_PATH = str((dir_path / "record").absolute()) + "/"
DATA_PATH = str((dir_path / "data").absolute()) + "/"

__zx_plugin_name__ = "ChatLian"
__plugin_usage__ = """
usage：
    ChatLian
    指令：
        莲莲 [对话]：与莲莲对话
        莲莲骂?[对象]：发送莲莲藏话
        爱国：发送莲莲爱国表情
        罕见：发送莲莲罕见语音
        重置历史：重置对话历史
        查看模型/切换模型：查看/切换对话模型
        载入人格/删除人格：载入/删除莲莲人格
    示例：莲莲 介绍下自己
    示例：莲莲骂他
""".strip()
__plugin_des__ = "莲莲"
__plugin_cmd__ = ["莲莲/莲莲骂/爱国/罕见"]
__plugin_type__ = ("群内小游戏",)
__plugin_version__ = 0.1
__plugin_author__ = "evan-gyy"
__plugin_settings__ = {
    "level": 5,
    "default_status": True,
    "limit_superuser": False,
    "cmd": __plugin_cmd__,
}
__plugin_cd_limit__ = {
    "cd": 2,
    "rst": "莲莲思考中，请稍候再试"
}

# Yuan1.0
set_yuan_account(config.yuan_account, config.yuan_phone)

yuan = Yuan(engine=config.yuan_engine,
            max_tokens=config.yuan_max_tokens,
            topK=config.yuan_topK,
            topP=config.yuan_topP,
            input_prefix=config.yuan_input_prefix,
            input_suffix=config.yuan_input_suffix,
            output_prefix=config.yuan_output_prefix,
            output_suffix=config.yuan_output_suffix,
            append_output_prefix_to_query=config.yuan_append_output_prefix_to_query,
            frequencyPenalty=config.yuan_frequencyPenalty,
            responsePenalty=config.yuan_responsePenalty,
            noRepeatNgramSize=config.yuan_noRepeatNgramSize)

with open(config.yuan_persona_path, 'r', encoding='utf-8') as f:
    persona = f.read()
    yuan.add_example(Example(inp=persona, out="\n"))
    yuan.add_example(Example(inp="你是谁呀", out="我是莲莲呀！"))
    yuan.add_example(Example(inp="你喜欢我吗", out="我最喜欢你了！我的主人！"))
    logger.info("bot_info loaded successfully")

# ChatGLM
model_path = config.chatglm_model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
chatglm = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
chatglm = chatglm.eval()
os_name = platform.system()

with open(config.yuan_example_path, 'r', encoding='utf-8') as f:
    persona = [line.strip() for line in f.readlines()]
    examples = [(persona[i], persona[i + 1]) for i in range(len(persona) // 2)]

model_list = ['Yuan1.0', 'ChatGLM-6B']
chat_info = {}
max_ctx = 10

dxl = on_command("莲莲", priority=5, block=True)
dxl_kw = on_keyword({"罕见", "爱国"}, priority=5, block=True)

show_mode = on_command("查看模型", priority=5, block=True)
change_mode = on_command("切换模型", priority=5, block=True)

reset = on_command("重置历史", priority=5, block=True)

load_persona = on_command("载入人格", priority=5, block=True)
del_persona = on_command("删除人格", priority=5, block=True)


@dxl.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    global chat_info
    text = args.extract_plain_text().strip()
    chat_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)

    if not text:
        file = random_file()
        await dxl.finish(MessageSegment.image(file))

    elif text.startswith("骂"):
        record = random_file(RECORD_PATH + '/zh')
        await dxl.finish(MessageSegment.record(record))

    else:
        text = '莲莲，' + text
        logger.info(f"问：{text}")
        if chat_id not in chat_info:
            chat_init(chat_id)

        mode = chat_info[chat_id]['mode']
        conv = chat_info[chat_id]['history']
        per = chat_info[chat_id]['examples']

        if mode.lower() in 'Yuan1.0'.lower():
            try:
                msg = yuan.submit_API(prompt=text, trun="”").replace(',', '，')
                logger.info(f"答：{msg}")
                await dxl.send(msg)

            except ValueError as val_err:
                logger.error(str(val_err))
                await dxl.send("返回为空，请修改问题后重试")

            except AttributeError as att_err:
                logger.error(str(att_err))
                await dxl.send("连接超时，请检查网络后重试")

            except Exception as e:
                logger.error(str(e))

        elif mode.lower() in 'ChatGLM-6B'.lower():
            try:
                loop = asyncio.get_event_loop()
                response, conv, time_delta = await loop.run_in_executor(None, ask_glm, text,
                                                                        conv if conv else per + conv)
                logger.info(f"回答: {response}s")
                logger.info(f"耗时: {time_delta}s")
                await dxl.send(response, at_sender=True)

            except Exception as e:
                logger.error(str(e))
                traceback.print_exc()

        conv = conv[1:] if len(conv) > max_ctx else conv
        logger.info(f"历史句数量: {len(conv)}")
        chat_info[chat_id]['history'] = conv


    file = similar_meme(text.replace("莲莲", ""), logger, 3)
    await dxl.send(MessageSegment.image(file))


@dxl_kw.handle()
async def _(bot: Bot, event: Event):
    text = event.get_plaintext().strip()
    if "罕见" in text:
        file = random_file(RECORD_PATH + '/hj')
        await dxl.send(MessageSegment.record(file))
    if "爱国" in text:
        file = random_file(IMG_PATH, 'ag\d+')
        await dxl.send(MessageSegment.image(file))


@reset.handle()
async def _(bot: Bot, event: MessageEvent):
    chat_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)
    if chat_id in chat_info:
        chat_info[chat_id]['history'] = []
        await reset.send('已重置历史')
    else:
        await reset.send('无历史记录')


@show_mode.handle()
async def _(bot: Bot, event: MessageEvent):
    chat_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)
    if chat_id in chat_info:
        await show_mode.send(f"当前模型: {chat_info[chat_id]['mode']}")
    else:
        await show_mode.send(f"未指定模型")


@change_mode.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    global chat_info
    text = args.extract_plain_text().strip()

    chat_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)
    if chat_id not in chat_info:
        chat_init(chat_id)

    if text in model_list:
        chat_info[chat_id]['mode'] = text
        logger.info(f"{chat_id}切换模型为{text}")
        await change_mode.send(f"切换模型为 {text}")
    else:
        model_mode_msg = '\n'.join(model_list)
        logger.info(text)
        logger.info(f"{chat_id}当前模型为{chat_info[chat_id]['mode']}")
        await change_mode.send(f"支持的模型: \n{model_mode_msg}")


@load_persona.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    global chat_info
    chat_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)
    if chat_id not in chat_info:
        chat_init(chat_id)
    chat_info[chat_id]['examples'] = examples
    await load_persona.send('载入成功')


@del_persona.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    global chat_info
    chat_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)
    if chat_id not in chat_info:
        chat_init(chat_id)
    chat_info[chat_id]['examples'] = []
    await del_persona.send('删除成功')


def chat_init(chat_id):
    global chat_info
    chat_info[chat_id] = {
        'mode': model_list[-1],
        'examples': examples,
        'history': []
    }
    logger.info(chat_info[chat_id])
    logger.info(f"{chat_id}模型初始化成功")


def ask_glm(msg, conv):
    start = time.time()
    response, history = chatglm.chat(tokenizer, msg, history=conv)
    return trans_mark(response), history, time.time() - start
