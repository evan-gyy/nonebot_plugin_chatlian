import asyncio
import json
import os
import platform
import random
import re
import sys
import time
import traceback
from pathlib import Path
from queue import PriorityQueue as PQ

import gensim
import jieba
from nonebot import on_command, on_keyword
from nonebot.adapters.onebot.v11 import Bot, Event, MessageSegment, Message, MessageEvent, GroupMessageEvent
from nonebot.log import logger
from nonebot.params import CommandArg
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from .inspurai import Yuan, set_yuan_account, Example
from .config import config

dir_path = Path(__file__).parent / "resources"
IMG_PATH = str((dir_path / "meme").absolute()) + "/"
RECORD_PATH = str((dir_path / "record").absolute()) + "/"
DATA_PATH = str((dir_path / "data").absolute()) + "/"

__zx_plugin_name__ = "随机莲莲"
__plugin_usage__ = """
usage：
    随机莲莲
    指令：
        莲莲：发送莲莲表情包/聊天
        莲莲骂?[对象]：发送莲莲藏话
        狗叫 ?[长/短]：发送莲莲狗叫
        爱国：发送莲莲爱国表情
        罕见：发送莲莲罕见语音
        
        示例：随机莲莲
        示例：莲莲骂我
        示例：狗叫 长
""".strip()
__plugin_des__ = "随机莲莲"
__plugin_cmd__ = ["莲莲/莲莲骂/爱国/罕见/狗叫"]
__plugin_type__ = ("群内小游戏",)
__plugin_version__ = 0.1
__plugin_author__ = "evan-gyy"
__plugin_settings__ = {
    "level": 5,
    "default_status": True,
    "limit_superuser": False,
    "cmd": __plugin_cmd__,
}
__plugin_block_limit__ = {"rst": "我知道你很急，但你先别急"}

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
model_path = "/mnt/libra/gyy/chatglm/chatglm-6b"
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

dxl = on_keyword({"莲莲", "罕见", "爱国"}, priority=5, block=True)
dxl_gj = on_command("狗叫", priority=5, block=True)
dxl_zh = on_command("莲莲骂", priority=4, block=True)

show_mode = on_command("查看模型", priority=5, block=True)
change_mode = on_command("切换模型", priority=5, block=True)

reset = on_command("重置历史", priority=5, block=True)
load_persona = on_command("载入人格", priority=5, block=True)
del_persona = on_command("删除人格", priority=5, block=True)


@dxl.handle()
async def _(bot: Bot, event: MessageEvent):
    global chat_info
    text = event.get_plaintext().strip()
    chat_id = str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)
    if "罕见" in text:
        file = random_file(RECORD_PATH + '/hj')
        await dxl.finish(MessageSegment.record(file))

    elif "爱国" in text:
        file = random_file(IMG_PATH, 'ag\d+')
        await dxl.finish(MessageSegment.image(file))

    elif "莲莲" in text:
        if text == "莲莲":
            file = random_file()
            await dxl.finish(MessageSegment.image(file))
        else:
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
                    response, conv, time_delta = await loop.run_in_executor(None, ask_glm, text, per + conv)
                    logger.info(f"回答: {response}s")
                    logger.info(f"耗时: {time_delta}s")
                    # logger.info(f"历史句数量: {len(conv)}")
                    await dxl.send(response, at_sender=True)

                except Exception as e:
                    logger.error(str(e))
                    traceback.print_exc()

                conv = conv[1:] if len(conv) > max_ctx else conv
                chat_info[chat_id]['history'] = conv

            file = similar_meme(text.replace("莲莲", ""), 3)
            await dxl.send(MessageSegment.image(file))


@dxl_gj.handle()
async def _(bot: Bot, event: Event, args: Message = CommandArg()):
    text = args.extract_plain_text().strip().split()
    path = RECORD_PATH + '/gj'
    file = random_file(path, '.*')

    if text:
        if "长" in text:
            file = random_file(path, 'long\d+')
        elif "短" in text:
            file = random_file(path, '\d+')

    await dxl_gj.finish(MessageSegment.record(file))


@dxl_zh.handle()
async def _(bot: Bot, event: Event):
    record = random_file(RECORD_PATH + '/zh')
    await dxl_zh.finish(MessageSegment.record(record))


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


def similar_meme(sentence, n=3):
    vector_path = DATA_PATH + 'sgns.weibo.bigram-char.bin'
    wv = gensim.models.KeyedVectors.load(vector_path, mmap='r')

    with open(DATA_PATH + 'meme.json', 'r', encoding='utf-8') as f:
        meme = json.load(f)
    with open(DATA_PATH + 'cn_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().split()

    cut = [token for token in jieba.lcut(sentence) if token not in stopwords]
    logger.info(f"分词结果：{cut}")

    try:
        sims = PQ()
        for k, v in meme.items():
            sm = wv.n_similarity(cut, jieba.lcut(v))
            sims.put([1 - sm, k])

    except ZeroDivisionError:
        logger.error("分词结果为空或匹配失败，随机发送表情")
        return random_file(IMG_PATH)

    res = [sims.get() for _ in range(n)]
    logger.info(f"相似度Top3: {res[:3]}")

    if 1 - res[0][0] > 0.5 and res[1][0] > 0.5:
        return f"file:///{IMG_PATH}/" + res[0][1] + '.jpg'
    else:
        return f"file:///{IMG_PATH}/" + random.choice(res[:3])[1] + '.jpg'


def random_file(path=IMG_PATH, regex='\d+', end='\.\w+'):
    file_list = os.listdir(path)
    match_list = []

    for file in file_list:
        match = re.match(f'{regex}{end}', file)
        if match:
            match_list.append(file)

    return f"file:///{path}/{random.choice(match_list)}"


def trans_mark(str):
    # E_pun = u'，。！？【】（）《》“‘：；［］｛｝&，．？（）＼％－＋￣~＄#＠=＿、／'
    # C_pun = u',.!?[]()<>"\':;[]{}&,.?()\\%-+~~$#@=_//'
    E_pun = u'，。！？“‘：；'
    C_pun = u',.!?"\':;'
    table = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
    return str.translate(table)
