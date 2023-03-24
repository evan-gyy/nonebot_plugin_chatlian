# ChatLian

AI莲莲，基于ChatGLM等对话模型实现

# 效果

莲莲不仅会与你亲密聊天，也会用表情包表达她的感受

<center class="half">
    <img src="resources/example2.jpg" width="180"/>
    <img src="resources/example3.jpg" width="200"/>
    <img src="resources/example1.jpg" width="150"/>
</center>

# 指令

```
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
```
# 使用

## 环境配置

```bash
pip install -r requirements.txt
```

## 对话

目前对话功能支持以下模型，请根据需要进行配置：

### ChatGLM

使用ChatGLM进行对话：需要本地部署，最低显存要求为6G，请从[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)下载模型，并修改`config.py`中的`chatglm_model_path`为模型存放路径。

```python
# config.py
chatglm_model_path: str = "/path/to/chatglm-6b"
```

### Yuan1.0

使用Yuan1.0进行对话：在[源1.0官网](https://air.inspur.com/home)申请API，并将账号和手机号填入`config.py`的对应位置。

```python
# config.py
yuan_account: str = "user"
yuan_phone: str = "phone"
```

## 表情包

莲莲可以根据对话内容回复表情包，请下载[预训练词向量](https://pan.baidu.com/s/1amvVC_Ue31KmBX0XVllMGA?pwd=j0q1)并将其放在`resources\data\`文件夹下。

# 更新

**2023/3/20**

1. 新增ChatGLM聊天功能
2. 支持切换对话模型、重置历史记录

**2023/1/20**

1. 新增莲莲表情包2.0

**2022/10/1**

1. 新增莲莲表情包AI，根据发送的指令寻找相似表情包
