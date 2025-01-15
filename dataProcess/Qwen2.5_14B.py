'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-01-13 19:09:40
LastEditors: J0hNnY1ee j0h1eenny@gmail.com
LastEditTime: 2025-01-13 23:21:01
FilePath: /implement_sft_bert/dataProcess/Qwen2.5_14B.py
Description: 

Copyright (c) 2025 by error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git, All Rights Reserved. 
'''
"""
Author: J0hNnY1ee j0h1eenny@gmail.com
Date: 2025-01-13 19:09:40
LastEditors: J0hNnY1ee j0h1eenny@gmail.com
LastEditTime: 2025-01-13 19:54:13
FilePath: /implement_sft_bert/dataProcess/Qwen2.5_14B.py
Description: 

Copyright (c) 2025 by J0hNnY1ee j0h1eenny@gmail.com, All Rights Reserved. 
"""
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer




# 封装生成对话的函数
def generate_response(prompt, model, tokenizer, max_new_tokens=2048):
    sys_prompt = """你是一个评论大师，你读完一篇新闻后能准确的表达自己的情感。
    假设感动=0,愤怒=1,搞笑=2,难过=3,新奇=4,震惊=5。你会以一个特定的身份读完一篇新闻后用一个数字来表达自己的情感，注意你只回答一个数字，没有其他任何话。
    下面是一个例子：
    问：你是一个18-29岁之间的男性，教育水平为大学，你看到了下面的新闻，你会怎么表达自己的情感？
    - 新闻:大学教授无偿支教山村，点亮孩子的未来 
    答：0。
    - 新闻:女 大学生 校内 死亡   生前 接 神秘电话 ( 图 ) 原 标题 ： 延大 西安 创新 学院 一 女生 校内 死亡   生前 接 神秘电话   最新消息 ：   1 月 2 日 9 时 ， @ 陕西 都市快报   最新消息 ， 延安大学 西安 创新 学院 离奇 死亡 女 大学生 ， 确认 系 他杀 ， 凶嫌 已 被 抓获 ， 系 被害 女子 男朋友 。   
    答：3。
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成响应
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 加载模型和分词器（放在函数外部）
model_name = "model/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)




ages = ["18-29", "30-39", "40-49", "50-59", "60-69"]
genders = ["男", "女"]
educations = ["初中及以下", "高中", "本科", "硕士及以上"]
emotions = ["movbed", "angry", "funny", "sad", "novel", "shocked"]

# 初始化数据
def initdata(data):
    for emotion in emotions:
        data[emotion] = 0
    data["Total"] = 0

# 处理单个数据
# 处理单个数据
def process_single(item, model, tokenizer):
    for age in ages:
        for gender in genders:
            for education in educations:
                prompt = f"你是一个{age}岁之间的{gender}性，教育水平为{education}，你看到了下面的新闻，你会怎么表达自己的情感？\n - 新闻：{item['news']}"
                response = generate_response(prompt, model, tokenizer)

                # 去除多余字符（空格、换行符等）
                response = response.strip()

                # 打印调试信息（包括类型和长度）
                print(f"响应：{response}（类型：{type(response)}，长度：{len(response)}）")

                # 处理 response
                if response.isdigit():  # 如果 response 是文本形式的数字
                    index = int(response)  # 转换为整数
                    if 0 <= index < len(emotions):  # 检查索引是否有效
                        emotion = emotions[index]
                        item[emotion] += 1
                        item["Total"] += 1
                    else:
                        print(f"无效的情绪索引: {response}")
                elif response in emotions:  # 如果 response 是情绪名称
                    item[response] += 1
                    item["Total"] += 1
                else:
                    print(f"无效的情绪响应: {response}")  # 打印无效响应以便调试

# 打开输入文件
with open("dataGenerate/output_updated.json", encoding="utf-8") as f:
    data = json.load(f)

# 打开输出文件（第一次写入时使用 "w" 模式）
output_file = "dataProcess/processed_output.json"
first_item = True

# 使用 tqdm 添加进度条
for item in tqdm(data, desc="处理进度", unit="条"):
    initdata(item)  # 初始化数据
    process_single(item, model, tokenizer)  # 处理数据

    # 将处理后的数据保存到输出文件
    with open(output_file, "a" if not first_item else "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=4)
        f.write("\n")  # 每条数据占一行

    first_item = False  # 第一次写入后，切换为追加模式