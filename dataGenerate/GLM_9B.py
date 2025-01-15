import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化 GLM 模型和分词器
model_name = "THUDM/glm-4-9b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

# 系统提示
systemPrompt = """
你是一个数据生成专家，非常擅长生成数据，而且会遵循指令。
请在每生成的数据开始加上一个 '- ' 的标记，以便于后续处理。
示例问答：
- 问：请生成一条感动的数据：
- 答：Total:1804 感动:888 愤怒:45 搞笑:123 难过:391 新奇:26 震惊:331 爱心市民 摄氏零下四十度雪夜 三次抢救冰封车内老人　感人事迹　 　近日，在吉林省延吉市一个村庄外的大街上发生了一起让人十分揪心的事情：天气温度降至摄氏零下四十多度的时候，一位老人被困在一辆严重被冰雪覆盖的小轿车内。幸运的是，经过三位热心市民的努力，最终将她安全解救了出来，并进行了紧急救助。此次行动虽然只是普通的民间之举，却足以彰显出了无私救援的精神力量。据现场目击者回忆，当晚九点半左右，三人骑摩托车前往村里看望亲友。正当他们准备返途离开之际，看到远处街道上有闪烁不定的灯光。“那时感觉情况不对劲！”其中一人说道，“因为当时外面风大，而且气温极低，那盏灯应该不会持续出现如此长时间。”于是，三人加快速度赶到那里查看原因。只见车窗内的人已昏迷过去，整个车身被厚厚的雪花包裹，周围没有其他人迹。“我们迅速靠近车子并砸碎了一块玻璃”，另一名参与者讲述道，“然后爬进了车厢”。但由于严寒加之门窗关闭已久时间较长等原因使得营救工作异常困难——尽管已经打开暖风系统试图取暖却没有立即见效。最终，经过大约二十分钟紧张而又耐心的合作，大家合力成功将老人抬出冰冷的驾驶舱，随即立即将其送至最近医院接受检查和治疗。目前这位女士状态平稳并无生命危险迹象。参与这次救援任务的所有人都表示：“当我们看到她还保持着原来姿势蜷缩在那里,那种心疼之情真是难以言语表达”。这一感人故事体现了人性中最温暖的一面—即使是在极端恶劣条件下也会毫不犹豫给予陌生之人关爱援助精神值得所有人敬佩赞赏！
注意要满足以下要求：
- 使用简体中文
- 尽量保证数据的随机性，不要生成重复的数据，且必须模仿示例数据的格式。
- 尽量让数据看起来像真实的数据，你用了虚拟人物名称也不用提示，要确保数据像真的数据。
- 数据必须用感动:xx 愤怒:xx 搞笑:xx 难过:xx 新奇:xx 震惊:xx 开始
- 让你生成感动的数据则，感动的值最大，让你生成愤怒的数据则，愤怒的值最大，以此类推。
"""

def generate_response_with_context(
    prompt, history, temperature=1.2, top_p=0.9, top_k=1, max_new_tokens=512
):
    """
    生成模型的响应内容，支持上下文记忆。

    Args:
        prompt (str): 用户输入的内容。
        history (list): 对话历史，包含之前的消息。
        temperature (float): 随机性控制参数。
        top_p (float): 核采样参数。
        top_k (int): 限制采样范围。
        max_new_tokens (int): 最大生成的 token 数量。

    Returns:
        str: 模型生成的响应内容。
        list: 更新后的对话历史。
    """
    # 将当前对话加入到历史中
    history.append({"role": "user", "content": prompt})

    # 构造模型输入的消息格式
    messages = [{"role": "system", "content": systemPrompt}] + history

    # 应用聊天模板
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    # 生成响应
    gen_kwargs = {
        "max_length": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 将模型的响应加入对话历史
    history.append({"role": "assistant", "content": response})

    return response, history


# 测试封装函数
if __name__ == "__main__":
    file_path = "dataGenerate/augment_data.txt"
    emotions = ["感动", "愤怒", "搞笑", "难过", "新奇", "震惊"]

    # 开始连续对话
    for _ in range(1000):
        conversation_history = []
        for e in emotions:
            user_prompt = f"请生成一条{e}的数据："
            result, conversation_history = generate_response_with_context(
                user_prompt, conversation_history
            )
            
            print("GLM:", result)
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(result + "\n")