"""
Author: J0hNnY1ee j0h1eenny@gmail.com
Date: 2025-01-08 22:49:06
LastEditors: J0hNnY1ee j0h1eenny@gmail.com
LastEditTime: 2025-01-09 15:19:27
FilePath: /implement_sft_bert/dataGenerate/Qwen2.5_7B.py
Description: 

Copyright (c) 2025 by J0hNnY1ee j0h1eenny@gmail.com, All Rights Reserved. 
"""

from transformers import AutoModelForCausalLM, AutoTokenizer


# 初始化模型和分词器
model_name = "model/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
systemPrompt = ""


def generate_response_with_context(
    prompt, history, temperature=1.2, top_p=0.9, top_k=50, max_new_tokens=512
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
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 转换为模型输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成响应
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.2,
        num_return_sequences=1
    )

    # 移除输入部分的 token，仅保留生成的内容
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # 解码生成的内容
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 将模型的响应加入对话历史
    history.append({"role": "assistant", "content": response})

    return response, history


# 测试封装函数
if __name__ == "__main__":
    file_path = "dataGenerate/augment_data.txt" 
    # 初始化对话历史
    conversation_history = []
    systemPrompt = """
    你是一个数据生成专家，非常擅长生成数据，而且会遵循指令。
    下面是1条示例数据：
    - Total:1113 感动:196 愤怒:42 搞笑:201 难过:19 新奇:339 震惊:316	52 岁 男子 随身带 弹弓 灭鼠   自称 战绩 超 200 只 ( 图 ) 自称 曾 是 “ 神枪手 ”   唐志忠 自诩 ， 练 了 10 多天 他 就 掌握 一些 门道 了 ， 而 这些 和 他 在 野战部队 的 经历 分不开 。 1981 年 当兵 ， 唐志忠 曾经 在 某 野战部队 呆 过 6 年 ， 在 一个 高射 机枪 连 里面 任 班长 。 “ 我们 那个 时候 用 的 是 14.5 毫米 的 子弹 ， 射击 1000 米外 的 坦克 、 碉堡 、 飞机 。 ” 唐志忠 说 ， 他们 所在 的 班 得 过 集体 三等功 ， 而 他 也 在 一次 步枪射击 比赛 中 ， 获得 过 “ 神枪手 ” 的 称号 。   在 《 水浒传 》 第 69 回中 ， 没 羽箭 张清以 “ 飞石 打 英雄 ” 的 方式 惊艳 亮相 ， 他 也 和 戴宗 、 花荣 等 人 一起 荣膺 “ 梁山泊 五绝 ” 。   在 羊 犀 立交 旁 的 百仁 菜市场 ， 52 岁 的 唐志忠 也 在 苦练 类似 “ 绝技 ” ， 已经 练 了 三年 ， 他 要 对付 的 是 市场 里 的 老鼠 。 比起 张清 的 伸手 就 打 ， 他 还 需要 一个 弹弓 、 一颗 钢珠 。 白天 是 鸡 贩子 ， 晚上 是 打鼠 人 ， 每天 清晨 ， 他 的 鸡笼 里 都 会 摆 上 几只 鲜血淋漓 的 老鼠 ， 多数 都 是 头部 穿孔 。   杀鼠者   七八米 外 射中 打火机   晚上 8 点 ， 百仁 菜市场 灯光 昏黄 ， 人 已经 散 去 。 路灯 下 ， 一个 干瘦 的 影子 被拉得 老长 ， 影子 不动 了 ， 空旷 的 市场 响起 “ 砰 ” 的 一声 ， 紧接着 ， 是 “ 吱 ” 一声 惨叫 … …   每天晚上 ， 唐志忠 都 要 在 市场 里 一圈 一圈 转悠 ， 嘴上 叼 着 卷烟 ， 一只 手 插 在 西裤 裤兜 里 ， 风吹 起 polo 衫 ， 唐志忠 显得 更加 干瘦 。 一旦 发现 老鼠 ， 他 就 摸 出 弹弓 ， 拉开 、 瞄准 、 撒手 ， 三四秒 的 时间 ， 钢珠 就 “ 盯 着 ” 老鼠 去 了 。   在 菜市场 干 了 七八年 ， 唐志忠 每天 都 能 见到 老鼠 上蹿下跳 ， 肆无忌惮 。 “ 大白天 都 敢 从 人 的 面前 过 。 ” 他 说 ， 这些 家伙 根本 不怕 人 ， 到 了 晚上 ， 菜市场 就是 它们 的 地盘 。 三年 前 ， 唐志忠 看到 市场 外有 卖 弹弓 的 ， 他 灵机一动 就 买 了 一把 ， 准备 用 这些 “ 杀器 ” 对付 老鼠 。   这 以后 ， 唐志忠 的 西裤 裤兜 总是 鼓鼓囊囊 ， 一块 磁铁 ， 上面 吸满 了 豌豆 大小 的 钢珠 ， 腰后 衣服 一 撩 ， 皮带 上 的 不锈钢 弹弓 就 闪闪发光 。 拔弓 、 装弹 ， 唐志忠 把 自己 打扮 得 跟 黑猫 警长 一样 ， 跟 市场 里 的 老鼠 打起 了 持久战 。   挽弓 当 挽强 ， 出手 就 爆头 。 唐志忠 笑了笑 ， “ 打头 一下 就要 搞定 ， 打 心脏 要 打 两三下 ！ ”   昨   日 ， 唐志忠 拿出 一个 打火机 挑战 自己 ， 站 在 七八米 外 ， 唐志忠 拉 了 三下 ， 最终 将 打火机 击中 ， 他 有些 不好意思 ， 歪着头 盯 着 远处 ， “ 还是 有点 紧张 。 ” 他 又 转身 找 了   一只 老鼠 ， 隔 着 两重 铁网 ， 钢珠 穿过 ， 只 听 老鼠 “ 吱 ” 了 一声 。 “ 唉 ！ 这下 没 打中 头 ！ ” 唐志忠 回到 门市 ， 笼子 里装 了 四五只 老鼠 ， “ 昨晚 发挥 得 就 比较 好 。 ”   老唐 称 ：   三年 杀 了 200 多只 老鼠   熟能生巧 ， 唐志忠 的 精确度 也 是 钢珠 喂 出来 的 。 一 开始 ， 他 把 市场 上卖 剩下 的 南瓜 放在 墙上 打 。 “ 这个 时候 主要 是 摸索 姿势 。 ” 他 站 在 门市 前 ， 把 弓拉到 眼眶 下 ， 抵住 面颊 ， “ 三点一线 ， 手要 摆正 。 ” 而 如果 跪姿 射击 ， 弹弓 就 可以 横着 打 。   南瓜 打 得 瓜瓤 横飞 ， 能够 准确 命中 后 ， 唐志忠 又 摆出 了 红牛 罐 。 每天 中午 吃完饭 ， 他 的 门前 就是 “ 砰砰 砰 ” 的 声音 。 “ 每天 训练 一个 小时 。 ” 看着 丈夫 打 得 不亦乐乎 ， 妻子 杨志琼 在 一旁 笑 得 合不拢嘴 ， “ 让 他 打 嘛 ， 只要 不 影响 卖鸡 的 生意 。 ”   同是 卖 鸡 的 商贩 谢先生 说 ， 老唐 只要 看 得到 的 东西 ， 就 能够 打 到 ， 最 开始 练习 的 时候 ， 还是 打坏 过 一些 东西 ， 椅子 、 盆子 都 被 打坏 过 。   如今 ， 唐志忠 在 菜市场 打出 了 名 ， 只要 一问 姓 唐 的 打 老鼠 的 人 ， 商贩 就 会 指 唐志忠 的 店面 。 “ 每天 他 都 要 打 两三只 。 ” 市场 里 的 谢先生 介绍 ， 市场 中午 过后 就 没什么 生意 了 ， 到 了 晚上 8 点 就 没什么 人 了 ， 老唐 还是 不会 走 ， 在 市场 里 转悠 ， “ 这个 时候 就是 老鼠 横行 的 高峰 。 ”   据 唐志忠 自己 介绍 ， 除了 假期 ， 他 都 不会 离开 市场 ， 三年 的 时间 ， 差不多 射杀 了 200 多只 老鼠 。   唐志忠 表示 自己 还会 继续 练习 ， “ 毕竟 现在 只能 定点 狙击 ， 如果 老鼠 跑 起来 还是 打不着 。 ” 唐志忠 说 ， 以后 的 练习 将 改成 射击 硬币 ， 提高 自己 的 精准度 。   成都 商报 记者   宦小淮   摄影记者   王勤
    请在每生成的数据开始加上一个 '- ' 的标记，以便于后续处理。
    示例问答：
    - 问：请生成一条感动的数据：
    - 答：Total:1804 感动:888 愤怒:45 搞笑:123 难过:391 新奇:26 震惊:331 爱心市民 摄氏零下四十度雪夜 三次抢救冰封车内老人　感人事迹　 　近日，在吉林省延吉市一个村庄外的大街上发生了一起让人十分揪心的事情：天气温度降至摄氏零下四十多度的时候，一位老人被困在一辆严重被冰雪覆盖的小轿车内。幸运的是，经过三位热心市民的努力，最终将她安全解救了出来，并进行了紧急救助。此次行动虽然只是普通的民间之举，却足以彰显出了无私救援的精神力量。据现场目击者回忆，当晚九点半左右，三人骑摩托车前往村里看望亲友。正当他们准备返途离开之际，看到远处街道上有闪烁不定的灯光。“那时感觉情况不对劲！”其中一人说道，“因为当时外面风大，而且气温极低，那盏灯应该不会持续出现如此长时间。”于是，三人加快速度赶到那里查看原因。只见车窗内的人已昏迷过去，整个车身被厚厚的雪花包裹，周围没有其他人迹。“我们迅速靠近车子并砸碎了一块玻璃”，另一名参与者讲述道，“然后爬进了车厢”。但由于严寒加之门窗关闭已久时间较长等原因使得营救工作异常困难——尽管已经打开暖风系统试图取暖却没有立即见效。最终，经过大约二十分钟紧张而又耐心的合作，大家合力成功将老人抬出冰冷的驾驶舱，随即立即将其送至最近医院接受检查和治疗。目前这位女士状态平稳并无生命危险迹象。参与这次救援任务的所有人都表示：“当我们看到她还保持着原来姿势蜷缩在那里,那种心疼之情真是难以言语表达”。这一感人故事体现了人性中最温暖的一面—即使是在极端恶劣条件下也会毫不犹豫给予陌生之人关爱援助精神值得所有人敬佩赞赏！
    - 问：请生成一条难过的数据：
    - 答：Total:372 感动:7 愤怒:19 搞笑:51 难过:238 新奇:12 震惊:45	女子 低头 专心 玩 手机 坠河 溺亡 原 标题 ： 专注 低头 看 手机 ， 她 落水 溺亡   走 在 路上 ， 坐在 车里 ， 大家 都 是 低头 摆弄 手机 的 “ 低头 族 ” ， 可 有人 却 因此 失去 了 生命 。   2015 年 12 月 29 日晚 ， 28 岁 的 王 某 一边 玩 手机 ， 一边 在 浙江 温州 平阳县 鳌 江镇 厚 垟 村 的 河边 散步 。   厚 垟 村 四周 河流 环绕 ， 村民 沿岸 而居 。 她 经过 的 河道 约 四五米 宽 ， 水面 距离 路面 不足 1 米 ， 周遭 都 没有 栏杆 阻隔 。 因为 水质 不错 ， 村民 们 平时 都 在 河边 洗衣服 。   因为 她 一直 在 看 手机 ， 没 注意 到 河 ， 一下 掉 了 进去 ， 再也 没 上来 。   视频 记录 整个 落水 过程   警方 提供 的 监控 视频 显示 ， 20 时 22 分 36 秒 ， 王某 走进 画面 ， 此时 她 还 走 在 村道 中央 偏右 的 位置 ， 距离 河道 还有 约 2 米 的 距离 。 她 一直 在 看 手机 ， 手指 不停 地 在 屏幕 上 滑动 ， 丝毫 没有 察觉到 行走 的 方向 已经 偏离 ， 越来越 靠近 河道 。   20 时 22 分 53 秒 ， 她 右脚 踩 空 ， 整个 人斜 着 掉 进 了 河里 ， 溅 起 大片 的 水花 ， 手机 在 磕 到 岸边 后 也 落入 水 里 。 她 的 头部 在 水面 上 ， 双手 使劲 在 水里 挥动 ， 挣扎 着想 游 向 对岸 的 台阶 。   画面 中 ， 她 的 头 在 水中 和 水面 上 反复 出现 ， 显然 呛 了 不少 水 ， 可 距离 上 几乎 没有 改变 。   20 时 24 分 左右 ， 她 似乎 改变 了 主意 ， 朝刚 掉落 的 岸边 划 去 ， 连续 举高 了 双臂 。 8 秒钟 后 ， 当 她 再 一次 举 高 双臂 准备 发力 时 ， 整个 人 后仰 倒 在 了 水里 。   20 时 24 分 23 秒 ， 王某 消失 在 画面 中 ， 水面 恢复 了 平静 。   整个 挣扎 的 过程 持续 了 约 90 秒 ， 其间 始终 没 人 经过 该 路段 。 从 监控 看 ， 周边 的 住户 都 大门 紧闭 ， 二楼 也 没有 灯光 。   老公 发现 河面 漂着 妻子 鞋子   记者 从平阳 警方 处 了解 到 ， 王某 出 生于 1987 年 ， 贵州 人 ， 丈夫 杨某 在 鳌 江 打工 多年 ， 有 两个 未满 10 岁 的 孩子 ， 在 当地 念书 。 当晚 ， 丈夫 临时 加班 ， 王某 在 电话 中说 先去 附近 逛 一下 再 回家 。   杨某 下班 没 见到 妻子 ， 还 以为 是 去 老乡 家 玩 了 ， 没想到 王某 一夜 未归 。 次日 ， 他 在 四处寻找 时 ， 发现 河面上 漂着 妻子 的 鞋子 ， 才 报 了 警 。 家属 透露 ， 王某 平时 吃饭 、 睡觉时 也 都 习惯 低头 玩 手机 。   “ 其实 河水 并 不 深 ， 像 我 一米 八 的 个头 ， 河水 差不多 到 胸口 的 位置 。 ” 平阳 公安 的 周 警官 告诉 我 ， 村民 反映 河里 淤泥 很深 ， 掉进去 后 可能 会因 打滑 无法 站 直 。   警方 提醒 ： 走路 时 千万 不要 玩 手机 ， 不仅 对 眼睛 不好 ， 同时 也 容易 影响 人们 对 周围 事物 的 感知力 ， 让 人们 无法 正确 判断 周围环境 的 安全性 ， 一旦 遭遇 意外 ， 后果 不堪设想 。   据 《 都市快报 》   手机 悲剧   2015 年 10 月 24 日 下午 3 点 ， 浙江 义乌 福田 街道 阳光 大道 尚经村 路段 ， 一名 男子 被 一辆 浦江 牌照 货车 当场 撞死 ， 目击者 称 死者 过 马路 时正用 手机 看电视 。   2015 年 11 月 28 日 下午 ， 南京 鼓楼区 的 一处 铁路 天桥 上 ， 41 岁 的 王 某 ， 酒后 一边 玩 手机 一边 走路 ， 不慎 踏空 滚 下 天桥 台阶 身亡 。   2015 年 5 月 13 日 下午 ， 广东 中山 坦洲 镇 十四 村三华 百货 对 开 路口 发生 一起 惨烈 交通事故 ， 路边 监控 拍下 了 事发 经过 ： 一名 穿 短裙 的 年轻 女子 右手 接听 手机 ， 当 她 过 马路 时 ， 一辆 白色 货车 迅速 将 其 撞倒 在 地 ， 随即 又 被 一辆 泥头车 辗轧 过去 。   责任编辑 ： 苏 未然   SN226
    注意要满足以下要求：
    - 使用简体中文
    - 尽量保证数据的随机性，不要生成重复的数据，且必须模仿示例数据的格式。
    - 尽量让数据看起来像真实的数据，你用了虚拟人物名称也不用提示，要确保数据像真的数据。
    - 数据必须用感动:xx 愤怒:xx 搞笑:xx 难过:xx 新奇:xx 震惊:xx 开始
    - 让你生成感动的数据则，感动的值最大，让你生成愤怒的数据则，愤怒的值最大，以此类推。
    """

    # 开始连续对话
    # user_prompt = """

    # """

    # 生成模型响应
    emotions = ["感动", "愤怒", "搞笑", "难过", "新奇", "震惊"]
    for _ in range(1000):
        conversation_history = []
        for e in emotions:
            user_prompt = f"请生成一条{e}的数据："
            result, conversation_history = generate_response_with_context(
                user_prompt, conversation_history
            )
            
            
            print("Qwen:", result)
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(result + "\n")