'''
Author: J0hNnY1ee j0h1eenny@gmail.com
Date: 2025-01-16 18:46:55
LastEditors: J0hNnY1ee j0h1eenny@gmail.com
LastEditTime: 2025-01-16 18:51:17
FilePath: /implement_sft_bert/验证.py
Description: 

Copyright (c) 2025 by J0hNnY1ee j0h1eenny@gmail.com, All Rights Reserved. 
'''
import json

# 加载预测文件和答案文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 比较预测和答案
def compare_predictions(predictions, answers):
    correct = 0
    total = len(predictions)
    incorrect_ids = []  # 用于存储不正确的新闻ID
    
    for pred, ans in zip(predictions, answers):
        # 确保比较的是同一个新闻ID
        if pred['id'] == ans['id']:
            # 找到预测和答案中的最大情绪，排除 'id', 'news', 'Total'
            pred_max_emotion = max(
                (key for key in pred.keys() if key not in ['id', 'news', 'Total']),
                key=lambda x: pred[x]
            )
            ans_max_emotion = max(
                (key for key in ans.keys() if key not in ['id', 'news', 'Total']),
                key=lambda x: ans[x]
            )
            
            # 如果最大情绪一致，则认为是正确的
            if pred_max_emotion == ans_max_emotion:
                correct += 1
            else:
                incorrect_ids.append(pred['id'])  # 记录不正确的新闻ID
        else:
            print(f"警告: 新闻ID {pred['id']} 和 {ans['id']} 不匹配，跳过比较。")
    
    # 计算准确率
    accuracy = correct / total
    return accuracy, incorrect_ids

# 文件路径
predictions_file = 'data/train.json'
answers_file = 'data/train_ori.json'

# 加载数据
predictions = load_json(predictions_file)
answers = load_json(answers_file)

# 比较并计算准确率
accuracy, incorrect_ids = compare_predictions(predictions, answers)
print(f'准确率: {accuracy * 100:.2f}%')
print(f'不正确的新闻ID: {incorrect_ids}')