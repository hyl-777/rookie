import numpy as np
import pandas as pd
import re

# 数据清洗函数


def raw_text_cleaner(text):
    text = re.sub(r'<[^>]+>', '', text)      # 移除html标签
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)      # 删除特殊符号（保留字母数字和基本标点）
    text = re.sub(r'\s+', ' ', text).strip()      # 合并多余空格
    return text


# 读取信息
df = pd.read_csv('F:/AI/YoutubeCommentsDataSet.csv', encoding='utf-8')

# 处理缺失值：删除缺失行
df = df.dropna(subset=['Comment'])
# 应用清洗函数
df['cleaned'] = df['Comment'].apply(raw_text_cleaner)

# 保存结果
df.to_csv('cleaned_data.csv', index=False, encoding='utf-8-sig')
