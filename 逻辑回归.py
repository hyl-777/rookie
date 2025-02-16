import numpy as np
import pandas as pd
import re
import spacy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors

# 数据清洗函数


def raw_text_cleaner(text):
    text = re.sub(r'<[^>]+>', '', text)      # 移除html标签
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)      # 删除特殊符号（保留字母数字和基本标点）
    text = re.sub(r'\s+', ' ', text).strip()      # 合并多余空格
    return text

# 处理停用词函数


def process_text(text):
    # 创建Spacy文档对象
    doc = nlp(text)

    # 分词 + 停用词过滤 + 词形还原
    processed_tokens = [
        token.lemma_.lower()  # 词形还原并转小写
        for token in doc
        if not token.is_stop   # 过滤停用词
           and not token.is_punct  # 过滤标点
           and not token.is_space  # 过滤空格
           and len(token.lemma_) > 1  # 过滤单字符
    ]

    return processed_tokens


# 读取信息
df = pd.read_csv('F:/AI/YoutubeCommentsDataSet.csv', encoding='utf-8')

nlp = spacy.load("en_core_web_sm")

# 处理标签列，此处只做二分类，删除了为中性的评论
df['Sentiment'] = df['Sentiment'].replace({'positive': 1, 'negative': 0, 'neutral': -1}).astype(int)
df = df[df['Sentiment'] != -1]

# 处理缺失值：删除缺失行
df = df.dropna(subset=['Comment'])
# 应用清洗函数
df['Comment'] = df['Comment'].apply(raw_text_cleaner)

# spacy分词过程
tqdm.pandas(desc="Spacy处理进度")

df['processed_tokens'] = df['Comment'].progress_apply(process_text)

# 删除空行
df = df[df['processed_tokens'].apply(len) > 0]

# 保存结果
df.to_csv('spacy_processed_comments.csv', index=False)


# ========================
# 1. 加载预训练词向量
# ========================
def load_word_vectors():
    # 加载预训练词向量
    glove_path = "F:/AI/glove.6B.50d.txt"
    print("正在加载词向量...")

    word_vectors = KeyedVectors.load_word2vec_format(glove_path,
                                                     binary=False,
                                                     no_header=True)
    print(f"加载完成，词表大小：{len(word_vectors)}")
    return word_vectors


# 全局词向量实例
word_vectors = load_word_vectors()


# ========================
# 2. 文本向量化处理
# ========================
def tokens_to_vector(tokens):
    # 将分词列表转换为词向量平均值
    vectors = []
    for token in tokens:
        if token in word_vectors:  # 处理存在的词
            vectors.append(word_vectors[token])
        else:  # 处理未登录词（OOV）
            pass  # 也可添加随机初始化或零向量

    if len(vectors) == 0:  # 空处理保护
        return np.zeros(word_vectors.vector_size)

    return np.mean(vectors, axis=0)


# 应用向量化
df['features'] = df['processed_tokens'].apply(tokens_to_vector)


class CommentDataset(Dataset):  # 继承PyTorch的Dataset基类
    def __init__(self, features, labels):
        # 将特征列表转换为FloatTensor，并堆叠为二维张量
        self.features = torch.FloatTensor(np.vstack(features))

        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        # 返回数据集总样本数（以标签数量为准）
        return len(self.labels)

    def __getitem__(self, idx):
        # 根据索引返回单个样本的特征和标签
        return self.features[idx], self.labels[idx]


# ========================
# 4. 逻辑回归模型定义
# ========================
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# ========================
# 5. 主流程
# ========================

if __name__ == "__main__":
    # 假设数据中包含标签列'label'（0/1）
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df['features'].tolist(),        # 特征列表（词向量平均值）
        df['Sentiment'].tolist(),              # 标签列表（需确保已编码为0/1值）
        test_size=0.2,              # 训练测试8：2
        random_state=42
    )

    # 创建数据加载器
    batch_size = 64             # 单次处理样本量
    train_dataset = CommentDataset(X_train, y_train)
    test_dataset = CommentDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型
    input_dim = word_vectors.vector_size            # 词向量维度
    model = LogisticRegression(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ========================
    # 训练循环
    # ========================
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()           # 训练模式
        total_loss = 0

        for features, labels in train_loader:
            features = features.to(device)       # 数据送至GPU/CPU
            labels = labels.to(device)

            # 前向传播
            outputs = model(features).squeeze()         # 压缩维度 [64,1]→[64]
            loss = criterion(outputs, labels)       # 计算损失

            # 反向传播
            optimizer.zero_grad()       # 消除冗余维度
            loss.backward()             # 计算梯度
            optimizer.step()            # 根据梯度更新权重W和偏置b

            total_loss += loss.item()

        # 打印训练进度
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # ========================
    # 模型评估
    # ========================
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features).squeeze()
            predicted = (outputs > 0.5).float()          # 概率>0.5判为正类
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "text_classifier.pth")       # 保存模型权重而非整个模型，便于后续加载和推理。

    # 加载模型方式
    # model = LogisticRegression(input_dim)
    # model.load_state_dict(torch.load("text_classifier.pth"))

