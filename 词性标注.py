import spacy

# 导入英文词典文件，创建nlp分析对象
nlp = spacy.load("en_core_web_sm")

# 创建一个分析例句的文档对象
doc = nlp("lets not forget that apple pay in 2014 required a brand new iphone in order to use it a significant portion")

for token in doc:
    print(token.text,   # 文本：原始的单词文本
          token.lemma_,   # 词干化：单词的基本形式
          token.pos_,   # 词性标注：简单的通用词性标签
          token.tag_,   # 更详细的词性标注
          token.dep_,   # 语法依赖：即标记之间的关系
          token.shape_,   # 形状：单词形状——大写、标点、数字。
          token.is_alpha,   # 该token是一个alpha字符吗
          token.is_stop)   # 这个token是否在停止词列表中
