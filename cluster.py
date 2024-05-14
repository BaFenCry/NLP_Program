from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
import matplotlib.pyplot as plt
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

def predict_text(text, kmeans):
    text=chinese_word_cut(text)
    text = [text]
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text))
    tfidf_weight = tfidf.toarray()
    svd = TruncatedSVD(5000)  # 降到5000维
    normalizer = Normalizer(copy=False)  # 标准化
    lsa = make_pipeline(svd,normalizer)
    X = lsa.fit_transform(vectorizer.fit_transform(text))
    tfidf = transformer.fit_transform(X)
    tfidf_weight = tfidf.toarray()
    cluster_label=kmeans.predict(tfidf_weight)

    # 返回预测结果
    return cluster_label

def chinese_word_cut(mytext):
    # 文本预处理 ：去除一些无用的字符只提取出中文出来
    new_data = re.findall('[\u4e00-\u9fa5]+', mytext, re.S)
    new_data = " ".join(new_data)

    # 文本分词
    seg_list_exact = jieba.cut(new_data, cut_all=True)
    result_list = []
    # 加载停用词库
    with open('data/stopword.txt', encoding='utf-8') as f: # 可根据需要打开停用词库，然后加上不想显示的词语
        stop_words = set()
        for i in f.readlines():
            stop_words.add(i.replace("\n", "")) # 去掉读取每一行数据的\n
    # 去除停用词
    for word in seg_list_exact:
        if word not in stop_words and len(word) > 1:
            result_list.append(word)      
    return " ".join(result_list)

def cluster_K_means(self):
    data = pd.read_csv(self.cur_file)
    data = data.drop(columns=['title'])
    data['segment'] = data['content'].apply(chinese_word_cut)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(data['segment']))
    tfidf_weight = tfidf.toarray()
    svd = TruncatedSVD(5000)  # 降到5000维
    normalizer = Normalizer(copy=False)  # 标准化
    lsa = make_pipeline(svd,normalizer)
    X = lsa.fit_transform(vectorizer.fit_transform(data['segment']))
    tfidf = transformer.fit_transform(X)
    tfidf_weight = tfidf.toarray()
    kmeans = KMeans(n_clusters=int(self.cur_K), random_state=42)
    kmeans.fit(tfidf_weight)
    print("inertia: {}".format(kmeans.inertia_))
    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(tfidf_weight)
    x = []
    y = []
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    data['label'] = kmeans.labels_
    # for flag, label in zip(data['flag'], data['label']):
    #     print(flag, label)
    data['flag'].replace(to_replace={1:2,2:3,3:0,4:1},inplace=True)
    right = 0
    error = 0
    for i,j in zip(data['flag'],data['label']):
        if i == j:
            right+=1
        else:
            error+=1
    report=classification_report(y_true=data['flag'], y_pred=data['label'])

    return [report,vectorizer, transformer, svd, normalizer, kmeans]