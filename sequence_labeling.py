from HMM import *
from evaluate import *

def hidden_markov(self):
    param = train('data/HMMTrainSet.txt')
    text=self.input_edit.toPlainText()
    article = [text]
    # print(article)
    article_partition, tag_sequence = word_partition(param, article)
    # 打印分词结果及相应的状态标签序列
    HMM_print(article_partition,tag_sequence)
    gold_standard_file='gold_standard.txt'
    predicted_file='predicted.txt'
    # 将标准答案和预测结果转换为BMES标签格式
    convert_to_BMES(gold_standard_file, "BMES_tags.txt")
    convert_to_BMES(predicted_file, "test-xulie.txt")


    # 读取BMES标签格式的文件
    test_seq = read_file("test-xulie.txt")
    standard_seq = read_file("BMES_tags.txt")
    evaluate_HMM(test_seq,standard_seq)