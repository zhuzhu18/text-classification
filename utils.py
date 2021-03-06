import re
import numpy as np
from zhon.hanzi import punctuation
from collections import Counter


def get_raw_data(file_path):
    with open(file_path) as f:
        sentences = []
        labels = []
        for line in f:

            line = line.strip().split('\t')
            label, single_text = line[0], line[1]
            single_text = re.sub(r'[%s]'%punctuation, '', single_text)      # 去掉所有中文标点符号
            single_text = ['BOS'] + re.sub(r'[ ]{2,}', ' ', single_text).strip().split(' ') + ['EOS']   # 每句话的第一个分词是BOS,最后一个是EOS
            sentences.append(single_text)
            labels.append(label)

    return sentences, labels

def build_dict(sentences, max_words=10000):
    word_count = Counter()
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1
    rank_words = word_count.most_common(max_words)        # 出现次数最多的50000个分词
    num_total_words = len(rank_words)
    word_dict = {word[0]:index+1 for index, word in enumerate(rank_words)}
    word_dict['UNK'] = 0

    return word_dict, num_total_words

def encode(sentences, labels, word_dict, sort_by_len=True):
    length = len(sentences)
    out_seqs = []
    for i in range(length):
        seq = [word_dict[w] if w in word_dict else 0 for w in sentences[i]]       # 将每句话按单词字典中的索引构建成序列
        out_seqs.append(seq)
    def len_argsort(seqs):
        return sorted(range(len(seqs)), key=lambda x: len(seqs[x]))       # 将句子序列按长度从小到大排序的索引
    if sort_by_len:
        sorted_index = len_argsort(out_seqs)
        out_seqs = [out_seqs[index] for index in sorted_index]          # 将句子序列按长度从小到大排序
        labels = [labels[index] for index in sorted_index]              # 将每个句子序列的标签按句子的长度排序

    return out_seqs, labels

def pad_features(features, max_len, pad=0):
    padded_features = np.zeros((len(features), max_len), dtype=np.int64)
    for i in range(len(features)):
        if len(features[i]) >= max_len:
            padded_features[i] = features[i][:max_len]
        else:
            padded_features[i, :len(features[i])] = features[i]
    return padded_features

def batch(dataset, labels, batch_size, shuffle=True):
    datasize = len(dataset)
    batch_start = np.arange(0, datasize, batch_size)
    data_index = np.random.permutation(range(datasize)) if shuffle else np.arange(datasize)

    for i in batch_start[:-1]:
        batch_index = [data_index[j] for j in range(i, i+batch_size)]
        yield np.array([dataset[m] for m in batch_index]), np.array([labels[m] for m in batch_index])
    yield np.array([dataset[data_index[m]] for m in range(batch_start[-1], datasize)]), \
          np.array([labels[data_index[m]] for m in range(batch_start[-1], datasize)])

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

