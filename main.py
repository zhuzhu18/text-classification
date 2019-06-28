from config import Configure
import utils

config = Configure()
args = config.get_args()

train_data, labels = utils.get_raw_data(args.train_file)         # 获取一堆句子构成的列表

args.catogories = ['EnterSports', 'Military', 'Economics', 'Technology', 'Government']
args.cat_dict = dict(zip(args.catogories, range(len(args.catogories))))

word_vocab, num_total_words = utils.build_dict(train_data)
labels_to_idx = [args.cat_dict[label] for label in labels]

train_data, labels = utils.encode(train_data, labels_to_idx, word_vocab)


print(len(train_data[0]))       # 第0句话的长度,19个词
print(len(train_data[10]))      # 第10句话的长度,32个词
print(len(train_data[20]))
print(len(train_data[30]))
print(len(train_data[40]))
print(len(train_data[50]))
print(len(train_data[-1]))      # 最后一句话的长度,9294个词

print(len(train_data))          # 总的句子数,2500