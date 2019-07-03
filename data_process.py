import utils


def data_loader(args):
    train_data, train_labels = utils.get_raw_data(args.train_file)         # 获取一堆句子构成的列表
    val_data, val_labels = utils.get_raw_data(args.dev_file)

    args.catogories = ['EnterSports', 'Military', 'Economics', 'Technology', 'Government']
    args.cat_dict = dict(zip(args.catogories, range(len(args.catogories))))

    word_vocab, num_total_words = utils.build_dict(train_data)

    trainlabels_to_idx = [args.cat_dict[label] for label in train_labels]
    vallabels_to_idx = [args.cat_dict[label] for label in val_labels]

    train_data, train_labels = utils.encode(train_data, trainlabels_to_idx, word_vocab)
    val_data, val_labels = utils.encode(val_data, vallabels_to_idx, word_vocab)

    train_data = utils.pad_features(train_data, max_len=args.max_features)
    val_data = utils.pad_features(val_data, max_len=args.max_features)

    train_set = utils.batch(train_data.copy(), train_labels.copy(), args.batch_size)
    val_set = utils.batch(val_data.copy(), val_labels.copy(), args.batch_size)

    return train_set, val_set, num_total_words