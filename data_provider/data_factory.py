
from torch.utils.data import DataLoader, ConcatDataset

def data_provider(args, flag):
    from data_provider.data_loader import Dataset_PEMS, Dataset_Custom, Dataset_Weather

    data_dict = {
        'PEMS07': Dataset_PEMS,
        'PEMS03': Dataset_PEMS,
        'PEMS04': Dataset_PEMS,
        'PEMS08': Dataset_PEMS,
        'WTH': Dataset_Weather
    }

    # timeenc = 0 if args.embed != 'timeF' else 1
    timeenc = 0

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True

    drop_last = False
    batch_size = args.batch_size
    freq = args.freq


    if args.zero_shot and flag == 'test':
        Data = data_dict.get(args.target_data, Dataset_Custom)
        data_set = [Data(
                args = args,
                root_path=args.target_root_path,
                data_path=args.target_data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            )]
    else:
        data_set = []
        for idx, data in enumerate(args.data):
            Data_class = data_dict.get(data, Dataset_Custom)
            data_set.append(Data_class(
                args = args,
                root_path=args.root_path[idx],
                data_path=args.data_path[idx],
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            ))

    print(flag, sum([len(dataset) for dataset in data_set]))
    data_loader = [DataLoader(
        data_set[i],
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    ) for i in range(len(data_set))]
    return data_set, data_loader
