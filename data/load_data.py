from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from data.preprocess_earthquakes import process_data_earth
from data.preprocess_citibike import process_data_bike, download
from data.preprocess_covid19 import process_data_covid
from data.toy_data.toy_dataset import process_data_pinwheel


def data_loading(event_num, event_out, dataset, event_num_per, class_num, seqs):
    path = Path(__file__).parents[0]
    if dataset == 'earthquake':
        mean, std = process_data_earth(path / 'California.txt', event_num=event_num, seqs=seqs)  # 'variable'
        data_path1 = path / 'earthquakes_calif.npz'
        dim = 3
    elif dataset == 'covid19':
        mean, std = process_data_covid(path / 'us-counties.csv', event_num=event_num)  # 'variable'
        data_path1 = path / 'covid_nj_cases.npz'
        dim = 2
    elif dataset == 'citibike':
        mean, std = process_data_bike(event_num=event_num, dl = False)  # 'variable'
        data_path1 = path / 'citibike.npz'
        dim = 2
    elif dataset == 'pinwheel':
        mean, std = process_data_pinwheel(path / 'toy_data/hawkes.txt', event_num_per = event_num_per, event_num = event_num, num_classes = class_num, seqs = seqs)  # 'variable'  #50
        data_path1 = path / 'pinwheel.npz'
        dim = 2
    data = np.load(data_path1)
    files = data.files
    dataset_input = [data[file][:-event_out, :] for file in files]
    dataset_output = [data[file][-event_out:, :] for file in files]

    all_ds = [data[file][:, 1:dim+1] for file in files]

    print(f'we have {len(dataset_input)} number of sequences where each sequence has {dataset_input[0].shape[0]} length and {dataset_input[0].shape[1]} features')

    ds_NF = all_ds[0]
    for i in range(len(all_ds)-1):
        i += 1
        ds_NF = np.append(ds_NF, all_ds[i], axis=0)

    ds_NF_scaled = StandardScaler().fit_transform(ds_NF)

    return dataset_input, dataset_output, ds_NF_scaled, mean, std


def data_generation(event_num, event_out, dataset, event_num_per, class_num,  seqs , shauffled = False, batch_size=64):

    dataset_input, dataset_output, ds_NF, mean, std = data_loading(event_num, event_out, dataset, event_num_per, class_num, seqs)
    '''cache_path = Path(__file__).parents[0] / 'cache_dir'
    cache_path.mkdir(exist_ok=True)
    if not (cache_path / f'{dataset}_input.npy').is_file():
        dataset_input, dataset_output, ds_NF, mean, std = data_loading(event_num, event_out, dataset)
        np.save(cache_path / f'{dataset}_input.npy', np.stack(dataset_input))
        np.save(cache_path / f'{dataset}_output.npy', np.stack(dataset_output))
        np.save(cache_path / f'{dataset}_ds_nf.npy', np.stack(ds_NF))
        mean.to_pickle(cache_path / f'{dataset}_mean.csv')
        std.to_pickle(cache_path / f'{dataset}_std.csv')
    dataset_input = np.load(cache_path / f'{dataset}_input.npy')
    dataset_output = np.load(cache_path / f'{dataset}_output.npy')
    ds_NF = np.load(cache_path / f'{dataset}_ds_nf.npy')
    mean = pd.read_pickle(cache_path / f'{dataset}_mean.csv')
    std = pd.read_pickle(cache_path / f'{dataset}_std.csv')'''

    dataset_ = tf.data.Dataset.from_tensor_slices((dataset_input, dataset_output))
    dataset_NF = tf.data.Dataset.from_tensor_slices(ds_NF)
    n_batches = len(list(dataset_))
    print(f'number of batches are {n_batches}')


    shuffle_buffer = 1000

    if shauffled:
        dataset_ = dataset_.shuffle(shuffle_buffer)


    training_elements = int(n_batches * .7)
    print(f'training elements are {training_elements}')
    train_dataset = dataset_.take(training_elements)

    val_dataset = dataset_.skip(training_elements)
    val_dataset_length = len(list(val_dataset))
    val_elements = int(val_dataset_length * .6)
    print(f'val_elements are {val_elements}')
    validation_dataset = val_dataset.take(val_elements)
    test_dataset = val_dataset.skip(val_elements)
    print(f'for test we have {n_batches - training_elements - val_elements} elements')


    if not shauffled:
        train_dataset = train_dataset.shuffle(shuffle_buffer)
        validation_dataset = validation_dataset.shuffle(shuffle_buffer)
        test_dataset = test_dataset.shuffle(shuffle_buffer)

    train_dataset = train_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    dataset_NF = dataset_NF.batch(ds_NF.shape[0])
    '''for data in test_dataset:
        print(f' data.shape is {data[0].shape}')
        print(f'mean and std are {mean}, {std}')
        print(f'in test')
        for i in range(data[0].shape[0]):
            plt.figure()
            plt.scatter(data[0][i,:,1], data[0][i,:,2])
            plt.show(block=False)
            plt.pause(3)
            plt.close()
    for data in train_dataset:
        print(f' data.shape is {data[0].shape}')
        print(f'in train')
        for i in range(data[0].shape[0]):
            plt.figure()
            plt.scatter(data[0][i,:,1], data[0][i,:,2])
            plt.show(block=False)
            plt.pause(3)
            plt.close()'''


    return train_dataset, validation_dataset, test_dataset, dataset_NF, mean, std
