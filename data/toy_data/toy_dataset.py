# Copyright (c) Facebook, Inc. and its affiliates.

from functools import partial
import contextlib
import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.size']=18

import statistics


from data.toy_data.MHP import MHP

END_TIME = 3075.0


def generate(mhp, data_fn, ndim, num_classes):
    mhp.generate_seq(END_TIME)
    event_times, classes = zip(*mhp.data)
    classes = np.concatenate(classes)
    n = len(event_times)
    data = data_fn(n)
    seq = np.zeros((n, ndim + 2))
    seq[:, 0] = event_times
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):
        seq[:, 1:ndim + 1] = seq[:, 1:ndim + 1] + data_i * (i == classes)[:, None]
        seq[(i == classes), ndim + 1:ndim + 2] = i+1
    print(f'seq[:, ndim + 1:ndim + 2] is {seq[:, -1]}')
    return seq


def pinwheel(num_samples, num_classes):
    radial_std = 0.3
    tangential_std = 0.1
    num_per_class = num_samples
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)
    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * np.einsum("ti,tij->tj", features, rotations)


def gmm(num_samples):
    m = np.linspace(-2, 2, 3).reshape(3, 1)
    std = 0.2
    return (np.random.randn(1, num_samples) * std + m).reshape(-1, 1)


@contextlib.contextmanager
def temporary_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def process_data_pinwheel(filename, event_num_per, event_num, num_classes, seqs = 'fixed'):
    with temporary_seed(1):
        data = pinwheel(event_num_per, num_classes)
    classes = np.ones((data.shape[0],1))
    time_diff = []
    sequences = {}
    length = []
    hawkes_time = pd.read_csv(filename, header=None).T.iloc[:data.shape[0]]#.to_numpy().astype(np.float32)[:, :data.shape[0]].reshape(-1,1)
    hawkes_time.rename(columns={0: 'Time'}, inplace=True)
    hawkes_time['Lat'], hawkes_time['Long'] = data[:,0], data[:,1]
    hawkes_time['Class'] = classes


    time_diff.append(hawkes_time['Time'][0])
    for i in range(data.shape[0] - 1):
        time_diff.append(hawkes_time['Time'][i + 1] - hawkes_time['Time'][i])
    hawkes_time['Time_diff'] = time_diff
    mean = np.mean(hawkes_time[['Lat', 'Long', 'Class']], axis=0)
    std = np.std(hawkes_time[['Lat', 'Long', 'Class']], axis=0)
    sss = []
    if seqs == 'fixed':
        for range_ in range(5000):
            start = range_ * 2 #2
            seq_name = f'{range_}'
            df_ = hawkes_time.iloc[start:start + event_num]
            '''plt.figure()
            plt.scatter(df_.to_numpy()[:,1], df_.to_numpy()[:,2])
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()'''
            seq = df_.to_numpy().astype(np.float32)
            sss.append(seq)
            if seq.shape[0] < event_num:
                continue
            time, space, c, time_diff = \
                df_.to_numpy().astype(np.float32)[:, 0:1], df_.to_numpy().astype(np.float32)[:, 1:3], df_.to_numpy().astype(
                    np.float32)[:, 3:4], df_.to_numpy().astype(np.float32)[:, 4:5]
            sequences[seq_name] = np.concatenate([time, space, c, time_diff], axis=1)

            length.append(len(sequences[seq_name]))
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 5, 1)
        ax.scatter(hawkes_time.to_numpy()[:, 1], hawkes_time.to_numpy()[:, 2], marker = '*', c = 'black')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_yticklabels([])
        ax.set_xticklabels([])


        for i in range(2, 6):
            ax = fig.add_subplot(1, 5, i)
            ax.scatter(sss[(i-1)*200][:, 1], sss[(i-1)*200][:, 2], marker = '*', c = 'black')
            ax.set_xlim(-4,4)
            ax.set_ylim(-4, 4)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            fig.tight_layout()
            plt.savefig(f'pinwheel.png')

        fig = plt.figure(figsize=(20, 5))
        for i in range(5):
            ax = fig.add_subplot(1, 5, i+1)
            ax.hist(hawkes_time.to_numpy()[:, i], bins = 50)
            ax.tick_params(labelsize=18)
            ax.ticklabel_format(style='sci', scilimits=(0, 2), axis='y')
            ax.grid()
        fig.tight_layout()
        plt.savefig(f'pinwheelhist.png')
    elif seqs == 'variable':
        basedate = pd.Timestamp('2008-01-01T00:04:12Z')
        day_num = 40
        for weeks in range(600):
            date = basedate + pd.Timedelta(weeks=weeks)
            start = (date - basedate).days
            end = (date + pd.Timedelta(days=day_num) - basedate).days
            df_ = hawkes_time[hawkes_time['Time'] > start]
            df_ = df_[df_['Time'] < end]
            df_["Time"] = df_["Time"] - start
            seq_name = f'{weeks}'
            seq = df_.to_numpy().astype(np.float32)
            if seq.shape[0] < 40:
                continue
            elif np.max(df_["Time"]) <= 35:
                print('we are skipping becuz of time', seq_name)
                continue
            time, space, c, time_diff = df_.to_numpy().astype(np.float32)[:, 0:1], df_.to_numpy().astype(np.float32)[
                                                                                     :, 1:3], df_.to_numpy().astype(
                np.float32)[:, 3:4], df_.to_numpy().astype(np.float32)[:, 4:5]
            sequences[seq_name] = np.concatenate([time, space, c, time_diff], axis=1)
            length.append(len(sequences[seq_name]))

    print(f'min length is {min(length)}, max length is {max(length)}, average length is {statistics.mean(length)}.')

    print(f'in the forward process we have {len(sequences)} files')
    np.savez('data/pinwheel.npz', **sequences)



    '''alpha = 0.6
    m = np.array([0.1] * num_classes)
    a = np.diag([alpha] * (num_classes - 1), k=-1) + np.diag([alpha], k=num_classes - 1) + np.diag([0.0] * num_classes,
                                                                                                   k=0)
    w = 3.0
    time_diff = []
    sequences = {}
    mhp = MHP(mu=m, alpha=a, omega=w)
    with temporary_seed(13579):
        data_fn = partial(pinwheel, num_classes=num_classes)
        data_set = generate(mhp, data_fn, ndim=2, num_classes=num_classes)
    print(f'data_set is {pd.DataFrame(data_set)}')
    mean = pd.DataFrame(np.mean(data_set[:, 1:4], axis=0).reshape(1,-1))
    std = pd.DataFrame(np.std(data_set[:, 1:4], axis=0).reshape(1,-1))
    time_diff.append(data_set[0,0])
    for i in range(data_set.shape[0] - 1):
        time_diff.append(data_set[i + 1,0] - data_set[i,0])
    time_diff = np.array(time_diff)[:, np.newaxis].astype(np.float32)
    plt.scatter(data_set[:, 1], data_set[:, 2])
    plt.show()
    for range_ in range(1000):
        start = range_ * 80  #2
        seq_name = f'{range_}'
        print(f'event_num is {event_num}')
        df_ = data_set[start:start + event_num,:]
        plt.scatter(df_[:, 1], df_[:, 2])
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        seq = df_.astype(np.float32)
        if seq.shape[0] < event_num:
            print('we are skipping becuz of length', seq_name)
            continue
        time, space, mag, time_different = \
            df_.astype(np.float32)[:, 0:1], df_.astype(np.float32)[:, 1:3], df_.astype(
                np.float32)[:, 3:4], time_diff[start:start + event_num,:]
        #print(f'time[-1]  is {time[-1]}')
        sequences[seq_name] = np.concatenate([time, space, mag, time_different], axis=1)'''

    #np.savez('data/pinwheel.npz', **sequences)
    return mean, std

if __name__ == "__main__":
    num_classes = 7
    event_num_per = 1000
    event_num = 500
    mean, std = process_data_pinwheel(event_num_per = event_num_per, event_num = event_num, num_classes = num_classes)
    data = pinwheel(event_num_per, num_classes)
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):
        print(i)
        plt.scatter(data_i[:, 0], data_i[:, 1], c=f"C{i}", s=2)


    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig(f"pinwheel{num_classes}.png")
