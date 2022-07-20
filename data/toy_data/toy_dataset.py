# Copyright (c) Facebook, Inc. and its affiliates.

from functools import partial
import contextlib
import numpy as np


from MHP import MHP

END_TIME = 3075.0


def generate(mhp, data_fn, ndim, num_classes):
    mhp.generate_seq(END_TIME)
    event_times, classes = zip(*mhp.data)
    classes = np.concatenate(classes)
    n = len(event_times)
    #print(f'n is {n}')

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
    #print(f'labels are {sum(labels==0)}')

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

def process_data_pinwheel(event_num, num_classes):

    alpha = 0.6
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
    print(data_set[:,-1])
    mean = np.mean(data_set[:, 1:4], axis=0)
    std = np.std(data_set[:, 1:4], axis=0)
    time_diff.append(data_set[0,0])
    for i in range(data_set.shape[0] - 1):
        time_diff.append(data_set[i + 1,0] - data_set[i,0])
    time_diff = np.array(time_diff)[:, np.newaxis].astype(np.float32)
    for range_ in range(10000):
        start = range_ * 2
        seq_name = f'{range_}'
        df_ = data_set[start:start + event_num,:]
        seq = df_.astype(np.float32)
        if seq.shape[0] < event_num:
            print('we are skipping becuz of length', seq_name)
            continue
        time, space, mag, time_different = \
            df_.astype(np.float32)[:, 0:1], df_.astype(np.float32)[:, 1:3], df_.astype(
                np.float32)[:, 3:4], time_diff[start:start + event_num,:]
        #print(f'time[-1]  is {time[-1]}')
        sequences[seq_name] = np.concatenate([time, space, mag, time_different], axis=1)


    # print(f'min length is {min(length)}, max length is {max(length)}, average length is {statistics.mean(length)}.')

    # print(f'in the forward process we have {len(sequences)} files')
    np.savez('data/pinwheel.npz', **sequences)
    return mean, std

if __name__ == "__main__":

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean, std = process_data_pinwheel(event_num = 500, num_classes = 50)
    print(mean, std)
    num_classes = 10
    # rng = np.random.RandomState(13579)
    data = pinwheel(1000, num_classes)
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):
        plt.scatter(data_i[:, 0], data_i[:, 1], c=f"C{i}", s=2)

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig(f"pinwheel{num_classes}.png")
