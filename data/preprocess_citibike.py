# Copyright (c) Facebook, Inc. and its affiliates.

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import zipfile
from data.download_utils import download_url

import matplotlib.pyplot as plt

def download(root="", year=2019):
    for month in range(1, 13):
        url = f"https://s3.amazonaws.com/tripdata/{year}{month:02d}-citibike-tripdata.csv.zip"
        dirname = os.path.join(root, "citibike", "raw")
        if download_url(url, os.path.join(root, "citibike", "raw")):
            filename = os.path.join(dirname, os.path.basename(url))
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(dirname)


def process_data_bike(root="", year=2019, event_num=500, dl = True, data_path = 'data/citibike.npz'):
    np.random.seed(0)
    if dl:
        download()
        print(f'are we here?')

        ###########


        dfs = []
        for month in tqdm(range(1, 13)):
            filepath = os.path.join(root, "citibike", "raw", f"{year}{month:02d}-citibike-tripdata.csv")
            dfs.append(pd.read_csv(filepath))
        df = pd.concat(dfs)
        stds = df.std(0)

        df["starttime"] = pd.to_datetime(df["starttime"])

        sequences = {}
        fig = plt.figure(figsize=(20, 5))
        for i in range(5):
            ax = fig.add_subplot(1, 5, i+1)
            ax.hist(df.to_numpy()[:, i], bins=50)
            ax.tick_params(labelsize=18)
            ax.ticklabel_format(style='sci', scilimits=(0, 2), axis='y')
            ax.grid()
        fig.tight_layout()
        plt.savefig(f'citibikehist.png')
        for range_ in range(5000):
            start = range_ * 2
            seq_name = f'{range_}'
            df_ = df.iloc[start:start + event_num]
            df_ = df_.sort_values(by=["starttime"])
            starttime = df_["starttime"]
            if df_.shape[0] < event_num:
                continue

            t = (
                    pd.DatetimeIndex(starttime).hour * 60 * 60 +
                    pd.DatetimeIndex(starttime).minute * 60 +
                    pd.DatetimeIndex(starttime).second +
                    pd.DatetimeIndex(starttime).microsecond * 1e-6
            )
            t = np.array(t) / 60 / 60
            t = t.astype(np.float32)

            time_diff = [t[0]]

            for i in range(t.shape[0] - 1):
                time_diff.append(t[i + 1] - t[i])

            y = df_["start station latitude"]
            x = df_["start station longitude"]
            case = df_['birth year']

            x = np.array(x).astype(np.float32)
            y = np.array(y).astype(np.float32)
            case = np.array(case).astype(np.float32)

            seq = np.stack([t, x, y, case, time_diff], axis=1)
            sequences[seq_name] = seq

        np.savez(data_path, **sequences)
        data = np.load(data_path)
        files = data.files
        ds = [data[file][:, 1:4] for file in files]
        ds_array = np.concatenate(ds, axis=0).astype(np.float32)
        mean = np.mean(pd.DataFrame(ds_array).astype('float32'), axis=0)
        std = np.std(pd.DataFrame(ds_array).astype('float32'), axis=0)
        ###########


    else:
        data = np.load(data_path)
        files = data.files

        ds = [data[file][:, 1:4] for file in files]
        ds_array = np.concatenate(ds, axis = 0).astype(np.float32)
        mean = np.mean(pd.DataFrame(ds_array).astype('float32'), axis = 0)
        std = np.std(pd.DataFrame(ds_array).astype('float32'), axis = 0)

        ds_all = [data[file] for file in files]
        df = pd.DataFrame(np.concatenate(ds_all, axis=0).astype(np.float32)).astype('float32')
        fig = plt.figure(figsize=(20, 5))
        for i in range(5):
            ax = fig.add_subplot(1, 5, i+1)
            ax.hist(df.to_numpy()[:, i], bins=50)
            ax.tick_params(labelsize=18)
            ax.ticklabel_format(style='sci', scilimits=(0, 2), axis='y')
            ax.grid()
        fig.tight_layout()
        plt.savefig(f'citibikehist.png')


    return mean, std


def add_spatial_noise(coords, std=[0., 0., 0., 0., 0.]):
    return coords + np.random.randn(*coords.shape) * std


if __name__ == "__main__":
    mean, std = process_data_bike(event_num=50, dl = True, data_path = 'citibike.npz')
