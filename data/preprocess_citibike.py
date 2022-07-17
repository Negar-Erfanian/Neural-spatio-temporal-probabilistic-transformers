# Copyright (c) Facebook, Inc. and its affiliates.

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import zipfile
from data.download_utils import download_url


def download(root="", year=2019):
    for month in range(1, 13):
        url = f"https://s3.amazonaws.com/tripdata/{year}{month:02d}-citibike-tripdata.csv.zip"
        dirname = os.path.join(root, "citibike", "raw")
        if download_url(url, os.path.join(root, "citibike", "raw")):
            filename = os.path.join(dirname, os.path.basename(url))
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(dirname)


def process_data_bike(root="", year=2019, event_num=500):
    np.random.seed(0)

    dfs = []
    for month in tqdm(range(1, 13)):
        filepath = os.path.join(root, "citibike", "raw", f"{year}{month:02d}-citibike-tripdata.csv")
        dfs.append(pd.read_csv(filepath))
    df = pd.concat(dfs)
    # print(f'df is {df.columns}')
    stds = df.std(0)
    std_x = stds["start station longitude"]
    std_y = stds["start station latitude"]

    df["starttime"] = pd.to_datetime(df["starttime"])
    mean = np.mean(df[["start station latitude", "start station longitude", "birth year"]], axis=0)
    std = np.std(df[["start station latitude", "start station longitude", "birth year"]], axis=0)
    sequences = {}
    for range_ in range(5000):
        start = range_ * 2
        seq_name = f'{range_}'
        df_ = df.iloc[start:start + event_num]
        # print(f'df_ is {df_}')
        starttime = df_["starttime"]
        if df_.shape[0] < event_num:
            print('we are skipping becuz of length', seq_name)
            continue

        year = pd.DatetimeIndex(starttime).year[0]
        month = pd.DatetimeIndex(starttime).month[0]
        day = pd.DatetimeIndex(starttime).day[0]

        t = (
                pd.DatetimeIndex(starttime).hour * 60 * 60 +
                pd.DatetimeIndex(starttime).minute * 60 +
                pd.DatetimeIndex(starttime).second +
                pd.DatetimeIndex(starttime).microsecond * 1e-6
        )
        t = np.array(t) / 60 / 60

        time_diff = [t[0]]
        # Create numeric time column.

        for i in range(t.shape[0] - 1):
            time_diff.append(t[i + 1] - t[i])


        y = df_["start station latitude"]
        x = df_["start station longitude"]
        case = df_["birth year"]

        x = np.array(x)
        y = np.array(y)

        seq = np.stack([t, x, y, case, time_diff], axis=1)


        print(f'seq is {seq}')

        for i in range(20):
            # subsample_idx = np.sort(np.random.choice(seq.shape[0], seq.shape[0] // 500, replace=False))
            subsample_idx = np.random.rand(seq.shape[0]) < (1 / 500)
            while np.sum(subsample_idx) == 0:
                subsample_idx = np.random.rand(seq.shape[0]) < (1 / 500)

            sequences[seq_name + f"_{i:03d}"] = add_spatial_noise(seq[subsample_idx],
                                                                  std=[0., std_x * 0.02, std_y * 0.02])

            print(np.sum(subsample_idx))

    np.savez("data/citibike.npz", **sequences)

    return mean, std


def add_spatial_noise(coords, std=[0., 0., 0.]):
    return coords + np.random.randn(*coords.shape) * std


if __name__ == "__main__":
    download()
    mean, std = process_data_bike(event_num=500)
