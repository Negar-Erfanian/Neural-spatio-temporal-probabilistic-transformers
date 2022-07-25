import numpy as np
import pandas as pd

import statistics


def process_data_earth(filename, event_num, seqs='fixed'):  # 'variable'

    calif = pd.read_csv(filename, delimiter='\s+', header=None)
    calif.drop(columns=[6, 11, 12, 13, 14, 15, 16], inplace=True)
    calif.rename(
        columns={0: 'Year', 1: 'Month', 2: 'Day', 3: 'Hour', 4: 'Minute', 5: 'Second', 7: 'Lat', 8: 'Long', 9: 'Depth', 10: 'Mag'},
        inplace=True)
    mag_threshold = 2.5
    calif = calif[calif.Mag >= mag_threshold]

    print('mean(calif.Mag) is', statistics.mean(calif.Mag))
    print('max(calif.Mag) is', max(calif.Mag))
    print('min(calif.Mag) is', min(calif.Mag))
    print('max(calif.Lat) is', max(calif.Lat))
    print('min(calif.Lat) is', min(calif.Lat))
    print('min(calif.Lat) is', statistics.mean(calif.Lat))
    print('max(calif.Long) is', max(calif.Long))
    print('min(calif.Long) is', min(calif.Long))
    print('min(calif.Long) is', statistics.mean(calif.Long))
    print('max(calif.Depth) is', max(calif.Depth))
    print('min(calif.Depth) is', min(calif.Depth))
    print('mean(calif.Depth) is', statistics.mean(calif.Depth))
    calif.reset_index(inplace=True)
    calif.drop(columns='index', inplace=True)
    seq = []
    time_diff = []

    for i in range(calif.shape[0]):
        seq.append(
            f'{calif.Year[i]}-{calif.Month[i]:02d}-{calif.Day[i]:02d}T{calif.Hour[i]:02d}:{calif.Minute[i]:02d}:{calif.Second[i]:02f}0Z')

    calif['Time'] = seq

    calif.drop(columns={'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'}, inplace=True)
    df = calif[['Time', 'Lat', 'Long', 'Depth', 'Mag']]
    # df = calif[['Time', 'Lat', 'Long', 'Depth']]
    basedate = pd.Timestamp('2008-01-01T00:04:12Z')
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Time'] = df['Time'].apply(lambda x: (x - basedate).total_seconds() / 60 / 60 / 24)
    time_diff.append(df['Time'][0])
    for i in range(len(seq) - 1):
        time_diff.append(df['Time'][i + 1] - df['Time'][i])

    df['Time_diff'] = time_diff
    sequences = {}
    length = []
    day_num = 40
    event_num = event_num  # 300
    mean = np.mean(df[['Lat', 'Long', 'Depth', 'Mag']], axis=0)
    std = np.std(df[['Lat', 'Long', 'Depth', 'Mag']], axis=0)

    if seqs == 'fixed':
        print('we are here')
        for range_ in range(5000):
            start = range_ * 2
            seq_name = f'{range_}'
            df_ = df.iloc[start:start + event_num]
            seq = df_.to_numpy().astype(np.float32)
            if seq.shape[0] < event_num:
                print('we are skipping becuz of length', seq_name)
                continue
            time, space, mag, time_diff = \
                df_.to_numpy().astype(np.float32)[:, 0:1], df_.to_numpy().astype(np.float32)[:, 1:4], df_.to_numpy().astype(np.float32)[:, 4:5], df_.to_numpy().astype(np.float32)[:, 5:6]
            print(f'time[-1]  is {time[-1]}')
            sequences[seq_name] = np.concatenate([time, space, mag, time_diff], axis=1)

            length.append(len(sequences[seq_name]))

        #print(f'min length is {min(length)}, max length is {max(length)}, average length is {statistics.mean(length)}.')

        #print(f'in the forward process we have {len(sequences)} files')
        #np.savez('earthquakes_calif.npz', **sequences)



    elif seqs == 'variable':
        for weeks in range(600):
            date = basedate + pd.Timedelta(weeks=weeks)
            start = (date - basedate).days
            end = (date + pd.Timedelta(days=day_num) - basedate).days
            # print('start is ', start)
            # print('end is', end)

            df_ = df[df['Time'] > start]
            df_ = df_[df_['Time'] < end]
            df_["Time"] = df_["Time"] - start
            seq_name = f'{date.year}{date.month:02d}{date.day:02d}'
            seq = df_.to_numpy().astype(np.float32)
            if seq.shape[0] < 40:
                print('we are skipping becuz of length', seq_name)
                continue

            elif np.max(df_["Time"]) <= 35:
                print('we are skipping becuz of time', seq_name)
                continue

            time, space, mag, time_diff = df_.to_numpy().astype(np.float32)[:, 0:1], df_.to_numpy().astype(np.float32)[
                                                                                     :, 1:4], df_.to_numpy().astype(
                np.float32)[:, 4:5], df_.to_numpy().astype(np.float32)[:, 5:6]

            sequences[seq_name] = np.concatenate([time, space, mag, time_diff], axis=1)

            length.append(len(sequences[seq_name]))

        print(f'min length is {min(length)}, max length is {max(length)}, average length is {statistics.mean(length)}.')

        print(f'in the forward process we have {len(sequences)} files')
        #np.savez('data/earthquakes_calif.npz', **sequences)

    return mean, std


if __name__ == '__main__':
    mean, std = process_data_earth('California.txt', event_num=300, seqs='fixed')
