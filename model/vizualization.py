import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow_probability as tfp

tfd = tfp.distributions

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
# %matplotlib notebook
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

BBOXES = {
    "Earthquake": (37.22333, 31.86317, -114.026, -121.953, 49.09, -1.6, 7.2, 0.5),
}

MAPS = {
    "Earthquake": "data/map.png"

}

FIGSIZE = 10
DPI = 300


def timing(arr1, arr2, append=True):
    arr1 = arr1.numpy()
    if append:
        arr1[0] += arr2[-1]

    for i in range(1, len(arr1)):
        arr1[i] += arr1[i - 1]

    return arr1


def plot_expectedtime(timediff_in, timediff_out, out_pred_time, bij_time, event_num, savepath, count):
    b_n = 3
    fig, ax = plt.subplots(2, b_n,
                           figsize=(FIGSIZE, int(FIGSIZE / 3)),
                           tight_layout=True)

    for i in range(b_n):
        time_true_in = timing(timediff_in[-(i + 1)], timediff_in[-(i + 1)], False)
        time_pred = timing(out_pred_time[-(i + 1)], time_true_in)
        time_true = timing(timediff_out[-(i + 1)], time_true_in)
        xlim = np.linspace(0, event_num - 1, event_num)
        ylim = np.linspace(np.min(np.append(time_true_in, time_true)), np.max(np.append(time_true_in, time_true)),
                           event_num)

        ax[0, i].plot(np.append(time_true_in, time_pred), 'o', label='Predicted time')
        ax[0, i].plot(np.append(time_true_in, time_true), 'r*', label='True time', markersize=4)
        ax[0, i].plot(xlim, ylim, '-', label='linear')
        ax[0, i].set_title(f'{i + 1} in batch of {b_n}')

        ax[0, i].legend(loc='upper left')

        # yerror = tf.reshape(test_ds_out_stddev[idx][i][:nonzeros[idx][i][0]],-1)
        # ax[1,i].errorbar(np.linspace(0,len(time_pred[:nonzeros[idx][i][0]])-1, len(time_pred[:nonzeros[idx][i][0]])),time_pred[:nonzeros[idx][i][0]],yerror, linestyle='None', marker='o', label = 'Predicted time')
        xlim = np.linspace(0, len(time_pred) - 1, len(time_pred))
        ylim = np.linspace(np.min(time_pred), np.max(time_pred), len(time_pred))

        ax[1, i].plot(time_pred, 'o', label='Predicted time', markersize=12)
        ax[1, i].plot(time_true, 'r*', label='True time', markersize=8)
        ax[1, i].plot(xlim, ylim, '-', label='linear')
        ax[1, i].set_xlabel('Events')
        if i == 0:
            ax[1, i].set_ylabel('Time (Days)')

    plt.savefig(f'{savepath}/test_result{count}.png')
    plt.close()


def plot_expected_intensity(bij_time, timediff_out, savepath, count):
    N = 1000
    minx, maxx = -5, 5
    xlim = np.linspace(minx, maxx, N)
    xlim = xlim.reshape(-1, 1)

    arr = np.repeat(xlim[:, np.newaxis, :], 3, axis=1)
    arr = arr[:, np.newaxis, :, :]
    print(f'timediff_out shape is {timediff_out.shape}')
    # print(f'bij_time is {bij_time}')

    log_likelihood = bij_time.log_prob(arr)[:, -1, :]
    # print(f'log_lok shape is {log_lok.shape}')

    fig = plt.figure(figsize=(5, 5))
    fig, viewer = plt.subplots(1, log_likelihood.shape[-1])
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    # viewer[0].plot(log_lok)

    for i in range(log_likelihood.shape[-1]):
        log_likelihood1 = log_likelihood[:, i]

        viewer[i].plot(log_likelihood1, xlim)
        viewer[i].plot(timediff_out[0, i], 'r*')

    plt.savefig(f'{savepath}/test_batch{count}.png')

    plt.close()

def plot_expected_3d_density_gmm(curr_time, input_time, history_data, expected_data, model, curr_path, savepath, count, idx):
    N = 85
    stacks = []
    plts = 3
    ll = []
    fig = plt.figure(figsize=(18, 8))
    im = plt.imread(f'{curr_path}/data/map.png').transpose((1, 0, 2))
    for i in range(plts):
        current_time = curr_time[i]
        expected_loc = expected_data[:,i,:]
        #print(f'expected_loc shape is {int(expected_loc[0, 2])}')
        normalized_data = np.concatenate([history_data, expected_loc], axis=0)
        minx, maxx = tf.math.reduce_min(normalized_data[:, 0]), tf.math.reduce_max(normalized_data[:, 0])
        miny, maxy = tf.math.reduce_min(normalized_data[:, 1]), tf.math.reduce_max(normalized_data[:, 1])
        minz, maxz = tf.math.reduce_min(normalized_data[:, 2]), tf.math.reduce_max(normalized_data[:, 2])

        xlim = np.linspace(minx, maxx, N)
        ylim = np.linspace(miny, maxy, N)
        zlim = np.linspace(minz, maxz, N)

        X1, X2, X3 = np.meshgrid(xlim, ylim, zlim)
        arr = np.stack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)], axis=1)
        loglikelihood_fn = model.spatial_model.spatial_conditional_logprob_fn(current_time, input_time, history_data)
        loglikelihood = loglikelihood_fn(arr)
        predicted = tf.reduce_mean(arr*tf.math.exp(loglikelihood)[...,tf.newaxis],0)
        print(f'predicted is {predicted}')
        loglikelihood = loglikelihood.reshape(N, N, N)


        ax = fig.add_subplot(2, plts, i+1)
        # plot background
        ax.imshow(im, extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
                              np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])
        ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:,:,int(expected_data[i,2])] * 100, levels=800, alpha=0.7, cmap='RdGy')
        ax.scatter(predicted[0], predicted[1], marker='*', color='blue', s=150)
        ax.scatter(history_data[-10:, 0], history_data[-10:, 1], marker='*', color='black', s=70)
        ax.scatter(expected_loc[0,0], expected_loc[0,1], marker='o', color='green', s=100)


        ax = fig.add_subplot(2, plts, i+4, projection='3d')
        ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:, :, -1], levels=800, alpha=0.7, cmap='RdGy',
                    zdir='z', offset=zlim[-1])
        ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:, :, int(expected_loc[0, 2])] * 1000,
                    levels=800, alpha=0.7, cmap='RdGy', zdir='z', offset=expected_loc[0, 2])
        ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:, :, 0], levels=800, alpha=0.7, cmap='RdGy',
                    zdir='z', offset=zlim[0])
        ax.scatter3D(predicted[0], predicted[1], predicted[2], marker='*', color='blue', s=280)
        ax.scatter3D(history_data[-10:, 0], history_data[-10:, 1], history_data[-10:, 2], marker='*', color='black',
                     s=150)
        ax.scatter3D(expected_loc[0, 0], expected_loc[0, 1], expected_loc[0, 2], marker='o',
                     color='green', s=400)
        ax.set_zlim(minz, maxz)
        ax.view_init(+25, -140)

        fig.tight_layout()
        history_data = np.concatenate([history_data, predicted[tf.newaxis, ...]], axis=0)
        input_time = np.concatenate([input_time, current_time[None]], axis=0)
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig(f'{savepath}/test_batch{count}_{idx}.png')

    plt.close()



def plot_expected_3d_density(history_data, expected_data, model, curr_path, dec_dist_loc, savepath, count, idx):
    N = 100
    stacks = []
    normalized_data = np.concatenate([history_data, expected_data], axis=0)
    minx, maxx = tf.math.reduce_min(normalized_data[:, 0]), tf.math.reduce_max(normalized_data[:, 0])
    miny, maxy = tf.math.reduce_min(normalized_data[:, 1]), tf.math.reduce_max(normalized_data[:, 1])
    minz, maxz = tf.math.reduce_min(normalized_data[:, 2]), tf.math.reduce_max(normalized_data[:, 2])

    xlim = np.linspace(minx, maxx, N)
    ylim = np.linspace(miny, maxy, N)
    zlim = np.linspace(minz, maxz, N)

    X1, X2, X3 = np.meshgrid(xlim, ylim, zlim)
    arr = np.stack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)], axis=1)
    arr2 = np.repeat(arr[:, np.newaxis, :], 3, axis=1)
    print(f'arr2 shape is {arr2.shape}')

    y, log_likelihood = model.bij_loc(arr2, dec_dist_loc, test=True)
    print(f'log_likelihood shape is {log_likelihood.shape}')
    log_likelihood = log_likelihood[:, 0, :]

    fig = plt.figure(figsize=(18, 8))
    im = plt.imread(f'{curr_path}/data/map.png').transpose((1, 0, 2))  # background faults image
    # fig, viewer = plt.subplots(1, log_likelihood.shape[-1])
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)

    for i in range(log_likelihood.shape[-1] * 2):  # log_likelihood.shape[-1]

        if i <= 2:
            log_likelihood1 = log_likelihood[:, i]
            log_likelihood1 = log_likelihood1.reshape(N, N, N)
            ax = fig.add_subplot(2, log_likelihood.shape[-1], i + 1)
            # plot background
            ax.imshow(im, extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
                                  np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])
            ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood1)[:, :, 0] * 100, levels=800, alpha=0.7,
                        cmap='RdGy')

            ax.scatter(history_data[-10:, 0], history_data[-10:, 1], marker='*', color='black', s=70)
            ax.scatter(expected_data[i, 0], expected_data[i, 1], marker='o', color='green', s=100)


        else:
            log_likelihood1 = log_likelihood[:, i - 3]
            log_likelihood1 = log_likelihood1.reshape(N, N, N)
            ax = fig.add_subplot(2, log_likelihood.shape[-1], i + 1, projection='3d')
            ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood1)[:, :, -1], levels=800, alpha=0.7, cmap='RdGy',
                        zdir='z', offset=zlim[-1])
            # ax.contourf(X1[:,:,0], X2[:,:,0], tf.exp(log_likelihood1)[:,:,70], levels=800, alpha=0.7, cmap='RdGy', zdir='z', offset=zlim[70])
            ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood1)[:, :, int(expected_data[i - 3, 2])] * 1000,
                        levels=800, alpha=0.7, cmap='RdGy', zdir='z', offset=expected_data[i - 3, 2])
            # ax.contourf(X1[:,:,0], X2[:,:,0], tf.exp(log_likelihood1)[:,:,20]*100, levels=800, alpha=0.7, cmap='RdGy', zdir='z', offset=zlim[20])
            ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood1)[:, :, 0], levels=800, alpha=0.7, cmap='RdGy',
                        zdir='z', offset=zlim[0])

            ax.scatter3D(history_data[-10:, 0], history_data[-10:, 1], history_data[-10:, 2], marker='*', color='black',
                         s=150)
            ax.scatter3D(expected_data[i - 3, 0], expected_data[i - 3, 1], expected_data[i - 3, 2], marker='o',
                         color='green', s=400)
            ax.set_zlim(minz, maxz)
            ax.view_init(+25, -140)

    fig.tight_layout()
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig(f'{savepath}/test_batch{count}_{idx}.png')

    plt.close()


def plot_att_scores(test_att_weights_dec_time, savepath, count, mag):
    print(f' mag is {mag}')
    fig, ax = plt.subplots(3, 1,
                           figsize=(FIGSIZE, FIGSIZE),
                           tight_layout=True)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.8, wspace=0.05, hspace=0.005)
    layer1 = list(test_att_weights_dec_time.keys())[1]
    layer2 = list(test_att_weights_dec_time.keys())[3]
    img = ax[0].imshow(test_att_weights_dec_time[layer1][0, 0][:10, :40])
    ax[0].set_title(f'{layer1} head 1, first in batch with max mag {mag[0]}')
    ax[0].set_xlabel('Past events')
    ax[0].set_ylabel('Future events')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    # img = ax[1].imshow(test_att_weights_dec_time['decoder_layer2_block2_decenc_att'][0,2][:4, :40])
    # ax[1].set_title('encoder-decoder attention scores layer 1 head 3')
    # ax[1].set_xlabel('Past events')
    # ax[1].set_xlabel('Future events')
    # plt.colorbar(img, ax = ax[1])
    img = ax[1].imshow(test_att_weights_dec_time[layer2][1, 0][:10, :40])
    ax[1].set_title(f'{layer2} head 3, first in batch with max mag {mag[1]}')
    ax[1].set_xlabel('Past events')
    ax[1].set_ylabel('Future events')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

    img = ax[2].imshow(test_att_weights_dec_time[layer2][2, 0][:10, :40])
    ax[2].set_title(f'{layer2} head 5, first in batch with max mag {mag[2]}')
    ax[2].set_xlabel('Past events')
    ax[2].set_ylabel('Future events')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    plt.savefig(f'{savepath}/train_scoreresult{count}.png')
    # plt.show(block=False)
    # plt.pause(5)
    plt.close()


