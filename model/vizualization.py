import numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow_probability as tfp
from scipy import ndimage


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
    if not isinstance(arr1, np.ndarray):
        arr1 = arr1.numpy()

    if append:
        arr1[0] += arr2[-1]

    for i in range(1, len(arr1)):

        arr1[i] += arr1[i - 1]

    return arr1



def plot_expectedtime(timediff_in, timediff_out, out_pred_time, event_num, savepath, count, idx):
    b_n = 2
    fig, ax = plt.subplots(1, b_n, gridspec_kw={'width_ratios': [2,1]},
                           figsize=(10  , 2),
                           tight_layout=True)

    '''for i in range(b_n):
        time_true_in = timing(timediff_in[-(i + 1)], timediff_in[-(i + 1)], False)
        time_pred = timing(out_pred_time[-(i + 1)], time_true_in)
        time_true = timing(timediff_out[-(i + 1)], time_true_in)
        xlim = np.linspace(0, event_num - 1, event_num)
        ylim = np.linspace(np.min(np.append(time_true_in, time_true)), np.max(np.append(time_true_in, time_true)),
                           event_num)

        ax[0, i].plot(np.append(time_true_in, time_pred), 'o', label='Predicted time')
        ax[0, i].plot(np.append(time_true_in, time_true), 'r*', label='True time', markersize=4)
        ax[0, i].plot(xlim, ylim, '-', label='linear')
        #ax[0, i].set_title(f'{i + 1} in batch of {b_n}')

        ax[0, i].legend(loc='upper left')

        # yerror = tf.reshape(test_ds_out_stddev[idx][i][:nonzeros[idx][i][0]],-1)
        # ax[1,i].errorbar(np.linspace(0,len(time_pred[:nonzeros[idx][i][0]])-1, len(time_pred[:nonzeros[idx][i][0]])),time_pred[:nonzeros[idx][i][0]],yerror, linestyle='None', marker='o', label = 'Predicted time')
        xlim = np.linspace(0, len(time_pred) - 1, len(time_pred))
        ylim = np.linspace(np.min(time_pred), np.max(time_pred), len(time_pred))

        ax[1, i].plot(time_pred, 'o', label='Predicted time', markersize=12)
        ax[1, i].plot(time_true, 'r*', label='True time', markersize=8)
        ax[1, i].plot(xlim, ylim, '-', label='linear')'''
    time_true_in = timing(timediff_in[idx], timediff_in[idx], False)
    time_pred = timing(out_pred_time[idx], time_true_in)
    time_true = timing(timediff_out[idx], time_true_in)
    xlim = np.linspace(0, event_num - 1, event_num)
    ylim = np.linspace(np.min(np.append(time_true_in, time_true)), np.max(np.append(time_true_in, time_true)),
                       event_num)
    ax[0].plot(np.append(time_true_in, time_true), 'og', label='True time', markersize=8)
    ax[0].plot(np.append(time_true_in, time_pred), 'r*', label='Predicted time', markersize=4)
    ax[0].plot(xlim, ylim, '-', label='linear')
    # ax[0, i].set_title(f'{i + 1} in batch of {b_n}')

    ax[0].legend(loc='upper left')

    # yerror = tf.reshape(test_ds_out_stddev[idx][i][:nonzeros[idx][i][0]],-1)
    # ax[1,i].errorbar(np.linspace(0,len(time_pred[:nonzeros[idx][i][0]])-1, len(time_pred[:nonzeros[idx][i][0]])),time_pred[:nonzeros[idx][i][0]],yerror, linestyle='None', marker='o', label = 'Predicted time')
    xlim = np.linspace(0, len(time_pred) - 1, len(time_pred))
    ylim = np.linspace(np.min(time_pred), np.max(time_pred), len(time_pred))


    ax[1].plot(time_true, 'og', label='True time', markersize=12)
    ax[1].plot(time_pred, 'r*', label='Predicted time', markersize=15)
    ax[1].plot(xlim, ylim, '-', label='linear')


    plt.savefig(f'{savepath}/test_result{count}_{idx}.png')
    plt.close()


def plot_expected_intensity(bij_time, timediff_out, savepath, count):
    N = 1000
    minx, maxx = -5, 5
    xlim = np.linspace(minx, maxx, N)
    xlim = xlim.reshape(-1, 1)

    arr = np.repeat(xlim[:, np.newaxis, :], 3, axis=1)
    arr = arr[:, np.newaxis, :, :]

    for j in range(10):
        log_likelihood = bij_time.log_prob(arr)[:, j, :]

        fig = plt.figure(figsize=(5, 5))
        fig, viewer = plt.subplots(1, log_likelihood.shape[-1])
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.1)
        # viewer[0].plot(log_lok)

        for i in range(log_likelihood.shape[-1]):
            log_likelihood1 = log_likelihood[:, i]

            viewer[i].plot(log_likelihood1, xlim)
            viewer[i].plot(timediff_out[j, i], 'r*')
        fig.tight_layout()
        plt.savefig(f'{savepath}/test_batch{count}_sample{j}.png')

    plt.close()

def plot_expected_3d_density_gmm(curr_time, input_time, history_data, expected_data, aux_state_in, aux_state_out, model, curr_path, savepath, count, idx):
    N = 80
    plts =  expected_data.shape[1]
    fig = plt.figure(figsize=(18, 8))
    im = plt.imread(f'{curr_path}/data/map.png').transpose((1, 0, 2))

    for i in range(plts):
        current_time = curr_time[:,i]
        expected_loc = expected_data[:,i,:][:, tf.newaxis,:]
        aux_state = tf.concat([aux_state_in, aux_state_out[:, i, :][..., tf.newaxis]], axis=1)

        #print(f'expected_loc shape is {int(expected_loc[0, 2])}')
        normalized_data = np.concatenate([history_data, expected_loc], axis=1)
        minx, maxx = tf.math.reduce_min(normalized_data[:,:, 0]), tf.math.reduce_max(normalized_data[:,:, 0])
        miny, maxy = tf.math.reduce_min(normalized_data[:,:, 1]), tf.math.reduce_max(normalized_data[:,:, 1])
        minz, maxz = tf.math.reduce_min(normalized_data[:,:, 2]), tf.math.reduce_max(normalized_data[:,:, 2])

        xlim = np.linspace(minx, maxx, N)
        ylim = np.linspace(miny, maxy, N)
        zlim = np.linspace(minz, maxz, N)

        X1, X2, X3 = np.meshgrid(xlim, ylim, zlim)
        arr = np.stack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)], axis=1)
        arr = tf.broadcast_to(arr[None], (expected_data.shape[0], *arr.shape))
        arr = arr.reshape(-1, arr.shape[-1])
        loglikelihood_fn = model.spatial_model.spatial_conditional_logprob_fn(current_time[...,None], input_time[...,None], history_data, aux_state=aux_state)
        loglikelihood = loglikelihood_fn(arr)
        norm_loglik = (loglikelihood - np.min(loglikelihood)) / (np.max(loglikelihood) - np.min(loglikelihood))
        #predicted = tf.reduce_max((arr*norm_loglik[...,tf.newaxis]).reshape(expected_data.shape[0], N*N*N, -1),1)  #(batch,d)
        predicted_arg = tf.math.argmax(norm_loglik.reshape(expected_data.shape[0], N*N*N),-1)
        arr_reshaped = arr.reshape(expected_data.shape[0], N * N * N, -1)
        predicted = []
        for j in range(expected_data.shape[0]):
            predicted.append(arr_reshaped[j, predicted_arg[j], :])

        predicted = tf.concat([predicted], axis=0)
        loglikelihood = loglikelihood.reshape(N, N, N)


        ax = fig.add_subplot(2, plts, i+1)

        if i == 0:
            first_result = tf.exp(loglikelihood)[:,:,int(expected_data[0][i,2])]*0.05

            imag = ax.imshow(ndimage.rotate(first_result, -90).T, cmap='RdGy', vmin=0.0, vmax=1.0,
                             extent=[np.min(normalized_data[:,:, 0]), np.max(normalized_data[:,:, 0]),
                                     np.min(normalized_data[:,:, 1]), np.max(normalized_data[:,:, 1])])
        else:
            result = tf.exp(loglikelihood)[:,:,int(expected_data[0][i,2])]*0.05 - first_result
            first_result = tf.exp(loglikelihood)[:,:,int(expected_data[0][i,2])]*0.05
            imag = ax.imshow(ndimage.rotate(np.abs(result), -90).T*4 , cmap='RdGy', vmin=0.4, vmax=0.5,
                             extent=[np.min(normalized_data[:,:, 0]), np.max(normalized_data[:,:, 0]),
                                     np.min(normalized_data[:,:, 1]), np.max(normalized_data[:,:, 1])])

        # plot background
        #ax.imshow(im, extent=[np.min(normalized_data[:,:, 0]), np.max(normalized_data[:,:, 0]),
        #                      np.min(normalized_data[:,:, 1]), np.max(normalized_data[:,:, 1])])
        #imag = ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:,:,int(expected_data[0][i,2])]*0.005, levels=800, alpha=0.7, cmap='RdGy', vmin = 0, vmax = 0.1)
        ax.scatter(predicted[0, 0], predicted[0, 1], marker='*', color='red', s=250)
        ax.scatter(history_data[0,-50:, 0], history_data[0,-50:, 1], marker='*', color='black', s=70)
        ax.scatter(expected_loc[0,0, 0], expected_loc[0,0, 1], marker='o', color='green', s=200)
        plt.colorbar(imag)

        ax = fig.add_subplot(2, plts, i+4, projection='3d')
        ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:, :, -1], levels=800, alpha=0.7, cmap='RdGy',
                    zdir='z', offset=zlim[-1])
        ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:, :, int(expected_loc[0,0, 2])] ,
                    levels=800, alpha=0.7, cmap='RdGy', zdir='z', offset=expected_loc[0,0, 2])
        ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(loglikelihood)[:, :, 0], levels=800, alpha=0.7, cmap='RdGy',
                    zdir='z', offset=zlim[0])
        ax.scatter3D(predicted[0, 0], predicted[0, 1], predicted[0, 2], marker='*', color='red', s=380)
        ax.scatter3D(history_data[0,-50:, 0], history_data[0,-50:, 1], history_data[0, -50:, 2], marker='*', color='black',
                     s=150)
        ax.scatter3D(expected_loc[0,0, 0], expected_loc[0,0, 1], expected_loc[0,0, 2], marker='o',
                     color='green', s=400)
        ax.set_zlim(minz, maxz)
        ax.view_init(+25, -140)




        history_data = np.concatenate([history_data, predicted[tf.newaxis, ...]], axis=1)
        input_time = np.concatenate([input_time, current_time[None]], axis=1)
        aux_state_in = aux_state

    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    fig.tight_layout()
    plt.savefig(f'{savepath}/test_batch{count}_{idx}.png')

    plt.close()

def plot_expected_2d_density_gmm(curr_time, input_time, history_data, expected_data, aux_state_in, aux_state_out, model, curr_path, savepath, count, idx):
    N = 100
    plts =  expected_data.shape[1]
    ll = []
    fig = plt.figure(figsize=(12, 4))
    #im = plt.imread(f'{curr_path}/data/map.png').transpose((1, 0, 2))
    for i in range(plts):
        current_time = curr_time[:,i]
        expected_loc = expected_data[:,i,:][:, tf.newaxis,:]
        aux_state = tf.concat([aux_state_in, aux_state_out[:, i, :][..., tf.newaxis]], axis=1)
        normalized_data = np.concatenate([history_data, expected_loc], axis=1)
        minx, maxx = tf.math.reduce_min(normalized_data[:,:, 0]), tf.math.reduce_max(normalized_data[:,:, 0])
        miny, maxy = tf.math.reduce_min(normalized_data[:,:, 1]), tf.math.reduce_max(normalized_data[:,:, 1])

        xlim = np.linspace(minx, maxx, N)
        ylim = np.linspace(miny, maxy, N)

        X1, X2 = np.meshgrid(xlim, ylim)
        arr = np.stack([X1.reshape(-1), X2.reshape(-1) ], axis=1)
        arr = tf.broadcast_to(arr[None], (expected_data.shape[0], *arr.shape))
        arr = arr.reshape(-1, arr.shape[-1])
        loglikelihood_fn = model.spatial_model.spatial_conditional_logprob_fn(current_time[..., None],
                                                                              input_time[..., None], history_data,
                                                                              aux_state=aux_state)
        loglikelihood = loglikelihood_fn(arr)
        norm_loglik = (loglikelihood - np.min(loglikelihood)) / (np.max(loglikelihood) - np.min(loglikelihood))

        #predicted = tf.reduce_max((arr * norm_loglik[..., tf.newaxis]).reshape(expected_data.shape[0], N * N , -1),1)  # (batch,d)
        predicted_arg = tf.math.argmax(norm_loglik.reshape(expected_data.shape[0], N * N ), -1)
        arr_reshaped = arr.reshape(expected_data.shape[0], N * N, -1)
        predicted = []
        for j in range(expected_data.shape[0]):
            predicted.append(arr_reshaped[j, predicted_arg[j], :])

        predicted = tf.concat([predicted], axis=0)
        loglikelihood = norm_loglik.reshape(N, N)
        #print(tf.exp(loglikelihood))


        ax = fig.add_subplot(1, plts, i+1)

        if i == 0:
            first_result = tf.exp(loglikelihood)

            imag = ax.imshow(ndimage.rotate(first_result, -90).T*0.38, cmap='RdGy', vmin=0.8, vmax=1.0,
                             extent=[np.min(normalized_data[:,:, 0]), np.max(normalized_data[:,:, 0]),
                                     np.min(normalized_data[:,:, 1]), np.max(normalized_data[:,:, 1])])
        else:
            result = tf.exp(loglikelihood) - first_result
            first_result = tf.exp(loglikelihood)
            imag = ax.imshow(ndimage.rotate(np.abs(result), -90).T*40 , cmap='RdGy', vmin=0.02, vmax=0.1,
                             extent=[np.min(normalized_data[:,:, 0]), np.max(normalized_data[:,:, 0]),
                                     np.min(normalized_data[:,:, 1]), np.max(normalized_data[:,:, 1])])

        # plot background
        #ax.imshow(im, extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
        #                      np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])
        #imag = ax.contourf(X1, X2, tf.exp(loglikelihood)*0.1, levels=800, alpha=0.7, cmap='RdGy', vmin = 0.2, vmax = 0.3)
        ax.scatter(predicted[0,0], predicted[0,1], marker='*', color='red', s=200)
        ax.scatter(history_data[0,-50:, 0], history_data[0,-50:, 1], marker='*', color='black', s=70)
        ax.scatter(expected_loc[0,0,0], expected_loc[0,0,1], marker='o', color='green', s=200)



        fig.tight_layout()
        history_data = np.concatenate([history_data, predicted[tf.newaxis, ...]], axis=1)
        input_time = np.concatenate([input_time, current_time[None]], axis=1)
        aux_state_in = aux_state
        plt.colorbar(imag)
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig(f'{savepath}/test_batch{count}_{idx}.png')

    plt.close()



def plot_expected_3d_density(history_data, expected_data, model, curr_path, dec_dist_loc, savepath, count, idx):
    N = 100

    normalized_data = np.concatenate([history_data, expected_data], axis=0)
    minx, maxx = tf.math.reduce_min(normalized_data[:, 0]), tf.math.reduce_max(normalized_data[:, 0])
    miny, maxy = tf.math.reduce_min(normalized_data[:, 1]), tf.math.reduce_max(normalized_data[:, 1])
    minz, maxz = tf.math.reduce_min(normalized_data[:, 2]), tf.math.reduce_max(normalized_data[:, 2])

    xlim = np.linspace(minx, maxx, N)
    ylim = np.linspace(miny, maxy, N)
    zlim = np.linspace(minz, maxz, N)

    X1, X2, X3 = np.meshgrid(xlim, ylim, zlim)
    arr = np.stack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)], axis=1)
    arr2 = np.repeat(arr[:, np.newaxis, :], expected_data.shape[0], axis=1)
    #print(f'arr2 shape is {arr2.shape}')
    #print(f'expected_data shape is {expected_data.shape}')

    y, log_likelihood = model.bij_loc(arr2, dec_dist_loc, test=True)
    log_likelihood = log_likelihood[:, 0, :]

    fig = plt.figure(figsize=(18, 8))
    im = plt.imread(f'{curr_path}/data/map.png').transpose((1, 0, 2))  # background faults image
    # fig, viewer = plt.subplots(1, log_likelihood.shape[-1])
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)

    for i in range(log_likelihood.shape[-1] * 2):  # log_likelihood.shape[-1]

        if i <= log_likelihood.shape[-1]-1:
            log_likelihood1 = log_likelihood[:, i]
            log_likelihood1 = log_likelihood1.reshape(N, N, N)
            ax = fig.add_subplot(2, log_likelihood.shape[-1], i + 1)
            # plot background
            #ax.imshow(im, extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
            #                      np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])
            if i ==0:
                first_result = tf.exp(log_likelihood1)[:, :, int(expected_data[i, 2])]
                imag = ax.contourf(X1[:, :, 0], X2[:, :, 0], first_result , levels=800, alpha=0.7
                                   , cmap='RdGy', vmin = 0.0, vmax = 0.05)
                imag = ax.imshow(first_result*10, cmap='RdGy', vmin=0.0, vmax=1,
                                 extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
                                         np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])
            else:
                result = tf.exp(log_likelihood1)[:, :, int(expected_data[i, 2])] - first_result
                first_result = tf.exp(log_likelihood1)[:, :, int(expected_data[i, 2])]
                imag = ax.contourf(X1[:, :, 0], X2[:, :, 0],
                                   result, levels=800, alpha=0.7
                                   , cmap='RdGy', vmin=0.0, vmax=0.05)
                imag = ax.imshow(np.abs(result), cmap='RdGy', vmin=0.0, vmax=0.5,
                                 extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
                                         np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])

            ax.scatter(history_data[-50:, 0], history_data[-50:, 1], marker='*', color='black', s=70)
            ax.scatter(expected_data[i, 0], expected_data[i, 1], marker='o', color='green', s=200)
            plt.colorbar(imag)

        else:
            log_likelihood1 = log_likelihood[:, i - log_likelihood.shape[-1]]
            log_likelihood1 = log_likelihood1.reshape(N, N, N)
            ax = fig.add_subplot(2, log_likelihood.shape[-1], i + 1, projection='3d')
            ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood1)[:, :, -1], levels=800, alpha=0.7, cmap='RdGy',
                        zdir='z', offset=zlim[-1])
            ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood1)[:, :, int(expected_data[i - log_likelihood.shape[-1], 2])] ,
                        levels=800, alpha=0.7, cmap='RdGy', zdir='z', offset=expected_data[i - log_likelihood.shape[-1], 2])
            ax.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood1)[:, :, 0], levels=800, alpha=0.7, cmap='RdGy',
                        zdir='z', offset=zlim[0])

            ax.scatter3D(history_data[-50:, 0], history_data[-50:, 1], history_data[-50:, 2], marker='*', color='black',
                         s=150)
            ax.scatter3D(expected_data[i - log_likelihood.shape[-1], 0], expected_data[i - log_likelihood.shape[-1], 1], expected_data[i - log_likelihood.shape[-1], 2], marker='o',
                         color='green', s=400)
            ax.set_zlim(minz, maxz)
            ax.view_init(+25, -140)


    fig.tight_layout()

    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig(f'{savepath}/test_batch{count}_{idx+1}.png')


    plt.close()

def plot_expected_2d_density(history_data, expected_data, model, curr_path, dec_dist_loc, savepath, count, idx):
    N = 100
    stacks = []
    normalized_data = np.concatenate([history_data, expected_data], axis=0)
    minx, maxx = tf.math.reduce_min(normalized_data[:, 0]), tf.math.reduce_max(normalized_data[:, 0])
    miny, maxy = tf.math.reduce_min(normalized_data[:, 1]), tf.math.reduce_max(normalized_data[:, 1])

    xlim = np.linspace(minx, maxx, N)
    ylim = np.linspace(miny, maxy, N)

    X1, X2 = np.meshgrid(xlim, ylim)
    arr = np.stack([X1.reshape(-1), X2.reshape(-1)], axis=1)
    arr2 = np.repeat(arr[:, np.newaxis, :], expected_data.shape[0], axis=1)
    #print(f'arr2 shape is {arr2.shape}')
    #print(f'expected_data shape is {expected_data.shape}')

    y, log_likelihood = model.bij_loc(arr2, dec_dist_loc, test=True)
    #print(f'log_likelihood shape is {log_likelihood.shape}')
    log_likelihood = log_likelihood[:, 0, :]

    fig = plt.figure(figsize=(12, 4))
    im = plt.imread(f'{curr_path}/data/map.png').transpose((1, 0, 2))  # background faults image
    # fig, viewer = plt.subplots(1, log_likelihood.shape[-1])
    # fig.tight_layout()

    for i in range(log_likelihood.shape[-1]):  # log_likelihood.shape[-1]
        #print(f'log_likelihood dim is {log_likelihood.shape}')
        log_likelihood1 = log_likelihood[:, i]
        log_likelihood1 = log_likelihood1.reshape(N, N)
        ax = fig.add_subplot(1, log_likelihood.shape[-1], i + 1)
        if i == 0:
            first_result = tf.exp(log_likelihood1)
            imag = ax.imshow(ndimage.rotate(first_result, -90).T*1.8, cmap='RdGy', vmin=0.0, vmax=1,
                             extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
                                     np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])
        else:
            result = tf.exp(log_likelihood1) - first_result
            first_result = tf.exp(log_likelihood1)
            imag = ax.imshow(ndimage.rotate(np.abs(result), -90).T, cmap='RdGy', vmin=0.0, vmax=0.004,
                             extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
                                     np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])

        # plot background
        #ax.imshow(im, extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]), np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])])
        #imag = ax.contourf(X1, X2, tf.exp(log_likelihood1) , levels=800, alpha=0.7,cmap='RdGy')
        #imag = ax.imshow(tf.exp(log_likelihood1) * 100, extent=[np.min(normalized_data[:, 0]), np.max(normalized_data[:, 0]),
        #                      np.min(normalized_data[:, 1]), np.max(normalized_data[:, 1])], origin='lower', cmap='RdGy')

        ax.scatter(history_data[-50:, 0], history_data[-50:, 1], marker='*', color='black', s=70)
        ax.scatter(expected_data[i, 0], expected_data[i, 1], marker='o', color='green', s=200)
        plt.colorbar(imag)

    #fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    fig.tight_layout()

    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig(f'{savepath}/test_batch{count}_{idx+1}.png')

    plt.close()


def plot_att_scores(att_weights_dec, savepath, count):
    for i in range(1,7,2):
        fig, ax = plt.subplots(3, 1,
                               figsize=(15, 5),
                               tight_layout=True)
        fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.8, wspace=0.1, hspace=0.1)

        layer = list(att_weights_dec.keys())[i]
        imag = ax[0].imshow(att_weights_dec[layer][0, 0][:10, :50])
        #ax[0].set_title(f'{layer} head 1, first in batch ')
        #ax[0].set_xlabel('Past events')
        #ax[0].set_ylabel('Future events')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(imag, cax=cax)
        img = ax[1].imshow(att_weights_dec[layer][1, 0][:10, :50])
        #ax[1].set_title(f'{layer} head 3, first in batch')
        #ax[1].set_xlabel('Past events')
        #ax[1].set_ylabel('Future events')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(img, cax=cax)

        img = ax[2].imshow(att_weights_dec[layer][2, 0][:10, :50])
        #ax[2].set_title(f'{layer} head 5, first in batch')
        #ax[2].set_xlabel('Past events')
        #ax[2].set_ylabel('Future events')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(img, cax=cax)
        plt.savefig(f'{savepath}/train_scoreresult{count}_{layer}.png')
        # plt.show(block=False)
        # plt.pause(5)
        plt.close()


