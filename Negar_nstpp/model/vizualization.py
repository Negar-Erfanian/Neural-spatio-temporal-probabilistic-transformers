import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

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


def plot_expected_density(history_data, expected_data, model, curr_path, dec_dist_loc, savepath, count, idx):
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


def _plot(names, samples, rows=1, legend=False):
    cols = int(len(samples) / rows)
    f, arr = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    print(arr)
    i = 0
    for r in range(rows):
        for c in range(cols):
            res = samples[i]
            X, Y = res[..., 0].numpy(), res[..., 1].numpy()
            if rows == 1:
                p = arr[c]
            else:
                p = arr[r, c]
            p.scatter(X, Y, s=10, color='red')
            p.set_xlim([-5, 5])
            p.set_ylim([-5, 5])
            p.set_title(names[i])

            i += 1
    plt.show()


def make_samples(loc_dist, base_dist, n_samples=1000):
    x = base_dist.sample((n_samples))
    samples = [x]
    names = [base_dist.name]
    for bijector in reversed(loc_dist.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)
    return names, samples


def plot_NFdensity(loc_dist, base_dist, n_samples=1000):
    z = base_dist.sample(n_samples)

    xx = loc_dist.bijector(z)

    X1, X2, X3 = np.meshgrid(xx[:, 0], xx[:, 1], xx[:, 2])
    print(X1.shape)

    prob1 = trainable_distribution.prob(np.stack([X1, X2, X3], axis=-1))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X1[:, :, 0], X2[:, :, 0], prob1[:, :, 0], 200, cmap='RdGy')
    plt.show()


def plot_NF(NF_batch, loc_dist, base_dist, n_samples=1000):
    X = loc_dist.bijector(base_dist.sample(n_samples))
    plt.figure()
    plt.title('generated out of noise')
    plt.scatter(X[:, 0], X[:, 1], s=10, color='red')
    plt.xlim(-2, 2.5)
    plt.ylim(-3.5, 2.5)
    plt.figure()
    plt.title('real data')
    plt.scatter(NF_batch[:, 0], NF_batch[:, 1], s=10, color='blue')
    plt.xlim(-2, 2.5)

    plt.ylim(-3.5, 2.5)
    plt.show()


def plot_hist(test_ds_out_pred, test_ds_time_out, test_ds_out_pred_loc, test_ds_loc_out, test_ds_out_mask, exp_path,
              count, figures_test="figures_test"):
    fig, ax = plt.subplots(2, 2,
                           figsize=(FIGSIZE, FIGSIZE),
                           tight_layout=True)
    for k in range(test_ds_out_mask.shape[0]):
        print(
            f'we have prediction outputs and true outputs in test as {tf.math.argmax(test_ds_out_pred[k], axis=-1)[..., tf.newaxis][test_ds_out_mask[k]]} and {test_ds_time_out[k][..., tf.newaxis][test_ds_out_mask[k]]}')
        print(
            f'we have prediction outputs and true outputs in test as {test_ds_out_pred_loc[k, :, 0][..., tf.newaxis][test_ds_out_mask[k]]} and {test_ds_loc_out[k, :, 0][..., tf.newaxis][test_ds_out_mask[k]]}')
        arr1 = tf.math.argmax(test_ds_out_pred[k], axis=-1)[..., tf.newaxis][test_ds_out_mask[k]].numpy()
        unique1, counts1 = np.unique(arr1, return_counts=True)
        x_min1, x_max1 = np.min(unique1), np.max(unique1)
        y_min1, y_max1 = np.min(counts1), np.max(counts1)

        arr2 = test_ds_time_out[k][..., tf.newaxis][test_ds_out_mask[k]].numpy()
        unique2, counts2 = np.unique(arr2, return_counts=True)
        x_min2, x_max2 = np.min(unique2), np.max(unique2)
        y_min2, y_max2 = np.min(counts2), np.max(counts2)

        x_min = np.minimum(x_min1, x_min2)
        y_min = np.minimum(y_min1, y_min2)
        x_max = np.maximum(x_max1, x_max2)
        y_max = np.maximum(y_max1, y_max2)

        ax[k, 0].hist(arr1)
        ax[k, 0].set_title('prediction')
        ax[k, 0].set_xlim((x_min, x_max))
        ax[k, 0].set_ylim((y_min, y_max))

        ax[k, 1].hist(arr2)
        ax[k, 1].set_title('true_vals')
        ax[k, 1].set_xlim((x_min, x_max))
        ax[k, 1].set_ylim((y_min, y_max))

    plt.savefig(f'{exp_path}/{figures_test}/test_result{count}-{k}.png')


def plot_density(test_ds_out_pred_loc, test_ds_loc_out, test_ds_time_out, test_ds_out_mask, savepath, count,
                 dataset_name="Earthquake"):
    N = 20

    x = np.linspace(BBOXES[dataset_name][0], BBOXES[dataset_name][1], N)
    y = np.linspace(BBOXES[dataset_name][2], BBOXES[dataset_name][3], N)
    z = np.linspace(BBOXES[dataset_name][4], BBOXES[dataset_name][5], N)

    s = np.stack([x, y, z], axis=1)

    X, Y, Z1 = np.meshgrid(s[:, 0], s[:, 1], s[:, 2])  # (N,N,N,N) for each
    S = np.stack([X.reshape(-1), Y.reshape(-1), Z1.reshape(-1)], axis=1)  # (N**4,4)

    if not MAPS[dataset_name]:
        map_img = plt.imread(MAPS[dataset_name])
        fig, axes = plt.subplots(2, 2, figsize=(10, 5))

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.01, hspace=0.02)
        for i in range(2):
            for j in range(2):
                axes[i, j].imshow(map_img, zorder=0, extent=tuple(np.array(BBOXES[dataset_name][:4]).T))

    else:
        fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE, FIGSIZE))

    for i in range(2):
        arr_t = test_ds_time_out[i][..., tf.newaxis][test_ds_out_mask[i]].numpy()
        arr1 = test_ds_out_pred_loc[i][test_ds_out_mask[i].reshape(-1)].numpy()
        arr2 = test_ds_loc_out[i][test_ds_out_mask[i].reshape(-1)].numpy()

        axes[i, 0].scatter(arr1[:, 0], arr1[:, 1], c=arr_t, marker='o')
        axes[i, 0].scatter(arr2[:, 0], arr2[:, 1], c=arr_t, marker='*')
        # axes[i,0].set_xlim(BBOXES[dataset_name][0], BBOXES[dataset_name][1])
        # axes[i,0].set_ylim(BBOXES[dataset_name][2], BBOXES[dataset_name][3])

        axes[i, 1].scatter(arr1[:, 0], arr1[:, 2], c=arr_t, marker='o')
        axes[i, 1].scatter(arr2[:, 0], arr2[:, 2], c=arr_t, marker='*')
        # axes[i,1].set_xlim(BBOXES[dataset_name][0], BBOXES[dataset_name][1])
        # axes[i,1].set_ylim(BBOXES[dataset_name][4], BBOXES[dataset_name][5])

    plt.savefig(f'{savepath}/test_result_loc{count}.png')
    # plt.show()
    '''i = 0
        #print('Z1[:,:,i] is',Z1[:,:,i])

    interested_region = []
    interested_indices = []
    for l in range(N):
        if z[l]> coords[-1, 2]-7 and z[l]< coords[-1, 2]+7:
            interested_indices.append(l)
            interested_region.append(z[l])
            print('Z1[0,0,l] is ',z[l])
    #print('interested_indices are ', interested_indices)

    k = 0
    for i in range(m1):
        for j in range(m2):
            axes[i,j].contourf(X[:,:,0], Y[:,:,0], Z[:,:,interested_indices[k]], levels=20, alpha=0.7, cmap='RdGy')
            axes[i,j].set_title(f'at depth {round(interested_region[k],2)}')
            if k<len(interested_indices)-1:
                k+=1
                print(k)


    new_coords = []
    #print('are we good?',coords.shape)
    for ll in range(coords.shape[0]-1):
        if coords[ll, 2]<=interested_region[0] and coords[ll, 2]>=interested_region[-1]:
            new_coords.append(list(coords[ll, :2]))
    #print('new_coords',np.array(new_coords).shape)
    new_coords = np.array(new_coords)

    if new_coords.ndim >1:
        for i in range(m1):
            for j in range(m2):

                axes[i,j].scatter(new_coords[:, 0], new_coords[:, 1], s=20, alpha=1.0, marker="x", color="k")
                axes[i,j].scatter(coords[-1, 0], coords[-1, 1], s=20, alpha=1.0, marker="o", color="r")
                axes[i,j].set_xlim(BBOXES[dataset_name][0], BBOXES[dataset_name][1])
                axes[i,j].set_ylim(BBOXES[dataset_name][2], BBOXES[dataset_name][3])

        fig.suptitle(f'next event at depth {round(coords[-1, 2],2)} after {coords.shape[0]-1} events',x=0.2, y=.05, horizontalalignment='left', verticalalignment='top', fontsize = 15)

            #if text:
                #txt = ax.text(0.15, 0.9, text,
                              #horizontalalignment="center",
                              #verticalalignment="center",
                              #transform=ax.transAxes,
                              #size=16,
                              #color='white')
                #txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='black')])

        #plt.axis('off')
        os.makedirs(os.path.join(savepath, f"figs_{i}"), exist_ok=True)
        #np.savez(f"{savepath}/figs/data{index}.npz", **{"X": X, "Y": Y, "Z": Z, "spatial_locations": coords})
        plt.savefig(os.path.join(savepath, f"density{index}.png"), bbox_inches='tight', dpi=DPI)
        plt.close()'''


def plt_loc_density(normalized_data, bij, best_base_dist):
    minx, maxx = tf.math.reduce_min(normalized_data[:, 0]), tf.math.reduce_max(normalized_data[:, 0])
    miny, maxy = tf.math.reduce_min(normalized_data[:, 1]), tf.math.reduce_max(normalized_data[:, 1])
    minz, maxz = tf.math.reduce_min(normalized_data[:, 2]), tf.math.reduce_max(normalized_data[:, 2])

    N = 100
    xlim = np.linspace(minx, maxx, N)
    ylim = np.linspace(miny, maxy, N)
    zlim = np.linspace(minz, maxz, N)

    # new_f = normalized_data[:,2]
    # normalized_data_shallow = normalized_data[new_f<=1]
    # normalized_data_med = normalized_data[new_f<=3]
    # new_f_med = normalized_data_med[:,2]
    # normalized_data_med = normalized_data_med[new_f_med>1.5]
    # normalized_data_deep = normalized_data[new_f>2]
    X1, X2, X3 = np.meshgrid(xlim, ylim, zlim)
    print(X1.shape)
    _, log_likelihood = bij(np.stack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)], axis=1), best_base_dist,
                            training=True, test=False)

    log_likelihood = log_likelihood.reshape(N, N, N)

    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    for i in range(10):
        plt.contourf(X1[:, :, 0], X2[:, :, 0], tf.exp(log_likelihood)[:, :, i * 10], levels=2000, alpha=0.7,
                     cmap='RdGy');
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], color="b", s=30)
        plt.colorbar()
        plt.show()

    # x = [] # Some array of images
    # fig = plt.figure(figsize=(5,5))
    # viewer = fig.add_subplot(111)
    # plt.ion() # Turns interactive mode on (probably unnecessary)
    # fig.show() # Initially shows the figure

    # i = 0
    # while True:

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # plt.contourf(X1[:,:,0], X2[:,:,0], tf.exp(log_likelihood)[:,:,i*10], levels=2000, alpha=0.7, cmap='RdGy');
    # plt.scatter(normalized_data_shallow[:, 0], normalized_data_shallow[:, 1], color="b",s=30)
    # plt.colorbar()
    # viewer.clear() # Clears the previous image
    # viewer.contour(X1[:,:,0], X2[:,:,0], tf.exp(log_likelihood)[:,:,i], levels=2000, alpha=0.7, cmap='RdGy');
    # plt.pause(.01) # Delay in seconds
    # fig.canvas.draw() #
    # i+=1
    # if i >=N:
    #    i=0

    '''fig = plt.figure(figsize=(15, 10))
    #ax = plt.axes(projection='3d')
    plt.contour(X1[:,:,0], X2[:,:,0], tf.exp(log_likelihood)[:,:,15], levels=2000, alpha=0.7, cmap='RdGy');
    #plt.scatter(normalized_data_med[:, 0], normalized_data_med[:, 1], color="b",s=50)
    plt.colorbar()

    fig = plt.figure(figsize=(15, 10))
    #ax = plt.axes(projection='3d')
    plt.contour(X1[:,:,0], X2[:,:,0], tf.exp(log_likelihood)[:,:,40], levels=2000, alpha=0.7, cmap='RdGy');
    #plt.scatter(normalized_data_deep[:, 0], normalized_data_deep[:, 1], color="b",s=50)
    plt.colorbar()'''


