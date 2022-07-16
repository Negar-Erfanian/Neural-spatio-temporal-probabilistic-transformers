try:
    from tensorflow.keras.layers import Normalization
except ImportError:
    from keras.layers import Normalization
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
tfk = tf.keras

import os
import numpy as np
import time
import gc
import shutil
from tqdm import tqdm
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

from model.functions import create_look_ahead_mask
from model.transformer import Transformer
from model.benchmark_spatiotemporal import Spatiotemporal
from model.vizualization import plot_expectedtime, plot_expected_intensity, plot_expected_density
from data.load_data import data_generation

from utils import *

args, _ = get_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
dataset = args.dataset
model_type = args.model_type
temporal_model = args.temporal_model
spatial_model = args.spatial_model
lr = args.lr
gpu_num = args.gpu_num
remove_all = bool(args.remove_all)
desc = args.desc

run_train = args.train

bij_layers = args.bij_layers
num_heads = args.num_heads
num_layers = args.num_layers
bijector_type = args.bijector_type

NF = args.NF
NFtrain = args.NFtrain
loc_layer_prob = args.loc_layer_prob
time_layer_prob = args.time_layer_prob

event_num = args.event_num
event_out = args.event_out
regularizer1 = 0.1
regularizer2 = 0.1

if dataset == 'earthquake':
    dim = 3
elif dataset in ['covid19', 'citibike']:
    dim = 2

all_experiments = 'experiment_results/'
if not os.path.exists(all_experiments):
    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments +f'{dataset}_' + f'{num_heads}heads_' + f'{num_layers}layers_' + f'{event_num}events_' \
           + f'{event_out}events_out_' + f'{time_layer_prob}_softsign_{loc_layer_prob}_RealNVP_noreg'

if os.path.exists(exp_path) and remove_all:
    shutil.rmtree(exp_path)

if not os.path.exists(exp_path):
    os.mkdir(exp_path)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

tf.random.set_seed(0)  # 11


def train(num_epochs, batch_size, num_layers, num_heads, event_num, event_out, dataset, model_type, lr, exp_path):
    # Print the experiment setup:
    print('Experiment setup:')
    print('---> num_epochs: {}'.format(num_epochs))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset: {}'.format(dataset))
    print('---> Learning rate: {}'.format(lr))
    print('---> experiment path: {}'.format(exp_path))

    train_dataset, validation_dataset, test_dataset, dataset_NF, mean, std = \
        data_generation(event_num=event_num, event_out=event_out, dataset = dataset, batch_size=batch_size)

    for i in range(5):
        gc.collect()

    time_vector = np.zeros([num_epochs, 1])  # time per epoch

    optimizer = tf.keras.optimizers.Adam()  # learning_rate = lr
    train_losses = []
    valid_losses = []


    event_in = event_num - event_out
    if model_type == 'transformer':
        model = Transformer(
            num_layers=num_layers,
            embedding_dim_enc=64,
            embedding_dim_dec=64,
            num_heads=num_heads,
            fc_dim=32,
            dim_out_time=1,
            dim_out_loc=dim,
            max_positional_encoding_input=event_in,
            max_positional_encoding_target=event_out,
            time_layer_prob = time_layer_prob
        )
    else:
        model = Spatiotemporal(temporal_model, spatial_model)



    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    # not used
    @tf.function
    def monte_carlo_estimate_of_kl_divergence(n, q_sampler, q_density, p_density):
        x = q_sampler(n)
        KL_estimate = tf.reduce_mean(q_density(x) - p_density(x))
        return (KL_estimate)

    @tf.function
    def batch_processing(batch, dataset):
        ds_in, ds_out = batch
        norm = Normalization(mean=mean, variance=std**2)
        ds_time_in, ds_locationmag_in, ds_timediff_in = ds_in[:, :, 0], ds_in[:, :, 1:dim+2], ds_in[:, :, dim+2][..., tf.newaxis]
        ds_locationmag_in = norm(ds_locationmag_in)
        ds_time_in = tf.divide(
            tf.subtract(ds_time_in, tf.math.reduce_min(ds_time_in, axis=-1, keepdims=True)),
            tf.subtract(tf.math.reduce_max(ds_time_in, axis=-1, keepdims=True),
                        tf.math.reduce_min(ds_time_in, axis=-1, keepdims=True))
        )

        ds_time_out, ds_locationmag_out, ds_timediff_out = ds_out[:, :, 0], ds_out[:, :, 1:dim+2], ds_out[:, :, dim+2][..., tf.newaxis]
        ds_locationmag_out = norm(ds_locationmag_out)
        ds_time_out = tf.divide(
            tf.subtract(ds_time_out, tf.math.reduce_min(ds_time_out, axis=-1, keepdims=True)),
            tf.subtract(tf.math.reduce_max(ds_time_out, axis=-1, keepdims=True),
                        tf.math.reduce_min(ds_time_out, axis=-1, keepdims=True))
        )

        ds_location_in, ds_mag_in = ds_locationmag_in[:, :, :dim], ds_locationmag_in[:, :, dim][..., tf.newaxis]
        ds_location_out, ds_mag_out = ds_locationmag_out[:, :, :dim], ds_locationmag_out[:, :, dim][..., tf.newaxis]

        ds_time_in = ds_time_in[..., tf.newaxis]
        ds_time_out = ds_time_out[..., tf.newaxis]
        ds_in_stack = [ds_time_in, ds_location_in, ds_mag_in, ds_timediff_in]
        ds_out_stack = [ds_time_out, ds_location_out, ds_mag_out, ds_timediff_out]

        look_ahead_mask_in = create_look_ahead_mask(ds_time_in.shape[1])
        look_ahead_mask_out = create_look_ahead_mask(ds_time_out.shape[1])

        return ds_in_stack, ds_out_stack, look_ahead_mask_in, look_ahead_mask_out

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    if run_train:
        train_log_dir = os.path.join(exp_path, 'logs', 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

        val_log_dir = os.path.join(exp_path, 'logs', 'val')
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            for train_batch in tqdm(train_dataset):
                for i in range(10):
                    gc.collect()
                train_ds_in_stack, train_ds_out_stack, train_ds_in_lookaheadmask, train_ds_out_lookaheadmask = \
                    batch_processing(train_batch, dataset)

                train_ds_time_out, train_ds_loc_out, train_ds_mag_out, train_ds_timediff_out = train_ds_out_stack

                if model_type == 'transformer':
                    with tf.GradientTape() as tape:
                        dec_dist_out_time, ds_out_pred_time, bij_time, dec_dist_out_loc, ds_out_pred_loc, bij_loc, att_weights_dec, att_weights_enc = \
                            model(train_ds_in_stack, train_ds_out_stack, True, train_ds_in_lookaheadmask, train_ds_out_lookaheadmask)
                        tape.watch(model.trainable_variables)
                        loss = - tf.reduce_mean(bij_time.log_prob(train_ds_timediff_out)) - tf.reduce_mean(bij_loc)
                        # +regularizer1*tf.norm(tf.math.subtract(train_ds_timediff_out,ds_out_pred_time), ord=1)+ regularizer2*tf.norm(tf.math.subtract(train_ds_loc_out,ds_out_pred_loc), ord=2)
                        grads = tape.gradient(loss, model.trainable_variables)
                else:
                    with tf.GradientTape() as tape:
                        loss, train_expected_times, train_expected_loc_func = model(train_ds_in_stack, train_ds_out_stack)
                        tape.watch(model.trainable_variables)
                        grads = tape.gradient(loss, model.trainable_variables)

                train_loss_metric.update_state(loss)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_losses.append(train_loss_metric.result().numpy())

            for i in range(10):
                gc.collect()

            # Validation\
            if epoch % 10 == 0:
                print(f'we are in validation now where epoch is {epoch}')

                for val_batch in validation_dataset:
                    for i in range(15):
                        gc.collect()
                    val_ds_in_stack, val_ds_out_stack, val_mask_in, val_mask_out = batch_processing(val_batch, dataset)

                    _, val_ds_loc_out, _, val_ds_timediff_out = val_ds_out_stack
                    val_ds_in_lookaheadmask = val_mask_in
                    val_ds_out_lookaheadmask = val_mask_out
                    if model_type =='transformer':
                        val_dec_dist_time, val_ds_out_pred_time, val_bij_time, val_dec_dist_loc, val_ds_out_pred_loc, val_bij_loc, val_att_weights_dec_time, val_att_weights_enc = \
                            model(val_ds_in_stack, val_ds_out_stack, False, val_ds_in_lookaheadmask, val_ds_out_lookaheadmask)
                        loss_val = -tf.reduce_mean(val_bij_time.log_prob(val_ds_timediff_out)) - tf.reduce_mean(val_bij_loc)


                    else:
                        loss_val, val_expected_times, val_expected_loc_func = model()
                    val_loss_metric(loss_val)

                    #print(f'we have time prediction outputs and true outputs in validation as {val_ds_out_pred_time[0]} and {val_ds_timediff_out[0]}')
                    #print(f'we have loc prediction outputs and true outputs in validation as {val_ds_out_pred_loc[0, :, 0]} and {val_ds_loc_out[0, :, 0]}')
                    #print('val loss is', val_loss_metric.result().numpy())
                valid_losses.append(val_loss_metric.result().numpy())
                with val_summary_writer.as_default():
                    tf.summary.scalar(
                        'val', val_loss_metric.result(), step=epoch)

                print("Epoch {:03d}: val: {:.3f}  ".format(epoch, val_loss_metric.result().numpy()))

                val_loss_metric.reset_states()

                for i in range(15):
                    gc.collect()

            # Saving logs
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'train', train_loss_metric.result(), step=epoch)

            print("Epoch {:03d}: train: {:.3f}  ".format(epoch, train_loss_metric.result().numpy()))

            train_loss_metric.reset_states()

            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

            epoch_end = time.time()
            time_vector[epoch] = epoch_end - epoch_start
            np.save(os.path.join(exp_path, 'time_vector.npy'), time_vector)
            print('epoch time:{}'.format(time_vector[epoch]))
            for i in range(15):
                gc.collect()

    if not run_train:

        test_log_dir = os.path.join(exp_path, 'logs', 'test')
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        test_loss_metric = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        print(f'we are in test now and model is {model}')
        count = 0
        test_losses = []
        for test_batch in train_dataset:
            for i in range(5):
                gc.collect()

            ####################

            test_ds_in_stack, test_ds_out_stack, test_ds_in_lookaheadmask, test_ds_out_lookaheadmask = batch_processing(
                test_batch, dataset)

            test_ds_time_in, test_ds_loc_in, test_ds_mag_in, test_ds_timediff_in = test_ds_in_stack

            test_ds_time_out, test_ds_loc_out, test_ds_mag_out, test_ds_timediff_out = test_ds_out_stack
            if model_type =='transformer':
                test_dec_dist_time, test_ds_out_pred_time, test_bij_time, test_dec_dist_loc, test_ds_out_pred_loc, test_bij_loc, test_att_weights_dec_time, test_att_weights_enc = model(
                    test_ds_in_stack, test_ds_out_stack, False, test_ds_in_lookaheadmask, test_ds_out_lookaheadmask)

                loss_test = -tf.reduce_mean(test_bij_time.log_prob(test_ds_timediff_out)) + \
                            -tf.reduce_mean(test_bij_loc) # +regularizer1*tf.norm(tf.math.subtract(test_ds_timediff_out,test_ds_out_pred_time), ord=1)+regularizer2*tf.norm(tf.math.subtract(test_ds_loc_out,test_ds_out_pred_loc), ord=2)
            else:
                loss_test, test_expected_times, test_expected_loc_func = model()

            test_loss_metric(loss_test)

            print(
                f'we have time prediction outputs and true outputs in test as {test_ds_out_pred_time[0]} and {test_ds_timediff_out[0]}')
            print(
                f'we have loc prediction outputs and true outputs in test as {test_ds_out_pred_loc[0, :, 0]} and {test_ds_loc_out[0, :, 0]}')
            print('test loss is', test_loss_metric.result().numpy())
            test_losses.append(test_loss_metric.result().numpy())
            with test_summary_writer.as_default():
                tf.summary.scalar(
                    'test', test_loss_metric.result(), step=count)

            print("Count {:03d}: test: {:.3f}  ".format(count, test_loss_metric.result().numpy()))

            test_loss_metric.reset_states()

            for i in range(15):
                gc.collect()

            print(f'-----------------')
            '''experiments_figs_scores = exp_path +'/figures_train_scores/'
            if os.path.exists(experiments_figs_scores) == False:
                os.mkdir(experiments_figs_scores)
            plot_att_scores(test_att_weights_dec_time, savepath = experiments_figs_scores, count = count, mag = mag)'''

            experiments_figs_time = exp_path + '/time_pred/'
            if os.path.exists(experiments_figs_time) == False:
                os.mkdir(experiments_figs_time)
            plot_expectedtime(test_ds_timediff_in, test_ds_timediff_out, test_ds_out_pred_time, test_bij_time,
                              event_num, savepath=experiments_figs_time, count=count)

            experiments_figs_timeintensity = exp_path + '/time_intensity_pred/'
            if os.path.exists(experiments_figs_timeintensity) == False:
                os.mkdir(experiments_figs_timeintensity)
            plot_expected_intensity(test_bij_time, test_ds_timediff_out, savepath=experiments_figs_timeintensity,
                                    count=count)

            curr_path = os.getcwd()
            experiments_figs_loc = exp_path + '/loc_pred/'
            if os.path.exists(experiments_figs_loc) == False:
                os.mkdir(experiments_figs_loc)

            idx = -1

            plot_expected_density(history_data=test_ds_loc_in[idx], expected_data=test_ds_loc_out[idx], model=model,
                                  curr_path=curr_path, dec_dist_loc=test_dec_dist_loc, savepath=experiments_figs_loc,
                                  count=count, idx=idx)

            #####################

            count += 1


if __name__ == '__main__':
    train(num_epochs,
          batch_size,
          num_layers,
          num_heads,
          event_num,
          event_out,
          dataset,
          model_type,
          lr,
          exp_path)
