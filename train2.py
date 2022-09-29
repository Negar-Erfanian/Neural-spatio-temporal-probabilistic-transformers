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
from model.vizualization import *
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
print(f'gpu_num is {gpu_num}')

run_train = args.train

bij_layers = args.bij_layers
num_heads = args.num_heads
num_layers = args.num_layers
bijector_type = args.bijector_type

NF = args.NF
NFtrain = args.NFtrain
loc_layer_prob = args.loc_layer_prob
time_layer_prob = args.time_layer_prob
dropout_rate = args.dropout_rate

event_num = args.event_num
event_out = args.event_out
class_num = args.class_num
event_num_per = args.event_num_per
seqs = args.seqs
shauffled = args.shauffled
lookaheadmaskin = args.lookaheadmaskin
lookaheadmaskout = args.lookaheadmaskout

regularizer1 = 0.1
regularizer2 = 0.1

if dataset == 'earthquake':
    dim = 3
elif dataset in ['covid19', 'citibike', 'pinwheel']:
    dim = 2

all_experiments = 'experiment_results/'
if not os.path.exists(all_experiments):
    os.mkdir(all_experiments)
print(f'model_type is {model_type}')
# experiment path
if model_type == 'transformer':
    exp_path = all_experiments + f'{dataset}_' + f'{model_type}_' + f'{num_heads}heads_' + f'{num_layers}layers_' \
               + f'{event_num}events_' \
               + f'{event_out}eventsout_' + f'{batch_size}batches_' + f'{num_epochs}epochs_' \
               + f'{time_layer_prob}_softsign_{loc_layer_prob}_RealNVP' \
               + '_' + f'{dropout_rate}dropout_rate'+ f'_{shauffled}_shauffled' \
               #+ f'_{lookaheadmaskin}_lookaheadmaskin' + f'_{lookaheadmaskout}_lookaheadmaskout'


else:
    exp_path = all_experiments + f'{dataset}_' + f'{model_type}_' +f'{temporal_model}_' +f'{spatial_model}_' \
               + f'{event_num}events_' + f'{event_out}eventsout_'+ f'{batch_size}batches_'+ f'{num_epochs}epochs_'\
               + f'{regularizer1}regularizer1_'+ f'{regularizer2}regularizer2'+ f'_{shauffled}_shauffled'
if dataset == 'pinwheel':
    exp_path = exp_path + f'_{class_num}classes' + f'_{event_num_per}eventperclass'


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
        data_generation(event_num=event_num, event_out=event_out, dataset = dataset, event_num_per = event_num_per, class_num = class_num, seqs= seqs, shauffled=shauffled,  batch_size=batch_size)

    for i in range(5):
        gc.collect()

    time_vector = np.zeros([num_epochs, 1])  # time per epoch
    '''lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.9)'''
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)  # learning_rate = lr
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
            time_layer_prob = time_layer_prob,
            dropout_rate = dropout_rate
        )
    else:
        model = Spatiotemporal(temporal_model, spatial_model, dim)



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
    def batch_processing(batch, normalize = True):
        ds_in, ds_out = batch
        norm = Normalization(mean=mean, variance=std**2)
        ds_time_in, ds_locationmag_in, ds_timediff_in = ds_in[:, :, 0], ds_in[:, :, 1:dim+2], ds_in[:, :, dim+2][..., tf.newaxis]
        ds_locationmag_in = norm(ds_locationmag_in)
        #ds_locationmag_in = tf.divide(
        #    tf.subtract(ds_locationmag_in, tf.math.reduce_min(ds_locationmag_in, axis=1, keepdims=True)),
        #    tf.subtract(tf.math.reduce_max(ds_locationmag_in, axis=1, keepdims=True),
        #                tf.math.reduce_min(ds_locationmag_in, axis=1, keepdims=True))
        #)
        ds_time_in = tf.divide(
            tf.subtract(ds_time_in, tf.math.reduce_min(ds_time_in, axis=-1, keepdims=True)),
            tf.subtract(tf.math.reduce_max(ds_time_in, axis=-1, keepdims=True),
                        tf.math.reduce_min(ds_time_in, axis=-1, keepdims=True))
        )


        ds_time_out, ds_locationmag_out, ds_timediff_out = ds_out[:, :, 0], ds_out[:, :, 1:dim+2], ds_out[:, :, dim+2][..., tf.newaxis]
        ds_locationmag_out = norm(ds_locationmag_out)
        #ds_locationmag_out = tf.divide(
        #    tf.subtract(ds_locationmag_out, tf.math.reduce_min(ds_locationmag_out, axis=1, keepdims=True)),
        #    tf.subtract(tf.math.reduce_max(ds_locationmag_out, axis=1, keepdims=True),
        #                tf.math.reduce_min(ds_locationmag_out, axis=1, keepdims=True))
        #)
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
        if lookaheadmaskin:
            look_ahead_mask_in = create_look_ahead_mask(ds_time_in.shape[1])
        else:
            look_ahead_mask_in = None
        if lookaheadmaskout:
            look_ahead_mask_out = create_look_ahead_mask(ds_time_out.shape[1])
        else:
            look_ahead_mask_out = None
        return ds_in_stack, ds_out_stack, look_ahead_mask_in, look_ahead_mask_out

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    if run_train:
        train_log_dir = os.path.join(exp_path, 'logs', 'train')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_loss_metric_time = tf.keras.metrics.Mean('train_loss_time', dtype=tf.float32)
        train_loss_metric_space = tf.keras.metrics.Mean('train_loss_space', dtype=tf.float32)

        val_log_dir = os.path.join(exp_path, 'logs', 'val')
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        val_loss_metric = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        val_loss_metric_time = tf.keras.metrics.Mean('val_loss_time', dtype=tf.float32)
        val_loss_metric_space = tf.keras.metrics.Mean('val_loss_space', dtype=tf.float32)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            for train_batch in train_dataset:
                for i in range(10):
                    gc.collect()
                train_ds_in_stack, train_ds_out_stack, train_ds_in_lookaheadmask, train_ds_out_lookaheadmask = \
                    batch_processing(train_batch, dataset)

                train_ds_time_out, train_ds_loc_out, train_ds_mag_out, train_ds_timediff_out = train_ds_out_stack
                if model_type == 'transformer':
                    with tf.GradientTape() as tape:
                        dec_dist_out_time, ds_out_pred_time, bij_time, dec_dist_out_loc, ds_out_pred_loc, bij_loc, att_weights_dec, att_weights_enc = \
                            model(train_ds_in_stack, train_ds_out_stack, True, train_ds_in_lookaheadmask, train_ds_out_lookaheadmask) #train_ds_in_lookaheadmask, train_ds_out_lookaheadmask
                        tape.watch(model.trainable_variables)
                        loss_time = - tf.reduce_mean(bij_time.log_prob(train_ds_timediff_out))\
                                    #+regularizer1*tf.norm(tf.math.subtract(train_ds_timediff_out,ds_out_pred_time), ord=1)
                        loss_space = - tf.reduce_mean(bij_loc) \
                                     #+ regularizer2*tf.norm(tf.math.subtract(train_ds_loc_out,ds_out_pred_loc), ord=2)
                        loss = loss_time + loss_space
                        grads = tape.gradient(loss, model.trainable_variables)
                    #tvars = model.trainable_variables
                    #g_vars = [var for var in tvars]
                    #print(g_vars)
                    #print(model.summary())
                    #for i in range(len(grads)):
                    #  print(f'grads shape is {grads[i].shape}')


                else:
                    with tf.GradientTape() as tape:

                        loss_time_enc, loss_space_enc, train_expected_times,\
                        train_expected_locs, train_temporal_loglik, train_spatial_loglik= model(train_ds_in_stack, train_ds_out_stack)
                        loss_time = -tf.reduce_mean(train_temporal_loglik)\
                                    + regularizer1 * tf.norm(tf.math.subtract(train_ds_time_out, train_expected_times),ord=2)
                        loss_space = -tf.reduce_mean(train_spatial_loglik)\
                                     + regularizer2 * tf.norm(tf.math.subtract(train_ds_loc_out, train_expected_locs), ord=2)

                        loss = loss_time + loss_space
                        tape.watch(model.trainable_variables)
                        grads = tape.gradient(loss, model.trainable_variables)


                train_loss_metric.update_state(loss)
                train_loss_metric_time.update_state(loss_time)
                train_loss_metric_space.update_state(loss_space)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                print(f'we have train loss as {train_loss_metric.result().numpy()}')
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

                    val_ds_time_out, val_ds_loc_out, _, val_ds_timediff_out = val_ds_out_stack

                    val_ds_in_lookaheadmask = val_mask_in
                    val_ds_out_lookaheadmask = val_mask_out
                    if model_type =='transformer':
                        val_dec_dist_time, val_ds_out_pred_time, val_bij_time, val_dec_dist_loc, val_ds_out_pred_loc, val_bij_loc, val_att_weights_dec, val_att_weights_enc = \
                            model(val_ds_in_stack, val_ds_out_stack, False, val_ds_in_lookaheadmask, val_ds_out_lookaheadmask) #val_ds_in_lookaheadmask, val_ds_out_lookaheadmask
                        loss_val_time = -tf.reduce_mean(val_bij_time.log_prob(val_ds_timediff_out))\
                                        #+regularizer1 * tf.norm(tf.math.subtract(val_ds_timediff_out, val_ds_out_pred_time), ord=1)
                        loss_val_space = - tf.reduce_mean(val_bij_loc) \
                                         #+ regularizer2*tf.norm(tf.math.subtract(val_ds_loc_out,val_ds_out_pred_loc), ord=2)

                        loss_val = loss_val_time +loss_val_space
                        print(f'we have time prediction outputs and true outputs in validation as {val_ds_out_pred_time[0]} and {val_ds_timediff_out[0]}')


                    else:
                        loss_val_time_enc, loss_val_space_enc, val_expected_times, \
                        val_expected_locs, val_temporal_loglik, val_spatial_loglik = model(val_ds_in_stack, val_ds_out_stack)
                        loss_val_time = -tf.reduce_mean(val_temporal_loglik)+ regularizer1 * tf.norm(tf.math.subtract(val_ds_time_out, val_expected_times),ord=2)
                        loss_val_space = -tf.reduce_mean(val_spatial_loglik)+ regularizer2 * tf.norm(tf.math.subtract(val_ds_loc_out, val_expected_locs), ord=2)
                        loss_val = loss_val_time + loss_val_space
                        print(f'we have time prediction outputs and true outputs as {val_expected_times[:, 0]} and {val_ds_time_out[0]}')
                    val_loss_metric(loss_val)
                    val_loss_metric_time(loss_val_time)
                    val_loss_metric_space(loss_val_space)
                    #print(f'we have loc prediction outputs and true outputs in validation as {val_ds_out_pred_loc[0, :, 0]} and {val_ds_loc_out[0, :, 0]}')
                    #print('val loss is', val_loss_metric.result().numpy())
                valid_losses.append(val_loss_metric.result().numpy())
                with val_summary_writer.as_default():
                    tf.summary.scalar(
                        'val', val_loss_metric.result(), step=epoch)
                    tf.summary.scalar(
                        'val_time', val_loss_metric_time.result(), step=epoch)
                    tf.summary.scalar(
                        'val_space', val_loss_metric_space.result(), step=epoch)

                print("Epoch {:03d}: validation_loss: {:.3f}| validation_loss_time: {:.3f}| validation_loss_space: {:.3f}".format(epoch, val_loss_metric.result().numpy(), val_loss_metric_time.result().numpy(), val_loss_metric_space.result().numpy()))

                val_loss_metric.reset_states()
                val_loss_metric_time.reset_states()
                val_loss_metric_space.reset_states()

                for i in range(15):
                    gc.collect()

            # Saving logs
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'train', train_loss_metric.result(), step=epoch)
                tf.summary.scalar(
                    'train_time', train_loss_metric_time.result(), step=epoch)
                tf.summary.scalar(
                    'train_space', train_loss_metric_space.result(), step=epoch)

            print("Epoch {:03d}: train_loss: {:.3f}| train_loss_time: {:.3f}| train_loss_space: {:.3f} ".format(epoch, train_loss_metric.result().numpy(), train_loss_metric_time.result().numpy(), train_loss_metric_space.result().numpy()))

            train_loss_metric.reset_states()
            train_loss_metric_time.reset_states()
            train_loss_metric_space.reset_states()

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
        test_loss_metric_time = tf.keras.metrics.Mean('test_loss_time', dtype=tf.float32)
        test_loss_metric_space = tf.keras.metrics.Mean('test_loss_space', dtype=tf.float32)
        test_loss_metric_lst = []
        test_loss_metric_time_lst = []
        test_loss_metric_space_lst = []
        print(f'we are in test now and model is {model}')
        count = 0
        test_losses = []
        for test_batch in test_dataset:
            for i in range(5):
                gc.collect()

            ####################

            test_ds_in_stack, test_ds_out_stack, test_ds_in_lookaheadmask, test_ds_out_lookaheadmask = batch_processing(
                test_batch, dataset)
            test_ds_time_in, test_ds_loc_in, test_ds_mag_in, test_ds_timediff_in = test_ds_in_stack
            test_ds_time_out, test_ds_loc_out, test_ds_mag_out, test_ds_timediff_out = test_ds_out_stack
            aux_state_in  = test_ds_time_in
            #print(f'test_ds_loc_in is {test_ds_loc_in[0]}')
            #print(f'test_ds_mag_in is {test_ds_mag_in[0]}')
            #print(f'test_ds_time_in is {test_ds_time_in[0]}')
            #print(f'test_ds_loc_out is {test_ds_loc_out[0]}')
            #print(f'test_ds_mag_out is {test_ds_mag_out[0]}')
            #print(f'test_ds_time_out is {test_ds_time_out[0]}')
            #tf.concat([test_ds_mag_in, test_ds_mag_out[:,0,:][..., tf.newaxis]], axis = 1)
            aux_state_out = test_ds_time_out
            if model_type =='transformer':
                test_dec_dist_time, test_ds_out_pred_time, test_bij_time, test_dec_dist_loc, test_ds_out_pred_loc, test_bij_loc, test_att_weights_dec, test_att_weights_enc = model(
                    test_ds_in_stack, test_ds_out_stack, False, test_ds_in_lookaheadmask, test_ds_out_lookaheadmask) #test_ds_in_lookaheadmask, test_ds_out_lookaheadmask
                loss_test_time = -tf.reduce_mean(test_bij_time.log_prob(test_ds_timediff_out))#+regularizer1 * tf.norm(tf.math.subtract(test_ds_timediff_out, test_ds_out_pred_time), ord=1)
                loss_test_space = -tf.reduce_mean(test_bij_loc)#+ regularizer2*tf.norm(tf.math.subtract(test_ds_loc_out,test_ds_out_pred_loc), ord=2)
                loss_test = loss_test_time +loss_test_space  #+regularizer1*tf.norm(tf.math.subtract(test_ds_timediff_out,test_ds_out_pred_time), ord=1)+regularizer2*tf.norm(tf.math.subtract(test_ds_loc_out,test_ds_out_pred_loc), ord=2)
                '''experiments_figs_scores = exp_path +'/figures_train_scores/'
                if os.path.exists(experiments_figs_scores) == False:
                    os.mkdir(experiments_figs_scores)
                plot_att_scores(test_att_weights_dec, savepath = experiments_figs_scores, count = count)

                experiments_figs_time = exp_path + '/time_pred/'
                if os.path.exists(experiments_figs_time) == False:
                    os.mkdir(experiments_figs_time)
                for idx in range(len(test_ds_loc_out)):
                    plot_expectedtime(test_ds_timediff_in, test_ds_timediff_out, test_ds_out_pred_time,
                                      event_num, savepath=experiments_figs_time, count=count, idx = idx)'''
                '''experiments_figs_timeintensity = exp_path + '/time_intensity_pred/'
                if os.path.exists(experiments_figs_timeintensity) == False:
                    os.mkdir(experiments_figs_timeintensity)
                plot_expected_intensity(test_bij_time, test_ds_timediff_out, savepath=experiments_figs_timeintensity,
                                        count=count)'''

                curr_path = os.getcwd()
                experiments_figs_loc = exp_path + '/loc_pred/'
                if os.path.exists(experiments_figs_loc) == False:
                    os.mkdir(experiments_figs_loc)

                for idx in range(len(test_ds_loc_out)):

                    if dim ==2:
                        plot_expected_2d_density(history_data=test_ds_loc_in[idx], expected_data=test_ds_loc_out[idx], model=model,
                                                curr_path=curr_path, dec_dist_loc=test_dec_dist_loc, savepath=experiments_figs_loc,
                                                count=count, idx=idx)
                    else:
                        plot_expected_3d_density(history_data=test_ds_loc_in[idx], expected_data=test_ds_loc_out[idx],
                                                 model=model,
                                                 curr_path=curr_path, dec_dist_loc=test_dec_dist_loc,
                                                 savepath=experiments_figs_loc,
                                                 count=count, idx=idx)
            else:
                loss_test_time_enc, loss_test_space_enc, test_expected_times, \
                test_expected_locs, test_temporal_loglik, test_spatial_loglik = model(test_ds_in_stack, test_ds_out_stack, training = False)

                loss_test_time = -tf.reduce_mean(test_temporal_loglik)#+ regularizer1 * tf.norm(tf.math.subtract(test_ds_time_out, test_expected_times),ord=2)
                loss_test_space = -tf.reduce_mean(test_spatial_loglik)#+ regularizer2 * tf.norm(tf.math.subtract(test_ds_loc_out, test_expected_locs), ord=2)
                loss_test = loss_test_time + loss_test_space
                test_expected_times = np.transpose(test_expected_times)
                test_expected_time_diff = test_expected_times[:, 0][...,None]
                for i in range(test_expected_times.shape[-1]-1):
                    diff = (test_expected_times[:,i+1] - test_expected_times[:,i])[...,None]
                    test_expected_time_diff = np.append(test_expected_time_diff, diff, axis = 1)


                '''experiments_figs_time = exp_path + '/time_pred_benchmark/'
                if os.path.exists(experiments_figs_time) == False:
                    os.mkdir(experiments_figs_time)
                for idx in range(len(test_ds_loc_out)):
                    plot_expectedtime(test_ds_timediff_in, test_ds_timediff_out, test_expected_time_diff,
                                      event_num, savepath=experiments_figs_time, count=count, idx = idx)
                print(f'we have time prediction outputs and true outputs as {test_expected_times[0]} and {test_ds_time_out[0]}')'''
                curr_path = os.getcwd()
                '''experiments_figs_loc_gmm = exp_path + '/loc_pred_gmm/'
                if os.path.exists(experiments_figs_loc_gmm) == False:
                    os.mkdir(experiments_figs_loc_gmm)
               
                for idx in range(test_ds_time_out.shape[0]):
                    if dim == 2:
                        plot_expected_2d_density_gmm(test_ds_time_out[idx,:3,0][tf.newaxis], test_ds_time_in[idx,:,0][tf.newaxis], test_ds_loc_in[idx][tf.newaxis], test_ds_loc_out[idx,:3,:][tf.newaxis], aux_state_in = aux_state_in[idx][tf.newaxis], aux_state_out = aux_state_out[idx][tf.newaxis], model = model, curr_path = curr_path, savepath = experiments_figs_loc_gmm, count = count, idx = idx)
                    else:
                        plot_expected_3d_density_gmm(test_ds_time_out[idx,:3,0][tf.newaxis], test_ds_time_in[idx,:,0][tf.newaxis], test_ds_loc_in[idx][tf.newaxis], test_ds_loc_out[idx,:3,:][tf.newaxis], aux_state_in = aux_state_in[idx][tf.newaxis], aux_state_out = aux_state_out[idx][tf.newaxis], model = model , curr_path = curr_path, savepath = experiments_figs_loc_gmm, count = count, idx = idx)'''
            test_loss_metric(loss_test)
            test_loss_metric_time(loss_test_time)
            test_loss_metric_space(loss_test_space)
            test_loss_metric_lst.append(loss_test)
            test_loss_metric_time_lst.append(loss_test_time)
            test_loss_metric_space_lst.append(loss_test_space)

            test_losses.append(test_loss_metric.result().numpy())
            with test_summary_writer.as_default():
                tf.summary.scalar(
                    'test', test_loss_metric.result(), step=count)
                tf.summary.scalar(
                    'test_time', test_loss_metric_time.result(), step=count)
                tf.summary.scalar(
                    'test_loc', test_loss_metric_space.result(), step=count)


            #test_loss_metric.reset_states()
            #test_loss_metric_time.reset_states()
            #test_loss_metric_space.reset_states()

            for i in range(15):
                gc.collect()





            #####################

            count += 1
        print("Count {:03d}: test_loss: {:.3f}| test_loss_time: {:.3f}| test_loss_space: {:.3f}  ".format(count,
                                                                                                          test_loss_metric.result().numpy(),
                                                                                                          test_loss_metric_time.result().numpy(),
                                                                                                          test_loss_metric_space.result().numpy()))
        print("Count {:03d}: test_loss_std: {:.3f}| test_loss_time_std: {:.3f}| test_loss_space_std: {:.3f}  ".format(
            count, np.std(test_loss_metric_lst), np.std(test_loss_metric_time_lst), np.std(test_loss_metric_space_lst)))


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
