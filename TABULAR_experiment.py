import argparse
import random
import time
import pickle
import os
import json

import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow_probability as tfp

if __package__ is None or __package__ == '':
    from utils import generate_init_inducing_points_tabular, plot_tabular, generate_init_inducing_points, import_rotated_mnist, \
                    print_trainable_vars, parse_opt_regime, compute_bias_variance_mean_estimators, \
                    make_checkpoint_folder, pandas_res_saver, latent_samples_SVGPVAE, latent_samples_VAE_full_train
    from VAE_utils import mnistVAE, mnistCVAE, SVIGP_Hensman_decoder, tabularVAE
    from SVGPVAE_model import batching_predict_SVGPVAE, TabularDataSVGP, forward_pass_SVGPVAE_mnist, forward_pass_standard_VAE, \
                            mnistSVGP, forward_pass_standard_VAE_rotated_mnist, \
                            batching_encode_SVGPVAE, batching_encode_SVGPVAE_full, \
                            forward_pass_SVGPVAE_tabular, predict_CVAE
    from GPVAE_Casale_model import encode, casaleGP, forward_pass_Casale, predict_test_set_Casale, sort_train_data
    from SVIGP_Hensman_model import SVIGP_Hensman, forward_pass_deep_SVIGP_Hensman, predict_deep_SVIGP_Hensman
else:
    from .utils import generate_init_inducing_points_tabular, plot_tabular, generate_init_inducing_points, import_rotated_mnist, \
                    print_trainable_vars, parse_opt_regime, compute_bias_variance_mean_estimators, \
                    make_checkpoint_folder, pandas_res_saver, latent_samples_SVGPVAE, latent_samples_VAE_full_train
    from .VAE_utils import mnistVAE, mnistCVAE, SVIGP_Hensman_decoder, tabularVAE
    from .SVGPVAE_model import batching_predict_SVGPVAE, TabularDataSVGP, forward_pass_SVGPVAE_mnist, forward_pass_standard_VAE, \
                            mnistSVGP, forward_pass_SVGPVAE_tabular, \
                            batching_encode_SVGPVAE, batching_encode_SVGPVAE_full, \
                            bacthing_predict_SVGPVAE_rotated_mnist, predict_CVAE

    from .GPVAE_Casale_model import encode, casaleGP, forward_pass_Casale, predict_test_set_Casale, sort_train_data
    from .SVIGP_Hensman_model import SVIGP_Hensman, forward_pass_deep_SVIGP_Hensman, predict_deep_SVIGP_Hensman


tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def load_mnist_data(args, ending):
    MNIST_path = args.mnist_data_path
    train_data_dict = load_mnist_dict(ending, MNIST_path, 'train_data')
    eval_data_dict = load_mnist_dict(ending, MNIST_path, 'eval_data')
    test_data_dict = load_mnist_dict(ending, MNIST_path, 'test_data')
    return train_data_dict, eval_data_dict, test_data_dict

def load_mnist_dict(ending, MNIST_path, data_name):
    data_dict = pickle.load(open(MNIST_path + data_name + ending, 'rb'))
    data_dict['data_Y'] = data_dict.pop('images')
    data_dict['aux_X'] = data_dict.pop('aux_data')
    return data_dict

def tensor_slice(data_dict, batch_size, placeholder):
    data_Y = tf.data.Dataset.from_tensor_slices(data_dict['data_Y'])
    data_X = tf.data.Dataset.from_tensor_slices(data_dict['aux_X'])
    if placeholder:
        batch_size_placeholder = tf.compat.v1.placeholder(dtype=tf.int64, shape=())
    else:
        batch_size_placeholder = batch_size
    data = tf.data.Dataset.zip((data_Y, data_X)).batch(batch_size_placeholder)
    return data, batch_size_placeholder

def run_experiment_SVGPVAE(train_data_dict, eval_data_dict, test_data_dict,
    L, q, batch_size, nr_epochs, patience, n_neurons, dropout, activation, verbose, elbo_arg, M,
    nr_inducing_units, nr_inducing_per_unit, RE_cols, aux_cols, init_PCA=True,
    ip_joint=True, GP_joint=True, ov_joint=True,
    disable_gpu=True, beta_arg=0.001, lr_arg=0.001, alpha_arg=0.99, base_dir=os.getcwd(), expid='debug_TABULAR',
    jitter=0.000001, object_kernel_normalize=False, save=False, save_latents=False,
    save_model_weights=False, show_pics=False, kappa_squared=0.020, clip_qs=True,
    GECO=True, bias_analysis=False, opt_regime=['joint'], test_set_metrics=False,
    ram=1.0):
    """
    Function with tensorflow graph and session for SVGPVAE experiments on tabular data.
    For description of SVGPVAE see chapter 7 in SVGPVAE.tex

    :param args:
    :return:
    """
    # Problem with local TF setup, works fine in Google Colab
    if disable_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if save:
        # Make a folder to save everything
        extra = elbo_arg + "_" + str(beta_arg)
        chkpnt_dir = make_checkpoint_folder(base_dir, expid, extra)
        pic_folder = chkpnt_dir + "pics/"
        res_file = chkpnt_dir + "res/ELBO_pandas"
        res_file_GP = chkpnt_dir + "res/ELBO_GP_pandas"
        if "SVGPVAE" in elbo_arg:
            res_file_VAE = chkpnt_dir + "res/ELBO_VAE_pandas"
        print("\nCheckpoint Directory:\n" + str(chkpnt_dir) + "\n")

        # json.dump(args_dict, open(chkpnt_dir + "/json", "wt"))

    # train_data_dict, eval_data_dict, test_data_dict = load_mnist_data(args, ending = dataset + '.p')

    graph = tf.Graph()
    with graph.as_default():
        # ====================== 1) import data ======================
        train_data, _ = tensor_slice(train_data_dict, batch_size, placeholder=False)
        N_train = train_data_dict['data_Y'].shape[0]
        N_eval = eval_data_dict['data_Y'].shape[0]
        N_test = test_data_dict['data_Y'].shape[0]

        # eval data
        eval_data, eval_batch_size_placeholder = tensor_slice(eval_data_dict, batch_size, placeholder=True)

        # test data
        test_data, test_batch_size_placeholder = tensor_slice(test_data_dict, batch_size, placeholder=True)

        # init iterator
        iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(train_data),
            tf.compat.v1.data.get_output_shapes(train_data)
        )
        training_init_op = iterator.make_initializer(train_data)
        eval_init_op = iterator.make_initializer(eval_data)
        test_init_op = iterator.make_initializer(test_data)

        # get the batch
        input_batch = iterator.get_next()

        # ====================== 2) build ELBO graph ======================

        # init VAE object
        if elbo_arg == "CVAE":
            VAE = mnistCVAE(L=L)
        else:
            VAE = tabularVAE(input_batch[0].shape[1], L, n_neurons, dropout, activation)
        beta = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

        # placeholders
        y_shape = (None,) + train_data_dict['data_Y'].shape[1:]
        train_aux_X_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, len(RE_cols) + len(aux_cols) + M))
        train_data_Y_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=y_shape)
        test_aux_X_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, len(RE_cols) + len(aux_cols) + M))
        test_data_Y_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=y_shape)

        if "SVGPVAE" in elbo_arg:  # SVGPVAE
            inducing_points_init = generate_init_inducing_points_tabular(
                                                                 train_data_dict,
                                                                 RE_cols,
                                                                 aux_cols,
                                                                 nr_aux_units=nr_inducing_units,
                                                                 n_samp_per_aux=nr_inducing_per_unit,
                                                                 PCA=init_PCA, M=M)
            titsias = 'Titsias' in elbo_arg
            ip_joint = not ip_joint
            GP_joint = not GP_joint
            if ov_joint:
                if init_PCA:  # use PCA embeddings for initialization of object vectors
                    PC_cols = train_data_dict['aux_X'].columns[train_data_dict['aux_X'].columns.str.startswith('PC')]
                    object_vectors_init = train_data_dict['aux_X'].groupby('z0')[PC_cols].mean()
                else:  # initialize object vectors randomly
                    object_vectors_init = np.random.normal(0, 1.5, q * M).reshape(q, M)
            else:
                object_vectors_init = None

            # init SVGP object
            SVGP_ = TabularDataSVGP(titsias=titsias, fixed_inducing_points=ip_joint,
                              initial_inducing_points=inducing_points_init,
                              fixed_gp_params=GP_joint, object_vectors_init=object_vectors_init, name='main',
                              jitter=jitter, N_train=N_train,
                              L=L, K_obj_normalize=object_kernel_normalize,
                              RE_cols=RE_cols, aux_cols=aux_cols)

            # forward pass SVGPVAE
            C_ma_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
            lagrange_mult_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
            alpha_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

            elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, recon_data_Y, \
            inside_elbo_recon, inside_elbo_kl, latent_samples, \
            C_ma, lagrange_mult, mean_vectors = forward_pass_SVGPVAE_tabular(input_batch,
                                                                     beta=beta,
                                                                     vae=VAE,
                                                                     svgp=SVGP_,
                                                                     C_ma=C_ma_placeholder,
                                                                     lagrange_mult=lagrange_mult_placeholder,
                                                                     alpha=alpha_placeholder,
                                                                     kappa=np.sqrt(kappa_squared),
                                                                     clipping_qs=clip_qs,
                                                                     GECO=GECO,
                                                                     bias_analysis=bias_analysis)

            # forward pass standard VAE (for training regime from CASALE: VAE-GP-joint)
            recon_loss_VAE, KL_term_VAE, elbo_VAE, \
            recon_data_Y_VAE, qnet_mu_VAE, qnet_var_VAE, \
            latent_samples_VAE = forward_pass_standard_VAE(input_batch, vae=VAE)

        elif elbo_arg == "VAE" or elbo_arg == "CVAE":  # plain VAE or CVAE
            CVAE = elbo_arg == "CVAE"

            recon_loss, KL_term, elbo, \
            recon_data_Y, qnet_mu, qnet_var, latent_samples = forward_pass_standard_VAE(input_batch,
                                                                                        vae=VAE,
                                                                                        CVAE=CVAE)

        else:
            raise ValueError
        
        if "SVGPVAE" in elbo_arg:
            train_encodings_means_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, L))
            train_encodings_vars_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, L))

            qnet_mu_train, qnet_var_train, _ = batching_encode_SVGPVAE(input_batch, vae=VAE,
                                                                    clipping_qs=clip_qs)
            recon_data_Y_test, \
            recon_loss_test = batching_predict_SVGPVAE(input_batch,
                                                       vae=VAE,
                                                       svgp=SVGP_,
                                                       qnet_mu=train_encodings_means_placeholder,
                                                       qnet_var=train_encodings_vars_placeholder,
                                                       aux_data_train=train_aux_X_placeholder)

            # GP diagnostics
            GP_l, GP_amp, GP_ov, GP_ip = SVGP_.variable_summary()

        # bias analysis
        if bias_analysis:
            means, vars = batching_encode_SVGPVAE_full(train_data_Y_placeholder,
                                                       vae=VAE, clipping_qs=clip_qs)
            mean_vector_full_data = []
            for l in range(L):
                mean_vector_full_data.append(SVGP_.mean_vector_bias_analysis(index_points=train_aux_X_placeholder,
                                                                             y=means[:, l], noise=vars[:, l]))

        if save_latents:
            if "SVGPVAE" in elbo_arg:
                latent_samples_full = latent_samples_SVGPVAE(train_data_Y_placeholder, train_aux_X_placeholder,
                                                             vae=VAE, svgp=SVGP_, clipping_qs=clip_qs)
            else:
                latent_samples_full = latent_samples_VAE_full_train(train_data_Y_placeholder,
                                                                    vae=VAE, clipping_qs=clip_qs)
        # conditional generation for CVAE
        if elbo_arg == "CVAE":
            recon_data_Y_test, recon_loss_test = predict_CVAE(images_train=train_data_Y_placeholder,
                                                              images_test=test_data_Y_placeholder,
                                                              aux_data_train=train_aux_X_placeholder,
                                                              aux_data_test=test_aux_X_placeholder,
                                                              vae=VAE, test_indices=test_data_dict['aux_X'][:, 0])

        # ====================== 3) optimizer ops ======================
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        if GECO:  # minimizing GECO objective
            gradients = tf.gradients(elbo, train_vars)
        else:  # minimizing negative elbo
            gradients = tf.gradients(-elbo, train_vars)

        optim_step = optimizer.apply_gradients(grads_and_vars=zip(gradients, train_vars),
                                               global_step=global_step)

        # ====================== 4) Pandas saver ======================
        if save:
            res_vars = [global_step,
                        elbo,
                        recon_loss,
                        KL_term,
                        tf.math.reduce_min(qnet_mu),
                        tf.math.reduce_max(qnet_mu),
                        tf.math.reduce_min(qnet_var),
                        tf.math.reduce_max(qnet_var),
                        qnet_var]

            res_names = ["step",
                         "ELBO",
                         "recon loss",
                         "KL term",
                         "min qnet_mu",
                         "max qnet_mu",
                         "min qnet_var",
                         "max qnet_var",
                         "full qnet_var"]

            if 'SVGPVAE' in elbo_arg:
                res_vars += [inside_elbo,
                             inside_elbo_recon,
                             inside_elbo_kl,
                             ce_term,
                             tf.math.reduce_min(p_m),
                             tf.math.reduce_max(p_m),
                             tf.math.reduce_min(p_v),
                             tf.math.reduce_max(p_v),
                             latent_samples,
                             C_ma,
                             lagrange_mult]

                res_names += ["inside elbo",
                              "inside elbo recon",
                              "inside elbo KL",
                              "ce_term",
                              "min p_m",
                              "max p_m",
                              "min p_v",
                              "max p_v",
                              "latent_samples",
                              "C_ma",
                              "lagrange_mult"]

                res_vars_VAE = [global_step,
                                elbo_VAE,
                                recon_loss_VAE,
                                KL_term_VAE,
                                tf.math.reduce_min(qnet_mu_VAE),
                                tf.math.reduce_max(qnet_mu_VAE),
                                tf.math.reduce_min(qnet_var_VAE),
                                tf.math.reduce_max(qnet_var_VAE),
                                latent_samples_VAE]

                res_names_VAE = ["step",
                                 "ELBO",
                                 "recon loss",
                                 "KL term",
                                 "min qnet_mu",
                                 "max qnet_mu",
                                 "min qnet_var",
                                 "max qnet_var",
                                 "latent_samples"]

                res_vars_GP = [GP_l,
                               GP_amp,
                               GP_ov,
                               GP_ip]

                res_names_GP = ['length scale', 'amplitude', 'object vectors', 'inducing points']

                res_saver_VAE = pandas_res_saver(res_file_VAE, res_names_VAE)
                res_saver_GP = pandas_res_saver(res_file_GP, res_names_GP)

            res_saver = pandas_res_saver(res_file, res_names)

        # ====================== 5) print and init trainable params ======================
        print_trainable_vars(train_vars)

        init_op = tf.compat.v1.global_variables_initializer()

        # ====================== 6) saver and GPU ======================

        if save_model_weights:
            saver = tf.train.Saver(max_to_keep=3)

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=ram)

        # ====================== 7) tf.session ======================
        opt_regime = [r + '-' + str(nr_epochs) for r in opt_regime]
        nr_epochs, training_regime = parse_opt_regime(opt_regime)

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run(init_op)

            # training loop
            first_step = True  # switch for initialization of GECO algorithm
            C_ma_ = 0.0
            lagrange_mult_ = 1.0

            start_time = time.time()
            cgen_test_set_MSE = []

            print(f'GECO: {GECO}')
            best_eval_loss = np.inf
            best_loss_counter = 0
            stop_training = False

            for epoch in range(nr_epochs):
                # 7.1) train for one epoch
                sess.run(training_init_op)
                elbos, losses = [], []
                start_time_epoch = time.time()
                batch = 0
                if bias_analysis:
                    mean_vectors_arr = []
                while True:
                    try:
                        if GECO and "SVGPVAE" in elbo_arg and training_regime[epoch] != 'VAE':
                            if first_step:
                                alpha = 0.0
                            else:
                                alpha = alpha_arg
                            _, g_s_, elbo_, C_ma_, lagrange_mult_, recon_loss_, mean_vectors_ = sess.run([optim_step, global_step,
                                                                              elbo, C_ma, lagrange_mult,
                                                                              recon_loss, mean_vectors],
                                                                              {beta: beta_arg, lr: lr_arg,
                                                                               alpha_placeholder: alpha,
                                                                               C_ma_placeholder: C_ma_,
                                                                               lagrange_mult_placeholder: lagrange_mult_})
                            if bias_analysis:
                                mean_vectors_arr.append(mean_vectors_)
                        elif elbo_arg == "VAE" or elbo_arg == "CVAE":
                            _, g_s_, elbo_, recon_loss_ = sess.run(
                                [optim_step, global_step, elbo, recon_loss],
                                {beta: beta_arg, lr: lr_arg})
                        else:
                            _, g_s_, elbo_, recon_loss_ = sess.run([optim_step, global_step, elbo, recon_loss],
                                                      {beta: beta_arg, lr: lr_arg,
                                                       alpha_placeholder: alpha_arg,
                                                       C_ma_placeholder: C_ma_,
                                                       lagrange_mult_placeholder: lagrange_mult_})
                        elbos.append(elbo_)
                        losses.append(recon_loss_)
                        first_step = False  # switch for initizalition of GECO algorithm
                        if verbose:
                            print(f'epoch: {epoch}, batch: {batch}, elbo: {elbo_/ batch_size}, recon_loss: {recon_loss_/batch_size}')
                        batch += 1
                    except tf.errors.OutOfRangeError:
                        if bias_analysis:
                            mean_vector_full_data_ = sess.run(mean_vector_full_data,
                                                              {train_data_Y_placeholder: train_data_dict['data_Y'],
                                                               train_aux_X_placeholder: train_data_dict['aux_X']})

                            bias = compute_bias_variance_mean_estimators(mean_vectors_arr, mean_vector_full_data_)
                            print("Bias for epoch {}: {}".format(epoch, bias))
                        if (epoch + 1) % 10 == 0:
                            regime = training_regime[epoch] if "SVGPVAE" in elbo_arg else "VAE"
                            print('Epoch {}, opt regime {}, mean ELBO per batch: {}'.format(epoch, regime,
                                                                                            np.mean(elbos)))
                            MSE = np.sum(losses) / N_train
                            print('MSE loss on train set for epoch {} : {}'.format(epoch, MSE))

                            end_time_epoch = time.time()
                            print("Time elapsed for epoch {}, opt regime {}: {}".format(epoch,
                                                                                        regime,
                                                                                        end_time_epoch - start_time_epoch))
                        break

                # 7.2) calculate loss on eval set
                if "SVGPVAE" in elbo_arg:
                    losses = []
                    sess.run(eval_init_op, {eval_batch_size_placeholder: batch_size})
                    while True:
                        try:
                            recon_loss_ = sess.run(recon_loss, {beta: beta_arg, lr: lr_arg,
                                                                alpha_placeholder: alpha_arg,
                                                                C_ma_placeholder: C_ma_,
                                                                lagrange_mult_placeholder: lagrange_mult_})
                            losses.append(recon_loss_)
                        except tf.errors.OutOfRangeError:
                            MSE_eval = np.sum(losses) / N_eval
                            print('MSE loss on eval set for epoch {} : {}'.format(epoch, MSE_eval))
                            # early stopping
                            if MSE_eval < best_eval_loss:
                                best_eval_loss = MSE_eval
                                best_loss_counter = 0
                            else:
                                best_loss_counter += 1
                                if best_loss_counter == patience:
                                    stop_training = True
                            break
                if stop_training:
                    break

                # 7.3) save metrics to Pandas df for model diagnostics
                if save and (epoch + 1) % 10 == 0:
                    if test_set_metrics:
                        # sess.run(test_init_op, {test_batch_size_placeholder: N_test})  # see [update, 7.7.] above
                        sess.run(test_init_op, {test_batch_size_placeholder: batch_size})
                    else:
                        # sess.run(eval_init_op, {eval_batch_size_placeholder: N_eval})  # see [update, 7.7.] above
                        sess.run(eval_init_op, {eval_batch_size_placeholder: batch_size})

                    if "SVGPVAE" in elbo_arg:
                        # save elbo metrics depending on the type of forward pass (plain VAE vs SVGPVAE)
                        if training_regime[epoch] == 'VAE':
                            new_res = sess.run(res_vars_VAE, {beta: beta_arg})
                            res_saver_VAE(new_res, 1)
                        else:
                            new_res = sess.run(res_vars, {beta: beta_arg,
                                                          alpha_placeholder: alpha_arg,
                                                          C_ma_placeholder: C_ma_,
                                                          lagrange_mult_placeholder: lagrange_mult_})
                            res_saver(new_res, 1)

                        # save GP params
                        new_res_GP = sess.run(res_vars_GP, {beta: beta_arg,
                                                            alpha_placeholder: alpha_arg,
                                                            C_ma_placeholder: C_ma_,
                                                            lagrange_mult_placeholder: lagrange_mult_})
                        res_saver_GP(new_res_GP, 1)
                    else:
                        new_res = sess.run(res_vars, {beta: beta_arg})
                        res_saver(new_res, 1)

            # 7.4) calculate loss on test set and visualize reconstructed data

            losses, recon_data_Y_arr = [], []
            sess.run(test_init_op, {test_batch_size_placeholder: batch_size})
            # test set: reconstruction
            while True:
                try:
                    if "SVGPVAE" in elbo_arg:
                        recon_loss_, recon_data_Y_ = sess.run([recon_loss, recon_data_Y],
                                                                {beta: beta_arg,
                                                                alpha_placeholder: alpha_arg,
                                                                C_ma_placeholder: C_ma_,
                                                                lagrange_mult_placeholder: lagrange_mult_})
                    else:
                        recon_loss_, recon_data_Y_ = sess.run([recon_loss, recon_data_Y],
                                                                {beta: beta_arg})
                    losses.append(recon_loss_)
                    recon_data_Y_arr.append(recon_data_Y_)
                except tf.errors.OutOfRangeError:
                    MSE = np.sum(losses) / N_test
                    print('MSE loss on test set for epoch {} : {}'.format(epoch, MSE))
                    recon_data_Y_arr = np.concatenate(tuple(recon_data_Y_arr))
                    plot_tabular(test_data_dict['data_Y'],
                                recon_data_Y_arr,
                                title="Epoch: {}. Recon MSE test set:{}".format(epoch + 1, round(MSE, 4)))
                    if show_pics:
                        plt.show()
                        plt.pause(0.01)
                    if save:
                        plt.savefig(pic_folder + str(g_s_) + ".png")
                    break
            # test set: conditional generation SVGPVAE
            if "SVGPVAE" in elbo_arg:

                # encode training data (in batches)
                sess.run(training_init_op)
                means, vars = [], []
                while True:
                    try:
                        qnet_mu_train_, qnet_var_train_ = sess.run([qnet_mu_train, qnet_var_train])
                        means.append(qnet_mu_train_)
                        vars.append(qnet_var_train_)
                    except tf.errors.OutOfRangeError:
                        break
                means = np.concatenate(means, axis=0)
                vars = np.concatenate(vars, axis=0)

                # predict test data (in batches)
                sess.run(test_init_op, {test_batch_size_placeholder: batch_size})
                recon_loss_cgen, recon_data_Y_cgen = [], []
                while True:
                    try:
                        loss_, recon_Y_batch_ = sess.run([recon_loss_test, recon_data_Y_test],
                                                {train_aux_X_placeholder: train_data_dict['aux_X'],
                                                    train_encodings_means_placeholder: means,
                                                    train_encodings_vars_placeholder: vars})
                        recon_loss_cgen.append(loss_)
                        recon_data_Y_cgen.append(recon_Y_batch_)
                    except tf.errors.OutOfRangeError:
                        break
                recon_loss_cgen = np.sum(recon_loss_cgen) / N_test
                recon_data_Y_cgen = np.concatenate(recon_data_Y_cgen, axis=0)

            # test set: conditional generation CVAE
            if elbo_arg == "CVAE":
                recon_loss_cgen, recon_data_Y_cgen = sess.run([recon_loss_test, recon_data_Y_test],
                                    {train_aux_X_placeholder: train_data_dict['aux_X'],
                                        train_data_Y_placeholder: train_data_dict['data_Y'],
                                        test_aux_X_placeholder: test_data_dict['aux_X'],
                                        test_data_Y_placeholder: test_data_dict['data_Y']})

            # test set: plot generations
            if elbo_arg != "VAE":
                cgen_test_set_MSE.append((epoch, recon_loss_cgen))
                print("Conditional generation MSE loss on test set for epoch {}: {}".format(epoch,
                                                                                            recon_loss_cgen))
                plot_tabular(test_data_dict['data_Y'],
                            recon_data_Y_cgen,
                            title="Epoch: {}. CGEN MSE test set:{}".format(epoch + 1, round(recon_loss_cgen, 4)))
                if show_pics:
                    plt.show()
                    plt.pause(0.01)
                if save:
                    plt.savefig(pic_folder + str(g_s_) + "_cgen.png")
                    with open(pic_folder + "test_metrics.txt", "a") as f:
                        f.write("{},{},{}\n".format(epoch + 1, round(MSE, 4), round(recon_loss_cgen, 4)))

            # save model weights
            if save and save_model_weights:
                saver.save(sess, chkpnt_dir + "model", global_step=g_s_)

            # log running time
            end_time = time.time()
            print("Running time for {} epochs: {}".format(epoch, round(end_time - start_time, 2)))

            # if "SVGPVAE" in elbo_arg:
            #     # report best test set cgen MSE achieved throughout training
            #     best_cgen_MSE = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0]
            #     print("Best cgen MSE on test set throughout training at epoch {}: {}".format(best_cgen_MSE[0],
            #                                                                                  best_cgen_MSE[1]))

            # save images from conditional generation
            if save and elbo_arg != "VAE":
                with open(chkpnt_dir + '/cgen_data_ys.p', 'wb') as test_pickle:
                    pickle.dump(recon_data_Y_cgen, test_pickle)

            # save latents
            if save_latents:
                if "SVGPVAE" in elbo_arg:
                    latent_samples_full_ = sess.run(latent_samples_full,
                                                    {train_data_Y_placeholder: train_data_dict['data_Y'],
                                                     train_aux_X_placeholder: train_data_dict['aux_X']})
                else:
                    latent_samples_full_ = sess.run(latent_samples_full,
                                                    {train_data_Y_placeholder: train_data_dict['data_Y']})
                with open(chkpnt_dir + '/latents_train_full.p', 'wb') as pickle_latents:
                    pickle.dump(latent_samples_full_, pickle_latents)

    return recon_data_Y_cgen


if __name__=="__main__":

    default_base_dir = os.getcwd()

    parser_tabular = argparse.ArgumentParser(description='Tabular data experiment.')
    parser_tabular.add_argument('--expid', type=str, default="debug_TABULAR", help='give this experiment a name')
    parser_tabular.add_argument('--base_dir', type=str, default=default_base_dir,
                              help='folder within a new dir is made for each run')
    parser_tabular.add_argument('--elbo', type=str, choices=['VAE', 'CVAE', 'SVGPVAE_Hensman', 'SVGPVAE_Titsias',
                                                           'GPVAE_Casale', 'GPVAE_Casale_batch', 'SVIGP_Hensman'],
                              default='VAE')
    parser_tabular.add_argument('--mnist_data_path', type=str, default='MNIST data/',
                              help='Path where rotated MNIST data is stored.')
    parser_tabular.add_argument('--batch_size', type=int, default=256)
    parser_tabular.add_argument('--nr_epochs', type=int, default=1000)
    parser_tabular.add_argument('--beta', type=float, default=0.001)
    parser_tabular.add_argument('--nr_inducing_points', type=float, default=2, help="Number of object vectors per angle.")
    parser_tabular.add_argument('--save', action="store_true", help='Save model metrics in Pandas df as well as images.')
    parser_tabular.add_argument('--GP_joint', action="store_true", help='GP hyperparams joint optimization.')
    parser_tabular.add_argument('--ip_joint', action="store_true", help='Inducing points joint optimization.')
    parser_tabular.add_argument('--ov_joint', action="store_true", help='Object vectors joint optimization.')
    parser_tabular.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    parser_tabular.add_argument('--save_model_weights', action="store_true",
                              help='Save model weights. For debug purposes.')
    parser_tabular.add_argument('--dataset', type=str, choices=['3', '36', '13679'], default='3')
    parser_tabular.add_argument('--show_pics', action="store_true", help='Show images during training.')
    parser_tabular.add_argument('--opt_regime', type=str, default=['joint-1000'], nargs="+")
    parser_tabular.add_argument('--L', type=int, default=16, help="Nr. of latent channels")
    parser_tabular.add_argument('--clip_qs', action="store_true", help='Clip variance of inference network.')
    parser_tabular.add_argument('--ram', type=float, default=1.0, help='fraction of GPU ram to use')
    parser_tabular.add_argument('--test_set_metrics', action='store_true',
                              help='Calculate metrics on test data. If false, metrics are calculated on eval data.')
    parser_tabular.add_argument('--GECO', action='store_true', help='Use GECO algorithm for training.')
    parser_tabular.add_argument('--alpha', type=float, default=0.99, help='Moving average parameter for GECO.')
    parser_tabular.add_argument('--kappa_squared', type=float, default=0.020, help='Constraint parameter for GECO.')
    parser_tabular.add_argument('--object_kernel_normalize', action='store_true',
                              help='Normalize object (linear) kernel.')
    parser_tabular.add_argument('--save_latents', action='store_true', help='Save Z . For t-SNE plots :)')
    parser_tabular.add_argument('--jitter', type=float, default=0.000001, help='Jitter for numerical stability.')
    parser_tabular.add_argument('--PCA', action="store_true",
                              help='Use PCA embeddings for initialization of object vectors.')
    parser_tabular.add_argument('--bias_analysis', action='store_true',
                              help="Compute bias of estimator for mean vector in hat{q}^Titsias for every epoch.")
    parser_tabular.add_argument('--M', type=int, default=8, help="Dimension of GPLVM vectors.")
    parser_tabular.add_argument('--disable_gpu', action="store_true", help='Disable GPU.')

    args_tabular = parser_tabular.parse_args()

    if args_tabular.elbo == "GPVAE_Casale":
        # run_experiment_rotated_mnist_Casale(args_mnist)
        pass

    elif args_tabular.elbo == "SVIGP_Hensman":
        dict_ = vars(args_tabular)
        # run_experiment_rotated_mnist_SVIGP_Hensman(args_mnist, dict_)
        pass

    else:  # VAE, CVAE, SVGPVAE_Hensman, SVGPVAE_Titsias
        dict_ = vars(args_tabular)  # [update, 23.6.] to get around weirdest bug ever
        train_data_dict, eval_data_dict, test_data_dict = load_mnist_data(args_tabular, ending = args_tabular.dataset + '.p')
        run_experiment_SVGPVAE(train_data_dict, eval_data_dict, test_data_dict,
            args_tabular.L, args_tabular.batch_size, args_tabular.nr_epochs, args_tabular.elbo,
            args_tabular.M, args_tabular.nr_inducing_points, args_tabular.PCA,
            args_tabular.ip_joint, args_tabular.GP_joint, args_tabular.ov_joint,
            args_tabular.disable_gpu,
            args_tabular.beta, args_tabular.lr, args_tabular.base_dir, args_tabular.expid,
            args_tabular.jitter, args_tabular.object_kernel_normalize, args_tabular.save,
            args_tabular.save_latents, args_tabular.save_model_weights, args_tabular.show_pics,
            args_tabular.kappa_squared, args_tabular.clip_qs, args_tabular.GECO,
            args_tabular.bias_analysis, args_tabular.opt_regime, args_tabular.test_set_metrics,
            args_tabular.mnist_data_path, args_tabular.ram, args_tabular.dataset, True)

