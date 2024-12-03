from utils.data import Categorical2d
from utils.model import Causal2dModel
from utils.plotting import (save_plot_raw_causal_params,
                            save_plot_soft_causal_params,
                            save_loss_plot,
                            save_hist_params_across_runs,
                            save_plot_params_over_epochs,
                            create_visualization_video,
                            save_plot_distances,
                            save_plot_distances_for_1st_intervention,
                            save_plot_param_grads)
from utils.misc import evaluate, write_parameters, save_data
import utils.config as config
from scipy.special import softmax
from scipy.linalg import norm

from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
from datetime import datetime
from types import SimpleNamespace

# Flag whether to use WandB or not
WANDB = False

if WANDB:
    import wandb
    wandb.login()



def main():

    if WANDB:
        wandb.init(project="XXXX")


    starttime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # String used for creating a folder with plots.

    # Initialize random number generator
    rng = npr.default_rng(seed=None)

    write_parameters(starttime, config.params)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    #loss_fn = torch.nn.NLLLoss()


    # Initialize config parameters
    if WANDB:
        cfg = wandb.config
    else:
        cfg = SimpleNamespace(**config.params)

    all_soft_causal_params = np.zeros((cfg.num_runs, cfg.num_epochs, 2))
    all_raw_causal_params = np.zeros((cfg.num_runs, cfg.num_epochs, 2))
    all_eval_losses = np.zeros((cfg.num_runs, cfg.num_epochs, 2))

    # Training loop
    for run_id in tqdm(range(cfg.num_runs)):

        # Initialize new dataset for training
        data_generator = Categorical2d(factorization=cfg.factorization, smoothness=cfg.smoothness, n_categories=cfg.n_categories, beta=cfg.beta, format=cfg.format, rng=rng)

        p_B = data_generator.p_A @ data_generator.p_BgivenA
        p_AgivenB = 1 / p_B[:, None] * data_generator.p_A * data_generator.p_BgivenA.T

        model = Causal2dModel(model=cfg.model, n_categories=cfg.n_categories, emb_size=cfg.emb_size,
                                    temperature=cfg.temperature, use_bias=cfg.use_bias, hard=cfg.hard,
                                    positional_encoding=cfg.positional_encoding,
                                    positional_encoding_type=cfg.pos_enc_type,
                                    positional_encoding_scale=cfg.pos_enc_scale,
                                    normalization=cfg.normalization, use_input=cfg.use_input)
        

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        ############## Initialize logging arrays ##############
        probability_matrices = np.zeros((cfg.num_epochs, cfg.num_batches, 3, cfg.n_categories, cfg.n_categories))
        probability_vectors = np.zeros((cfg.num_epochs, cfg.num_batches, 3, cfg.n_categories))
        matrix_distances = np.zeros((cfg.num_epochs, cfg.num_batches, 2))
        raw_causal_params = np.zeros((cfg.num_epochs, cfg.num_batches, 2))
        soft_causal_params = np.zeros((cfg.num_epochs, cfg.num_batches, 2))
        train_loss = np.zeros((cfg.num_epochs, cfg.num_batches))
        eval_loss = np.zeros((cfg.num_epochs, cfg.num_batches, 2)) # Stores separate loss for A and B reconstruction
        causal_param_gradients = np.zeros((cfg.num_epochs, cfg.num_batches, 2))
        ########################################################

        # Generate a list of batches from the data_generator
        batched_data = [torch.Tensor(data_generator[cfg.batch_size]) for _ in range(cfg.num_batches)]

        # Train for a specified number of epochs
        for i in range(cfg.num_epochs):

            # Train on all batches consecutively
            for j in range(cfg.num_batches):

                ############## Log data required for video ##############
                p_B = data_generator.p_A @ data_generator.p_BgivenA # This is different to the also existing variable data_generator.p_B, which is however not used for generating data since it's not part of the factorization.
                p_AgivenB = 1/p_B[:,None] * data_generator.p_A * data_generator.p_BgivenA.T
                p_learned = softmax(model.W_v.weight.detach().cpu().numpy().T, axis=1)  # W_v.T (converted from logits to probabilities), if multiplied with A should give p(B)

                probability_matrices[i, j, 0] = p_learned
                probability_matrices[i, j, 1] = data_generator.p_BgivenA                                         # p_BgivenA
                probability_matrices[i, j, 2] = p_AgivenB                                                        # p_AgivenB
                matrix_distances[i, j, 0] = norm(p_learned - data_generator.p_BgivenA)
                matrix_distances[i, j, 1] = norm(p_learned - p_AgivenB)
                probability_vectors[i, j, 0] = data_generator.p_A
                probability_vectors[i, j, 1] = p_B

                ########## Train model ##########
                # Get a batch of datapoints
                inputs = batched_data[j]

                optimizer.zero_grad()

                # Pass inputs through model
                logits, train_causal_params, train_soft_causal_params = model(inputs)

                if cfg.loss == "marginal":
                    # Transpose logits and inputs in order to meet input dimensionality of CELoss
                    logits = logits.transpose(1, 2)
                    targets = inputs.transpose(1, 2)
                
                elif cfg.loss == "joint":
                    kronecker_product = lambda x1, x2: torch.einsum("nk,nl->nkl",x1,x2).reshape(cfg.batch_size, -1) #can't just use torch.kron here since it doesn't allow specifying dimensions
                    logits = kronecker_product(logits[:,0,:], logits[:,1,:])
                    targets = kronecker_product(inputs[:,0,:], inputs[:,1,:])

                # Calculate loss, backwards pass and parameter updates
                loss = torch.mean(loss_fn(logits, targets))
                loss.backward()
                optimizer.step()

                ############## Log general data ##############
                train_loss[i, j] = np.mean(loss.detach().cpu().numpy()) # Average train losses over all batches for this epoch and store for plotting
                eval_loss[i, j], eval_causal_params, eval_soft_causal_params = evaluate(model, data_generator, loss_fn, cfg.loss, cfg.eval_size)

                causal_param_gradients[i, j] = (model.causal_params.grad).detach().cpu().numpy()

                raw_causal_params[i, j] = np.mean(train_causal_params, axis=0)
                soft_causal_params[i, j] = np.mean(softmax(train_causal_params, axis=1), axis=0)

            # Potentially intervene and sample new data
            if (i+1) % cfg.distribution_change_freq == 0:
                data_generator.intervention()
                batched_data = [torch.Tensor(data_generator[cfg.batch_size]) for _ in range(cfg.num_batches)]

        ############## Log run data for across-run comparison ##############
        all_soft_causal_params[run_id] = soft_causal_params[:, -1, :] # Only save the soft causal parameters from the last batch of each epoch
        all_raw_causal_params[run_id] = raw_causal_params[:, -1, :] # Only save the causal_parameters from the last batch of each epoch
        all_eval_losses[run_id] = eval_loss[:, -1, :]
        
        ############## Save plots (and video) for current run ##############
        save_plot_raw_causal_params(raw_causal_params[:, -1, :], starttime, run_id)
        save_plot_soft_causal_params(soft_causal_params[:, -1, :], starttime, run_id)
        save_plot_param_grads(np.mean(causal_param_gradients, axis=1), starttime, run_id)
        save_loss_plot(np.mean(train_loss, axis=1), np.mean(eval_loss, axis=1), starttime, run_id)
        save_plot_distances(matrix_distances, causal_param_gradients, starttime, run_id)
        save_plot_distances_for_1st_intervention(matrix_distances, causal_param_gradients, cfg.distribution_change_freq, starttime, run_id)

        if cfg.record_video:
            # Save video of values during this run
            create_visualization_video(starttime, probability_matrices, matrix_distances, probability_vectors, raw_causal_params, soft_causal_params, cfg.num_epochs, cfg.num_batches, run_id)



    ##### Saving Logs to WANDB #####
    if WANDB:
        # Logging over epochs:
        for epoch, value in enumerate(np.mean(all_soft_causal_params[:, :, 0], axis=0)):
            wandb.log({'epoch': epoch, 'Param_A': value})

        # Logging Losses
        for epoch, value in enumerate(np.mean(all_eval_losses[:, :, 0], axis=0)):
            wandb.log({'epoch': epoch, 'Loss_A': value})
        for epoch, value in enumerate(np.mean(all_eval_losses[:, :, 1], axis=0)):
            wandb.log({'epoch': epoch, 'Loss_B': value})


        # Logging Mean Metrics and run ID
        wandb.log({"mean_c1": all_soft_causal_params[:, -1, 0].mean()})
        wandb.log({"mean_eval_loss": all_eval_losses[:, -1, :].mean()})
        wandb.log({"median_c1": np.median(all_soft_causal_params[:, -1, 0])})
        wandb.log({'Run_ID': starttime})

    ############## Save plots across-run ##############
    # Save overall results and raw softmaxed causal parameter values
    save_hist_params_across_runs(all_soft_causal_params[:, -1, :], starttime) # Only take the causal parameters of the last episode for each run
    save_plot_params_over_epochs(all_soft_causal_params, starttime)
    save_data(starttime, all_soft_causal_params, all_raw_causal_params)


if __name__ == '__main__':
    if WANDB:
        cfg = config.sweep_config
        sweep_id = wandb.sweep(sweep=cfg, project="XXXX")
        wandb.agent(sweep_id, function=main, count=200)

    else:
        main()