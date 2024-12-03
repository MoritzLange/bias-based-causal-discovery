# File for functions that generate visualizations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .misc import print_progress
import numpy as np
from pathlib import Path

def save_plot_raw_causal_params(causal_params: np.ndarray, starttime: str, run_id: str):
    """
    Generates a scatter plot of causal parameters for all epochs of a run, before they're softmaxed, and saves this to disk.

    Args:
        causal_parameters: Array of shape (num_epochs, 2) containing the causal parameters returned by the model
        starttime: String with time of the current run
        run_id: The id of the current train run
    """
    plt.figure()
    plt.scatter(range(len(causal_params)), causal_params[:,0], alpha=0.6, label="X")
    plt.scatter(range(len(causal_params)), causal_params[:,1], alpha=0.6, label="Y")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Raw causal parameters")
    plt.title(f"Raw causal parameters at\nthe end of each training epoch")
    Path(f"plots/{starttime}/raw_causal_params").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    plt.savefig(f"plots/{starttime}/raw_causal_params/run_{run_id}.png")
    plt.close()

def save_plot_soft_causal_params(soft_causal_params: np.ndarray, starttime: str, run_id: str):
    """
    Generates a scatter plot of causal parameters for all epochs of a run, after they're softmaxed, and saves this to disk.

    Args:
        soft_causal_parameters: Array of shape (num_epochs, 2) containing the soft causal parameters calculated from raw scores returned by the model
        starttime: String with time of the current run
        run_id: The id of the current train run
    """
    # Apply the softmax operation to the soft causal parameters
    plt.figure()
    plt.scatter(range(len(soft_causal_params)), soft_causal_params[:,0], alpha=0.6, label="X")
    plt.scatter(range(len(soft_causal_params)), soft_causal_params[:,1], alpha=0.6, label="Y")
    plt.legend()
    plt.ylim((-0.05,1.05))
    plt.xlabel("Epochs")
    plt.ylabel("Soft causal parameters")
    plt.title(f"Softmaxed causal parameters at\nthe end of each training epoch")
    Path(f"plots/{starttime}/soft_causal_params").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    plt.savefig(f"plots/{starttime}/soft_causal_params/run_{run_id}.png")
    plt.close()

def save_plot_param_grads(param_grads: np.ndarray, starttime: str, run_id: str):
    """
    Generates a line plot of gradients of causal parameters for all epochs of a run, and saves this to disk.

    Args:
        param_grads: Array of shape (num_epochs, 2) containing gradients of causal parameters
        starttime: String with time of the current run
        run_id: The id of the current train run
    """
    plt.figure()
    plt.plot(range(len(param_grads)), param_grads[:,0], alpha=0.6, label="a_1")
    plt.plot(range(len(param_grads)), param_grads[:,1], alpha=0.6, label="a_2")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Parameter gradients")
    plt.title(f"Causal parameter gradients \n averaged for each training epoch")
    Path(f"plots/{starttime}/param_grads").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    plt.savefig(f"plots/{starttime}/param_grads/run_{run_id}.png")
    plt.close()

def save_loss_plot(train_loss:np.ndarray, eval_loss: np.ndarray, starttime: str, run_id: str):
    """
    Generates a plot of train and test loss and saves this to disk.
    
    Args:
        train_loss: Array with a train loss value for each epoch
        test_loss: Array with a eval loss value for each epoch
        starttime: String with time of the current run
        run_id: The id of the current train run
    """
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(eval_loss[:, 0], label="Eval Loss A")
    plt.plot(eval_loss[:, 1], label="Eval Loss B", alpha=0.7)

    plt.legend()
    plt.title("Losses during training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    Path(f"plots/{starttime}/losses").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    plt.savefig(f"plots/{starttime}/losses/run_{run_id}.png")
    plt.close()

def save_plot_distances(matrix_distances: np.ndarray, param_gradients, starttime, run_id):
    """
    Plots the distances of the learned matrix to p_BgivenA and p_AgivenB over time.

    Args:
        matrix_distances: Array of shape (num_epochs, num_batches, 2)
        param_gradients: Array of shape (num_epochs, num_batches, 2)
        starttime: String with time of the current run
        run_id: The id of the current run
    """
    num_epochs, num_batches = matrix_distances.shape[:2]
    x_values = np.arange(num_epochs*num_batches)/num_batches
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,5))
    ax1.plot(x_values, matrix_distances.reshape(-1, 2)[:,0], label="p_BgivenA")
    ax1.plot(x_values, matrix_distances.reshape(-1, 2)[:,1], label="p_AgivenB")
    ax1.legend()
    ax1.set_title("Distances of learned matrix W_v to probability matrices")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Distance")
    ax2.plot(x_values, param_gradients.reshape(-1, 2)[:,0], label="a_1")
    ax2.plot(x_values, param_gradients.reshape(-1, 2)[:,1], label="a_2")
    ax2.legend()
    ax2.set_title("Causal parameter gradients at each batch")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Gradient")
    Path(f"plots/{starttime}/dists").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    fig.savefig(f"plots/{starttime}/dists/run_{run_id}.png")
    plt.close()

def save_plot_distances_for_1st_intervention(matrix_distances: np.ndarray, param_gradients, first_intervention, starttime, run_id):
    """
    Plots the distances of the learned matrix to p_BgivenA and p_AgivenB over time.

    Args:
        matrix_distances: Array of shape (num_epochs, num_batches, 2)
        param_gradients: Array of shape (num_epochs, num_batches, 2)
        first_intervention: After how many epochs the first intervention happens
        starttime: String with time of the current run
        run_id: The id of the current run
    """
    num_epochs, num_batches = matrix_distances.shape[:2]
    max_epoch = min(num_epochs, first_intervention*2)
    x_values = np.arange(max_epoch*num_batches)/num_batches
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,5))
    ax1.plot(x_values, matrix_distances[:max_epoch].reshape(-1, 2)[:,0], label="p_BgivenA")
    ax1.plot(x_values, matrix_distances[:max_epoch].reshape(-1, 2)[:,1], label="p_AgivenB")
    ax1.legend()
    ax1.set_title("Distances of learned matrix W_v\nto probability matrices")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Distance")
    ax1.axvline(x=first_intervention, linewidth=1, c="red")
    ax2.plot(x_values, param_gradients[:max_epoch].reshape(-1, 2)[:,0], label="a_1")
    ax2.plot(x_values, param_gradients[:max_epoch].reshape(-1, 2)[:,1], label="a_2")
    ax2.legend()
    ax2.set_title("Causal parameter gradients\nat each batch")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Gradient")
    ax2.axvline(x=first_intervention, linewidth=1, c="red")
    Path(f"plots/{starttime}/dists_beginning").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    fig.savefig(f"plots/{starttime}/dists_beginning/run_{run_id}.png")
    plt.close()


def save_hist_params_across_runs(soft_causal_params: np.ndarray, starttime):
    """
    Plots causal parameters over all runs of an experiment (i.e. an collection of runs)

    Args:
        soft_causal_params: Array of shape (num_runs, 2) containing the soft causal parameters from the last evaluation of each run
        starttime: String with time of the current run
    """
    plt.figure()
    bins = np.linspace(0,1,11) # bins [0, 0.1, ..., 0.9, 1]
    plt.hist(soft_causal_params[:,0], bins=bins, alpha=0.5, label="X")
    plt.hist(soft_causal_params[:,1], bins=bins, alpha=0.5, label="Y")
    plt.title("Soft causal parameters across runs\n(1 means independent variable, 0 means dependent variable)")
    plt.legend()
    Path(f"plots/{starttime}").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    plt.savefig(f"plots/{starttime}/soft_param_histogram.png")
    plt.close()

def save_plot_params_over_epochs(soft_causal_params: np.ndarray, starttime):
    """
    Plots median and 0/0.25/0.75/1 quantile of softmaxed causal parameters of the first value per epoch across runs.

    Args:
        soft_causal_params: Array of shape (num_runs, num_epochs, 2) containing the soft causal parameters of each epoch for each run
        starttime: String with time of the current run
    """   
    plt.figure()
    quantiles = np.quantile(soft_causal_params[:,:,0], q=[0, 0.25, 0.5, 0.75, 1], axis=0)
    plt.fill_between(range(soft_causal_params.shape[1]), quantiles[1,:], quantiles[3,:], alpha=0.2, color='#1f77b4', label="0.25 - 0.75 Quantiles")
    plt.fill_between(range(soft_causal_params.shape[1]), quantiles[0,:], quantiles[1,:], alpha=0.1, color='#1f77b4', label="0 - 1 Quantiles")
    plt.fill_between(range(soft_causal_params.shape[1]), quantiles[3,:], quantiles[4,:], alpha=0.1, color='#1f77b4')
    plt.plot(quantiles[2,:], c='#1f77b4', label="Median")
    plt.plot(np.mean(soft_causal_params[:,:,0], axis=0), c='#ff7f0e', label="Mean")

    plt.axhline(y = 0, color = 'grey', linestyle = ':')
    plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
    plt.axhline(y = 1, color = 'grey', linestyle = ':')
    plt.ylim((-0.05,1.05))
    plt.xlabel("Epochs")
    plt.ylabel("Causal parameters")
    plt.legend()
    Path(f"plots/{starttime}").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    plt.savefig(f"plots/{starttime}/median_params_per_epoch.png")
    plt.close()


def create_visualization_video(starttime, matrices_list, matrix_distances, vectors_list, causal_params, soft_causal_params, num_epochs, num_batches, run_id):
    """_summary_

    Args:
        starttime (_type_): _description_
        matrices_list (_type_): _description_
        matrix_distances (_type_): _description_
        vectors_list (_type_): _description_
        causal_params (_type_): _description_
        soft_causal_params (_type_): _description_
    """
    print(f"Caution: This video generation function was written for \n1. Categorical2d data with 10 categories,\n2. Causal2dModel and\n3. W_v without bias.\nIt might not work or show misleading information in other cases!")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Titles for matrices, vectors, and values
    matrix_titles = ["σ(W_v.T)", "p(B|A)", "σ(W_v.T)", "p(A|B)"]
    vector_titles = ["p(A)", "p(B)", "σ(i1)"]
    value_titles = ["Soft Att.", "Raw Att."]

    # Initialize text
    frame_text = fig.add_axes([0.5, 0.06, 0.1, 0.05])  # Adjust the position of the text
    text = frame_text.text(0.5, 0.5, "", ha="center", va="center")
    frame_text.axis("off")  # Turn off the axis frame for the frame_text

    # Initialize matrices
    matrix_width = 0.25  # Adjust the width of the matrix columns
    matrix_spacing = 0.02  # Add some white space between the matrices
    total_matrix_width = (matrix_width + matrix_spacing) * 2 - matrix_spacing
    matrix_start = (1 - total_matrix_width) / 2  # Center the matrices in the frame
    matrix_y_position = 0.75  # Adjust the vertical position of the matrices

    matrix_ax0 = fig.add_axes([matrix_start + 0 * (matrix_width + matrix_spacing), matrix_y_position, matrix_width, 0.2])
    matrix_ax0.set_title(matrix_titles[0])
    matrix_ax0.axis("off")
    mat0 = matrix_ax0.imshow(matrices_list[0, 0, 0], cmap="binary", vmin=0, vmax=1)

    matrix_ax1 = fig.add_axes([matrix_start + 1 * (matrix_width + matrix_spacing), matrix_y_position, matrix_width, 0.2])
    # matrix_ax1.set_title(matrix_titles[1])
    matrix_ax1.axis("off")
    mat1 = matrix_ax1.imshow(matrices_list[0, 0, 1], cmap="binary", vmin=0, vmax=1)
    mat1_title = matrix_ax1.text(1, -1, matrix_titles[1])

    matrix_ax2 = fig.add_axes([matrix_start + 0 * (matrix_width + matrix_spacing), matrix_y_position - 0.3, matrix_width, 0.2])
    matrix_ax2.set_title(matrix_titles[2])
    matrix_ax2.axis("off")
    mat2 = matrix_ax2.imshow(matrices_list[0, 0, 0], cmap="binary", vmin=0, vmax=1)

    matrix_ax3 = fig.add_axes([matrix_start + 1 * (matrix_width + matrix_spacing), matrix_y_position - 0.3, matrix_width, 0.2])
    # matrix_ax3.set_title(matrix_titles[3])
    matrix_ax3.axis("off")
    mat3 = matrix_ax3.imshow(matrices_list[0, 0, 2], cmap="binary", vmin=0, vmax=1)
    mat3_title = matrix_ax3.text(1, -1, matrix_titles[3])

    # Initialize vectors
    vector_width = 0.03  # Adjust the width of the vector columns
    vector_spacing = 0.04  # Add some white space between the vectors
    total_vector_width = (vector_width + vector_spacing) * 3 - vector_spacing
    vector_start = (1 - total_vector_width) / 3  # Put vectors on right third of frame
    vector_y_position = 0.15  # Adjust the vertical position of the vectors

    vector_ax0 = fig.add_axes([vector_start + 0 * (vector_width + vector_spacing), vector_y_position, vector_width, 0.2])
    vector_ax0.set_title(vector_titles[0])
    vector_ax0.axis("off")
    vec0 = vector_ax0.imshow(vectors_list[0, 0, 0].reshape(-1, 1), cmap="binary", aspect="auto", vmin=0, vmax=1)

    vector_ax1 = fig.add_axes([vector_start + 1 * (vector_width + vector_spacing), vector_y_position, vector_width, 0.2])
    vector_ax1.set_title(vector_titles[1])
    vector_ax1.axis("off")
    vec1 = vector_ax1.imshow(vectors_list[0, 0, 1].reshape(-1, 1), cmap="binary", aspect="auto", vmin=0, vmax=1)

    vector_ax2 = fig.add_axes([vector_start + 2 * (vector_width + vector_spacing), vector_y_position, vector_width, 0.2])
    vector_ax2.set_title(vector_titles[2])
    vector_ax2.axis("off")
    vec2 = vector_ax2.imshow(vectors_list[0, 0, 2].reshape(-1, 1), cmap="binary", aspect="auto", vmin=0, vmax=1)

    # vector_ax3 = fig.add_axes([vector_start + 3 * (vector_width + vector_spacing), vector_y_position, vector_width, 0.2])
    # vector_ax3.set_title(vector_titles[3])
    # vector_ax3.axis("off")
    # vec3 = vector_ax3.imshow(vectors_list[0, 0, 3].reshape(-1, 1), cmap="binary", aspect="auto", vmin=0, vmax=1)

    # Initialize values
    value_width = 0.03  # Adjust the width of the value columns
    value_spacing = 0.06  # Add some white space between the values
    total_value_width = (value_width + value_spacing) * 2 - value_spacing
    value_start = 2*(1 - total_value_width) / 3  # Center the values on the page
    value_y_position = 0.2  # Adjust the vertical position of the values
    value_labels = ["a_1", "a_2"]

    value_ax0 = fig.add_axes([value_start + 0 * (value_width + value_spacing), value_y_position, value_width, 0.05])
    value_ax0.set_title(value_titles[0])
    value_ax0.axis("off")
    value_ax0.text(-1.5, 0.1, value_labels[0], va="center")
    value_ax0.text(-1.5, 1, value_labels[1], va="center")
    val0 = value_ax0.imshow(soft_causal_params[0, 0].reshape(-1, 1), cmap="binary", aspect="auto", vmin=0, vmax=1)

    value_ax1 = fig.add_axes([value_start + 1 * (value_width + value_spacing), value_y_position, value_width, 0.05])
    value_ax1.set_title(value_titles[1])
    value_ax1.axis("off")
    value_ax1.text(-1.5, 0.1, value_labels[0], va="center")
    value_ax1.text(-1.5, 1, value_labels[1], va="center")
    val1 = value_ax1.imshow(causal_params[0, 0].reshape(-1, 1), cmap="binary", aspect="auto", vmin=np.min(causal_params), vmax=np.max(causal_params))

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Update function for animation
    def update(frame):
        current_epoch = frame // num_batches
        current_batch = frame % num_batches
        
        # Plot matrices
        mat0.set_data(matrices_list[current_epoch, current_batch, 0])
        mat1.set_data(matrices_list[current_epoch, current_batch, 1])
        mat2.set_data(matrices_list[current_epoch, current_batch, 0])
        mat3.set_data(matrices_list[current_epoch, current_batch, 2])

        # Write distance to learned matrix into titles of p_BgivenA and p_AgivenB
        mat1_title.set_text(matrix_titles[1] + f" | d = {matrix_distances[current_epoch, current_batch, 0]:.2f}")
        mat3_title.set_text(matrix_titles[3] + f" | d = {matrix_distances[current_epoch, current_batch, 1]:.2f}")

        # Plot vectors
        vec0.set_data(vectors_list[current_epoch, current_batch, 0].reshape(-1, 1))
        vec1.set_data(vectors_list[current_epoch, current_batch, 1].reshape(-1, 1))
        vec2.set_data(vectors_list[current_epoch, current_batch, 2].reshape(-1, 1))
        # vec3.set_data(vectors_list[current_epoch, current_batch, 3].reshape(-1, 1))
        
        # Plot values
        val0.set_data(soft_causal_params[current_epoch, current_batch].reshape(-1, 1))
        val1.set_data(causal_params[current_epoch, current_batch].reshape(-1, 1))
        
        # Update text
        text.set_text(f"Current epoch: {current_epoch:03d}/{num_epochs:03d} | Current batch: {current_batch:03d}/{num_batches:03d}")

        return [mat0, mat1, mat2, mat3, mat1_title, mat3_title, vec0, vec1, vec2, val0, val1, text]

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_epochs*num_batches, repeat=False, interval=1, cache_frame_data=False, blit=True)

    # Save animation as a video
    print("Saving video...")
    Path(f"plots/{starttime}/vids").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    ani.save(f"plots/{starttime}/vids/probabilities-run_{run_id}.mp4", writer="ffmpeg", dpi=100, fps=10, progress_callback= print_progress)
    plt.close(fig)
