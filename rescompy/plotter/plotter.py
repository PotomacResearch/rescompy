"""analysis.py

The Analysis submodule for rescompy.

This submodule contains convenience functions for analyzing and visualizing
training and prediction results from rescompy.
"""


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List
from pydantic import validate_arguments
import logging
import rescompy


#@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_actual(
    train_result:   Optional[rescompy.TrainResult]   = None,
    predict_result: Optional[rescompy.PredictResult] = None,
    max_plots:      int                              = 3,
    y_labels:       Optional[List[str]]              = None,
    tau:            float                            = 1.0,
    t_units:        str                              = r'$\tau$',
    save_loc:       Optional[str]                    = None,
    ):
    """The plotting function for actual (real-space) results.
    
    Creates a standardized plot for real-space results for the training phase
    and/or the prediction phase.
    
    Args:
        train_result (rescompy.TrainResult): The training result to be included
                                             in the plot.
        predict_result (rescompy.PredictResult): The prediction result to be
                                                 included in the plot.
        max_plots (int): The maximum number of signals to plot.
        y_labels (List[str]): The vertical-axis labels for the signals.
                              If not provided, arbitrary labels will be added.
        tau (float): The value of the reservior time-step.
        t_units (str): The units for time.
        save_loc (str): The location to save the finished plot.
    """

    # Insist that at least one of train_result and predict_result is provided.
    if train_result is None and predict_result is None:
        msg = "Must provide a train_result, a predict_result, or both."
        logging.error(msg)
        
    # Limit max_plots, if not enough signals are provided.
    if train_result is None:
        num_signals = predict_result.target_outputs.shape[1]
    elif predict_result is None:
        num_signals = train_result.target_outputs.shape[1]
    else:
        num_signals = max(train_result.target_outputs.shape[1],
                        predict_result.target_outputs.shape[1])
    max_plots = min(max_plots, num_signals)
        
    # Create default y_labels, if none provided.
    if y_labels is None:
        y_labels = [f"$v_{i}$" for i in range(max_plots)]
    else:
        if len(y_labels) > max_plots:
            y_labels = y_labels[:max_plots]
                    
    # Grab signals from train_result and/or predict_result.
    if train_result is not None:
        lookback_train = train_result.lookback_length
        v_train = train_result.reservoir_outputs
        v_target_train = train_result.target_outputs[lookback_train:]
        t_train = np.linspace(-v_train.shape[0]*tau, 0,
                              v_train.shape[0], False)
        tran_len = (-v_train.shape[0] + train_result.transient_length)
        tran_len *= tau
    if predict_result is not None:
        v_test = predict_result.reservoir_outputs
        v_target_test = predict_result.target_outputs
        t_test = np.linspace(0, v_test.shape[0]*tau, v_test.shape[0],
                             False)
        if v_target_test is not None:
            valid_time = predict_result.unit_valid_length*tau
    
    # Set up a few other things.
    x_label = f"t ({t_units})"
    
    # Set up  the plot.
    fig, ax = plt.subplots(max_plots)
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for plot_ind in range(max_plots):
        if train_result is not None:
            ax[plot_ind].plot(t_train, v_train[:, plot_ind],
                              color='red')
            ax[plot_ind].plot(t_train, v_target_train[:, plot_ind],
                              color='blue', ls='dashed')
            ax[plot_ind].axvline(x=tran_len, color='black',
                                 ls='dashed')
        if predict_result is not None:
            ax[plot_ind].plot(t_test, v_test[:, plot_ind], color='red')
            ax[plot_ind].plot(t_test, v_target_test[:, plot_ind],
                              color='blue', ls='dashed')
            if v_target_test is not None:
                ax[plot_ind].axvline(x=valid_time, color='black',
                                     ls='dashed')
        if train_result is not None and predict_result is not None:
            ax[plot_ind].axvline(x=0, color='black')
        if plot_ind < max_plots - 1:
            ax[plot_ind].set_xticks([])
        else:
            ax[plot_ind].set_xlabel(x_label)
        ax[plot_ind].set_ylabel(y_labels[plot_ind])
        
    # Save, if applicable.
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()

    
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_error(
    train_result:   Optional[rescompy.TrainResult]   = None,
    predict_result: Optional[rescompy.PredictResult] = None,
    max_plots:      int                              = 3,
    log_scale:      bool                             = True,
    y_labels:       Optional[List[str]]              = None,
    tau:            float                            = 1.0,
    t_units:        str                              = r'$\tau$',
    save_loc:       Optional[str]                    = None,
    ):
    """The plotting function for error results.
    
    Creates a standardized plot for error results for the training phase
    and/or the prediction phase.
    
    Args:
        train_result (rescompy.TrainResult): The training result to be included
                                             in the plot.
        predict_result (rescompy.PredictResult): The prediction result to be
                                                 included in the plot.
        max_plots (int): The maximum number of signals to plot.
        log_scale (bool): If true, will convert the vertical axis to log-scale.
        y_labels (List[str]): The vertical-axis labels for the signals.
                              If not provided, arbitrary labels will be added.
        tau (float): The value of the reservior time-step.
        t_units (str): The units for time.
        save_loc (str): The location to save the finished plot.
    """

    # Insist that at least one of train_result and predict_result is provided.
    if train_result is None and predict_result is None:
        msg = "Must provide a train_result, a predict_result, or both."
        logging.error(msg)
        
    # Insist that, if provided, predict_result must have target outputs.
    if predict_result is not None:
        if predict_result.target_outputs is None:
            msg = "predict_result must have target_outputs to plot error."
            logging.error(msg)
        
    # Limit max_plots, if not enough signals are provided.
    if train_result is None:
        num_signals = predict_result.target_outputs.shape[1]
    elif predict_result is None:
        num_signals = train_result.target_outputs.shape[1]
    else:
        num_signals = max(train_result.target_outputs.shape[1],
                        predict_result.target_outputs.shape[1])
    max_plots = min(max_plots, num_signals)
        
    # Create default y_labels, if none provided.
    if y_labels is None:
        y_labels = [f"$v_{i}$" for i in range(max_plots)]
    else:
        if len(y_labels) > max_plots:
            y_labels = y_labels[:max_plots]
                    
    # Grab signals from train_result and/or predict_result.
    if train_result is not None:
        lookback_train = train_result.lookback_length
        v_train = train_result.reservoir_outputs
        v_target_train = train_result.target_outputs[lookback_train:]
        t_train = np.linspace(-v_train.shape[0]*tau, 0,
                              v_train.shape[0], False)
        error_train = np.abs(v_target_train - v_train)
        tran_len = (-v_train.shape[0] + train_result.transient_length)
        tran_len *= tau
    if predict_result is not None:
        v_test = predict_result.reservoir_outputs
        v_target_test = predict_result.target_outputs
        t_test = np.linspace(0, v_test.shape[0]*tau, v_test.shape[0],
                             False)
        error_test = np.abs(v_target_test - v_test)
        valid_time = predict_result.unit_valid_length*tau
    
    # Set up a few other things.
    x_label = f"t ({t_units})"
    
    # Set up  the plot.
    fig, ax = plt.subplots(max_plots)
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for plot_ind in range(max_plots):
        if train_result is not None:
            ax[plot_ind].plot(t_train, error_train[:, plot_ind],
                              color='red')
            ax[plot_ind].axvline(x=tran_len, color='black',
                                 ls='dashed')
        if predict_result is not None:
            ax[plot_ind].plot(t_test, error_test[:, plot_ind],
                              color='red')
            ax[plot_ind].axvline(x=valid_time, color='black',
                                 ls='dashed')
        if train_result is not None and predict_result is not None:
            ax[plot_ind].axvline(x=0, color='black')
        if plot_ind < max_plots - 1:
            ax[plot_ind].set_xticks([])
        else:
            ax[plot_ind].set_xlabel(x_label)
        ax[plot_ind].set_ylabel(y_labels[plot_ind])
        if log_scale:
            ax[plot_ind].set_yscale('log')
        
    # Save, if applicable.
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def plot_phase_space(
    train_result:      Optional[rescompy.TrainResult]   = None,
    predict_result:    Optional[rescompy.PredictResult] = None,
    axes:              Tuple[int, ...]                  = (0, 1),
    include_transient: bool                             = True,
    save_loc:          Optional[str]                    = None,
    ):
    """The plotting function for phase-space results.
    
    Creates a standardized plot for phase-space for the training phase and/or
    the prediction phase.
    
    Args:
        train_result (rescompy.TrainResult): The training result to be included
                                             in the plot.
        predict_result (rescompy.PredictResult): The prediction result to be
                                                 included in the plot.
        axes (Tuple(int)): The axes to be plotted in phase space.
                           If 2 axes are provided, will generate 2D plots.
                           If 3 axes are provided, will generate 3D plots.
        save_loc (str): The location to save the finished plot.
    """
        
    # Insist that at least one of train_result and predict_result is provided.
    if train_result is None and predict_result is None:
        msg = "Must provide a train_result, a predict_result, or both."
        logging.error(msg)
        
    # Insist that either 2 or 3 axes are provided.
    if len(axes) not in [2, 3]:
        msg = "len(axes) must be either 2 or 3."
        logging.error(msg)
        
    # Grab signals from training and/or prediction results.
    if train_result is not None:
        v_00 = train_result.reservoir_outputs[
            train_result.transient_length:, axes]
        v_00_p = train_result.reservoir_outputs[
            :train_result.transient_length, axes]
        v_10 = train_result.target_outputs[:, axes]
    if predict_result is not None:
        v_01 = predict_result.reservoir_outputs[:, axes]
        if predict_result.target_outputs is not None:
            v_11 = predict_result.target_outputs[:, axes]
        
    # Set up a few other things.
    if train_result is not None and predict_result is not None:
        num_rows = 2
    else:
        num_rows = 1
    if predict_result is not None:
        if predict_result.target_outputs is None:
            num_cols = 1
        else:
            num_cols = 2
    else:
        num_cols = 2
            
    # Iterate through the plots.
    position = 0
    fig = plt.figure()
    
    # Add training result plots.
    if train_result is not None:
        
        # Create 2D plot if 2 axes provided.
        if len(axes) == 2:
            
            # Add reservoir outputs.
            position += 1
            ax1 = fig.add_subplot(num_rows, num_cols, position)
            if include_transient:
                ax1.scatter(v_00_p[:, 0], v_00_p[:, 1], s=0.1,
                            color='purple')
            ax1.scatter(v_00[:, 0], v_00[:, 1], s=0.1, color='blue')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_ylabel('Training')
            if position == 1:
                ax1.set_title('Reservoir Outputs')
                
            # Add target outputs.
            position += 1
            ax2 = fig.add_subplot(num_rows, num_cols, position,
                                  sharex=ax1, sharey=ax1)
            ax2.scatter(v_10[:, 0], v_10[:, 1], s=0.1, color='orange')
            ax2.set_xticks([])
            ax2.set_yticks([])
            if position == 2:
                ax2.set_title('Target Outputs')
                
        # Create 3D plot if 3 axes provided.
        else:
            
            # Add reservoir outputs.
            position += 1
            ax3 = fig.add_subplot(num_rows, num_cols, position,
                                 projection='3d')
            if include_transient:
                ax3.scatter(v_00_p[:, 0], v_00_p[:, 1], v_00_p[:, 2],
                            alpha=0.3, s=0.1, color='blue')
            ax3.scatter(v_00[:, 0], v_00[:, 1], v_00[:, 2], s=0.1,
                       color='blue')
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_zticks([])
            if position == 1:
                ax3.set_title('Reservoir Outputs')

            # Add target outputs.
            position += 1
            ax4 = fig.add_subplot(num_rows, num_cols, position,
                                  projection='3d', sharex=ax3,
                                  sharey=ax3)
            ax4.scatter(v_10[:, 0], v_10[:, 1], v_10[:, 2], s=0.1,
                       color='orange')
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_zticks([])
            if position == 2:
                ax4.set_title('Target Outputs')
            
    # Add prediction result plots.
    if predict_result is not None:
        
        # Create 2D plot if 2 axes provided.
        if len(axes) == 2:
            
            # Add reservoir outputs.
            position += 1
            ax = fig.add_subplot(num_rows, num_cols, position)
            ax.scatter(v_01[:, 0], v_01[:, 1], s=0.1, color='green')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('Prediction')
            if position == 1:
                ax.set_title('Reservoir Outputs')
                
            # Add target outputs.
            position += 1
            ax = fig.add_subplot(num_rows, num_cols, position)
            ax.scatter(v_11[:, 0], v_11[:, 1], s=0.1, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            if position == 2:
                ax.set_title('Target Outputs')
                
        # Create 3D plot if 3 axes provided.
        else:
            
            # Add reservoir outputs.
            position += 1
            ax = fig.add_subplot(num_rows, num_cols, position,
                                 projection='3d')
            ax.scatter(v_01[:, 0], v_01[:, 1], v_01[:, 2], s=0.1,
                       color='green')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            if position == 1:
                ax.set_title('Reservoir Outputs')
                
            # Add target outputs.
            if predict_result.target_outputs is not None:
                position += 1
                ax = fig.add_subplot(num_rows, num_cols, position,
                                     projection='3d')
                ax.scatter(v_11[:, 0], v_11[:, 1], v_11[:, 2], s=0.1,
                           color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                if position == 2:
                    ax.set_title('Target Outputs')
    plt.tight_layout()
    
    # Save, if applicable.
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def lyapunov_exponent(esn, train_result):
    pass


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def lyapunov_spectrum(esn, train_result, k):
    pass


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def synchronization_time(esn):
    pass