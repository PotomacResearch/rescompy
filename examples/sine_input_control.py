# Import statements.
import rescompy
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import time
import logging


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
SEED = 17


def main():
    
    # Create the random state for reproducibility.
    rng = default_rng(SEED)
    
    # Create an ESN for the task.
    esn = rescompy.ESN(2, 500, 10, 0.5, 1.0, 0, 0.15, SEED)
    
    # Create a batch of sine waves with different amplitudes and frequencies.
    # Parse each into inputs, outputs, training, and testing.
    inputs_train = []
    target_outputs_train = []
    inputs_test = []
    target_outputs_test = []
    for i in range(110):
        amp = rng.uniform(low=0.5, high=1.5)
        sine = amp*np.sin(2*np.pi*np.linspace(0, 20, 2000))
        u = np.vstack((np.repeat(amp - 1, 2000), sine)).T
        inputs_train += [u[:1000-1]]
        target_outputs_train += [u[1:1000, 1]]
        inputs_test += [u[1000-1:2000-1, 0]]
        target_outputs_test += [u[1000:2000, 1]]
        
    # Visualize these different signals.
    fig, ax = plt.subplots(2, 2, sharey=True)
    ax[0,0].plot(inputs_train[0])
    ax[0,0].set_title('inputs_train')
    ax[0,0].set_xticks([])
    ax[0,1].plot(target_outputs_train[0])
    ax[0,1].set_title('target_outputs_train')
    ax[0,1].set_xticks([])
    ax[1,0].plot(inputs_test[0])
    ax[1,0].set_title('inputs_test')
    ax[1,1].plot(target_outputs_test[0])
    ax[1,1].set_title('target_outputs_test')
    plt.tight_layout()
    plt.show()
        
    # Train the ESN on the training inputs and target outputs.
    train_result = esn.train(200, inputs_train[:100],
                             target_outputs_train[:100])

    # Plot a bit of the training signals and error.
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(train_result.reservoir_outputs[:5000], color='red')
    ax[0].plot(train_result.target_outputs[:5000], color='blue',
               ls='dashed')
    ax[0].set_xticks([])
    ax[1].plot(train_result.rmse[:5000])
    ax[1].set_yscale('log')
    plt.tight_layout()
    plt.show()

    # Define the mapper function.
    # This basically says, during prediction, the reservoir is driven by the 
    # amplitude and the output is fed back where the sine wave used to be.
    def mapper(inputs, outputs):
        return np.concatenate((inputs, outputs))
    
    # Predict and plot a bunch of results.
    for i in range(100, 110):
        predict_result = esn.predict(train_result, inputs=inputs_test[i],
                                     target_outputs=target_outputs_test[i],
                                     mapper=mapper)
        plt.plot(predict_result.reservoir_outputs, color='red')
        plt.plot(predict_result.target_outputs, color='blue',
                 ls='dashed')
        plt.ylim([-1.75, 1.75])
        plt.legend(['Reservoir Output', 'Target Output'], loc='upper right')
        plt.show()
        time.sleep(0.5)
        
        
if __name__ == '__main__':
    main()