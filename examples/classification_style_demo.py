# Import statements.
import numpy as np
import rescompy
import matplotlib.pyplot as plt
from typing import Callable

# Fix a seed and set some experimental parameters.
esn_seed = 15
size = 500
spectral_radius = .59
input_strength = .63
bias_strength = 0.5
leaking_rate = 0.1

# Set the length of the input signals and the number of training and testing 
# samples.
signal_length = 150
num_train_samples = 10000
num_test_samples = 100
    
# Arguments for Mixed Reservoir States feature vector. Include every twentieth 
# time step and do not allow more than ten states in a feature vector.
decimation_time = 20
max_num_states = 10

# Define a function to get input signals and target vectors for either training
# or testing.
def get_signals_and_targets(
        signal_length:      int,
        num_samples:        int,
        seed:               int,
        standardizer:       Callable = None
        ):
    
    sigmas = 10 * np.random.uniform(.8, 1.2, num_samples)
    rhos = 28 * np.random.uniform(.8, 1.2, num_samples)
    seeds = np.arange(seed, seed + num_samples + 1)
    
    targets = np.concatenate((sigmas[:, None], rhos[:, None]), axis = 1)
    targets = list(targets.reshape((num_samples, 1, 2)))
    
    signals = [rescompy.benchmarks.get_lorenz(
        seed = seed, return_length = signal_length, sigma = sigma, rho = rho)
        for seed, sigma, rho in zip(seeds, sigmas, rhos)]
    
    if standardizer is None:
        standardizer = rescompy.rescompy.Standardizer(signals[0])
        
    signals = [standardizer.standardize(signal) for signal in signals]
    
    return signals, targets, standardizer


def main():
	
    # Create and normalize a collection of Lorenz signals for training.
    train_signals, train_targets, standardizer = get_signals_and_targets(
        signal_length = signal_length, num_samples = num_train_samples, seed = 1)
    
    # Create and normalize a collection of Lorenz signals for testing.
    test_signals, test_targets, _ = get_signals_and_targets(
        signal_length = signal_length, num_samples = num_train_samples,
        seed = num_train_samples + 1, standardizer = standardizer)
    
    ### ~~~ EXPERIMENT 1: Infer the dynamical parameters of Lorenz signals ~~~ ###
    # We train to map short signals from trajectories on the Lorenz attractors 
	# with different values of the parameters rho and sigma to the 
	# corresponding values of rho and sigma using the Mixed Reservoir State
	# feature function. We then apply the learned mapping to a test signal of 
	# of the same length as each of the training signals.

    # Create an ESN with a fixed seed.
    esn = rescompy.ESN(3, size, 3, spectral_radius, input_strength,
					   bias_strength, leaking_rate, esn_seed)	

    train_result = esn.train(
		transient_length = 0,
		inputs = train_signals,
		target_outputs = train_targets,
		feature_function = rescompy.features.MixedReservoirStates(
			decimation_time,
			max_num_states
			),
		initial_state = np.zeros(esn.size),
		batch_size = 100,
		accessible_drives = [0, 1, -3, -2, -1],
		regression = rescompy.regressions.batched_ridge()
		)
	
    mean_rmse = 0
    plt.figure(constrained_layout = True)
    for j in range(num_test_samples):
        predict_result = esn.predict(
			train_result = train_result,
			inputs = test_signals[j],
			initial_state = np.zeros(esn.size),
			target_outputs = test_targets[j]
			)
        predicted_parameters = predict_result.reservoir_outputs
        plt.plot(predict_result.target_outputs[0, 0], predict_result.target_outputs[0, 1],
				 c = "red", marker = "o", linestyle = "none")
        plt.plot(predicted_parameters[0, 0], predicted_parameters[0, 1],
				 c = "blue", marker = "o", linestyle = "none")
        mean_rmse += predict_result.rmse / num_test_samples
    plt.legend(["Truth", "Predictions"])
    plt.xlabel("$\\sigma$")
    plt.ylabel("$\\rho$")
    plt.show()	
    print("Mean RMSE: ", mean_rmse)
	
    ### ~~~ EXPERIMENT 2: Infer the dynamical parameters of Lorenz signals ~~~ ###
    # We train to map short signals from trajectories on the Lorenz attractors 
	# with different values of the parameters rho and sigma to the 
	# corresponding values of rho and sigma using the Final State Only
	# feature function. We then apply the learned mapping to a test signal of 
	# of the same length as each of the training signals.

    # Create an ESN with a fixed seed.

    train_result = esn.train(
		transient_length = 0,
		inputs = train_signals,
		target_outputs = train_targets,
		feature_function = rescompy.features.FinalStateOnly(),
		initial_state = np.zeros(esn.size),
		batch_size = 100,
		accessible_drives = "final",
		regression = rescompy.regressions.batched_ridge()
		)

    plt.figure(constrained_layout = True)
    mean_rmse = 0
    for j in range(num_test_samples):
        predict_result = esn.predict(
			train_result = train_result,
			inputs = test_signals[j],
			initial_state = np.zeros(esn.size),
			target_outputs = test_targets[j]
			)
        predicted_parameters = predict_result.reservoir_outputs
        plt.plot(predict_result.target_outputs[0, 0], predict_result.target_outputs[0, 1],
				 c = "red", marker = "o", linestyle = "none")
        plt.plot(predicted_parameters[0, 0], predicted_parameters[0, 1],
				 c = "blue", marker = "o", linestyle = "none")
        mean_rmse += predict_result.rmse / num_test_samples
    plt.legend(["Truth", "Predictions"])
    plt.xlabel("$\\sigma$")
    plt.ylabel("$\\rho$")
    plt.show()	
    print("Mean RMSE: ", mean_rmse)    

if __name__ == '__main__':
    main()