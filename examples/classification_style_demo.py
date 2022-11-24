# Import statements.
import numpy as np
import rescompy
import matplotlib.pyplot as plt

# Fix a seed and set some experimental parameters.
esn_seed = 15
size = 500
signal_length = 150
spectral_radius = .59
input_strength = .63
bias_strength = 0.5
leaking_rate = 0.1

# Arguments for Mixed Reservoir States feature vector. Include every twentieth 
# time step and do not allow more than ten states in a feature vector.
decimation_time = 20
max_num_states = 10

num_train_signals = 10000
num_test_signals = 100

# Create and normalize a collection of Lorenz signals for training.
train_sigmas = 10 * np.random.uniform(.8, 1.2, num_train_signals)
train_rhos = 28 * np.random.uniform(.8, 1.2, num_train_signals)
train_seeds = np.arange(1, num_train_signals + 1)

train_targets = np.concatenate((train_sigmas[:, None], 
								train_rhos[:, None]), axis = 1)
train_targets = list(train_targets.reshape((num_train_signals, 1, 2)))
lorenz_signals = [None] * num_train_signals
lorenz_signals[0] = rescompy.benchmarks.get_lorenz(
	seed = train_seeds[0],
	return_length = signal_length,
	sigma = train_sigmas[0],
	rho = train_rhos[0],
	)
standardizer = rescompy.rescompy.Standardizer(lorenz_signals[0])
lorenz_signals[0] = standardizer.standardize(lorenz_signals[0])
for i in range(1, num_train_signals):
    lorenz_signals[i] = rescompy.benchmarks.get_lorenz(
		seed = train_seeds[i],
		return_length = signal_length,
		sigma = train_sigmas[i],
		rho = train_rhos[i],
		)
    lorenz_signals[i] = standardizer.standardize(lorenz_signals[i])

# Create and normalize a collection of Lorenz signals for testing.
test_sigmas = 10 * np.random.uniform(.8, 1.2, num_test_signals)
test_rhos = 28 * np.random.uniform(.8, 1.2, num_test_signals)
test_seeds = np.arange(num_train_signals + 1,
					   num_train_signals + num_test_signals + 1)

test_targets = np.concatenate((test_sigmas[:, None],
							   test_rhos[:, None]), axis = 1)
test_targets = list(test_targets.reshape((num_test_signals, 1, 2)))
test_signals = [None] * num_test_signals
test_signals[0] = rescompy.benchmarks.get_lorenz(
	seed = test_seeds[0],
	return_length = signal_length,
	sigma = test_sigmas[0],
	rho = test_rhos[0],
	)
test_signals[0] = standardizer.standardize(test_signals[0])
for i in range(1, num_test_signals):
    test_signals[i] = rescompy.benchmarks.get_lorenz(
		seed = test_seeds[i],
		return_length = signal_length,
		sigma = test_sigmas[i],
		rho = test_rhos[i],
		)
    test_signals[i] = standardizer.standardize(test_signals[i])
	

def main():
	
    ### ~~~ EXPERIMENT 1: Infer the dynamical parameters of Lorenz signals ~~~ ###
    # We train to map short signals from trajectories on the Lorenz attractors 
	# with different values of the parameters rho and sigma to the 
	# corresponding values of rho and sigma using the Mixed Reservoir State
	# feature function. We then apply the learned mapping to a test signal of 
	# of the same length as each of the training signals.

    # Create an ESN with a fixed seed.
    esn = rescompy.ESN(3, size, 3, spectral_radius, input_strength,
					   bias_strength, leaking_rate, esn_seed)	

    train_result = esn.train_batched(
		transient_lengths = 0,
		inputs = lorenz_signals,
		target_outputs = train_targets,
		feature_function = rescompy.features.MixedReservoirStates(
			decimation_time,
			max_num_states
			),
		initial_state = np.zeros(esn.size),
		batch_size = 100,
		accessible_drives = "final",
		)
	
    mean_rmse = 0
    plt.figure(constrained_layout = True)
    for j in range(num_test_signals):
        predict_result = esn.predict(
			train_result = train_result,
			predict_length = 0,
			inputs = test_signals[j],
			initial_state = np.zeros(esn.size),
			target_outputs = test_targets[j]
			)
        predicted_parameters = predict_result.reservoir_outputs
        plt.plot(predict_result.target_outputs[0, 0], predict_result.target_outputs[0, 1],
				 c = "red", marker = "o", linestyle = "none")
        plt.plot(predicted_parameters[0, 0], predicted_parameters[0, 1],
				 c = "blue", marker = "o", linestyle = "none")
        mean_rmse += predict_result.rmse / num_test_signals
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
    esn = rescompy.ESN(3, size, 10, spectral_radius, input_strength,
					   bias_strength, leaking_rate, esn_seed)	

    train_result = esn.train_batched(
		transient_lengths = 0,
		inputs = lorenz_signals,
		target_outputs = train_targets,
		feature_function = rescompy.features.FinalStateOnly(),
		initial_state = np.zeros(esn.size),
		batch_size = 100,
		accessible_drives = "final",
		)

    plt.figure(constrained_layout = True)
    mean_rmse = 0
    for j in range(num_test_signals):
        predict_result = esn.predict(
			train_result = train_result,
			predict_length = 0,
			inputs = test_signals[j],
			initial_state = np.zeros(esn.size),
			target_outputs = test_targets[j]
			)
        predicted_parameters = predict_result.reservoir_outputs
        plt.plot(predict_result.target_outputs[0, 0], predict_result.target_outputs[0, 1],
				 c = "red", marker = "o", linestyle = "none")
        plt.plot(predicted_parameters[0, 0], predicted_parameters[0, 1],
				 c = "blue", marker = "o", linestyle = "none")
        mean_rmse += predict_result.rmse / num_test_signals
    plt.legend(["Truth", "Predictions"])
    plt.xlabel("$\\sigma$")
    plt.ylabel("$\\rho$")
    plt.show()	
    print("Mean RMSE: ", mean_rmse)
    

if __name__ == '__main__':
    main()