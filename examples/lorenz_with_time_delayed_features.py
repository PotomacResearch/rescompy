# Import statements.
import numpy as np
import rescompy
import numba

# Fix a seed and set some experimental parameters.
seed = 15
size = 1000
transient_length = 10000
train_length = 50000
test_length = 5000
resync_length = 500
spectral_radius = .59
input_strength = .63
bias_strength = 0.5
leaking_rate = 0.1

# Include the current reservoir state and the previous two reservoir states in
# the current feature vector.
states_lookback_length = 2
states_decimation = 1

# Include the current reservoir input and the inputs from two time-steps and
# four time-steps in the past in the current feature vector.
inputs_lookback_length = 4
inputs_decimation = 2

# Create and normalize a long Lorenz signal.
u = rescompy.benchmarks.get_lorenz(seed=seed, return_length=train_length
                          + test_length)
u = rescompy.Standardizer(u).standardize(u)

def main():

    ### ~~~ EXPERIMENT 1: Lorenz control with time-delayed inputs and states ~~~ ###
    # For this experiment, we will train on all three variables and then 
    # predict after the training period by feeding back the z-component but 
	# providing x and y as inputs.
      
    inputs_train = u[:train_length-1]
    target_outputs_train = u[1:train_length]
    inputs_test = u[train_length-1:-1, :2]
    target_outputs_test = u[train_length:]
      
    # Create an ESN with a fixed seed.
    esn = rescompy.ESN(3,size, 10, spectral_radius, input_strength,
					   bias_strength, leaking_rate, seed)
      
    feature_function = rescompy.features.StatesAndInputsTimeShifted(
	states_lookback_length = states_lookback_length,
	inputs_lookback_length = inputs_lookback_length,
	states_decimation = states_decimation,
	inputs_decimation = inputs_decimation
	)

    # Train the ESN on the training signals.
    train_result = esn.train(transient_length, inputs = inputs_train,
							 target_outputs = target_outputs_train,
							 feature_function = feature_function)
    print("Feature size: ", train_result.features.shape[1])
              
    # Define an observer mapper function.
    @numba.jit(nopython=True, fastmath=True)
    def mapper(inputs, outputs):
        return np.concatenate((inputs, outputs[2:]))
      
    # Predict the signal in open-loop configuration directly after training.
    predict_result = esn.predict(train_result, inputs=inputs_test,
								 target_outputs = target_outputs_test,
								 mapper =  mapper)
    rescompy.plotter.plot_actual(predict_result=predict_result)
    print ("Train and Predict RMSE", np.mean(predict_result.rmse))
	
    # Predict the signal in open-loop configuration by resynchronizing to the end 
	# of the training period, which comes directly before the validation period.
    predict_result = esn.predict(train_result, inputs=inputs_test,
								 resync_signal = inputs_train[-resync_length:],
								 target_outputs = target_outputs_test,
								 mapper =  mapper)
    rescompy.plotter.plot_actual(predict_result=predict_result)
    print ("Resync and Predict RMSE",  np.mean(predict_result.rmse))
    	
    # Predict the signal in open-loop configuration by providing 
	# lookback_inputs and lookback_states directly.
    predict_result = esn.predict(train_result,
								 inputs = inputs_test,
								 lookback_inputs = inputs_train[-inputs_lookback_length-1:],
								 lookback_states = train_result.states[-states_lookback_length-1:],
								 target_outputs = target_outputs_test,
								 mapper =  mapper)
    rescompy.plotter.plot_actual(predict_result=predict_result)
    print ("Predict with Manually Supplied Lookbacks RMSE", np.mean(predict_result.rmse))
	
   
if __name__ == '__main__':
    main()