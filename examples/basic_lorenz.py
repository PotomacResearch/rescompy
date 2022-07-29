# Import statements.
import numpy as np
import rescompy

# Fix a seed and set some experimental parameters.
seed = 15
size = 1000
transient_length = 10000
train_length = 50000
test_length = 5000
leaking_rate = 0.1
bias_strength = 0.5

# Create and normalize a long Lorenz signal.
u = rescompy.benchmarks.get_lorenz(seed=seed, return_length=train_length
                          + test_length)
u = rescompy.Standardizer(u).standardize(u)


def main():
    
    ### ~~~ EXPERIMENT 1: Prediction ~~~ ###
    # For this experiment, we will input all 3 variables and predict them
    # autonomously after a training period.
    
    # Partition into training and test signals.
    u_train = u[:train_length-1]
    v_train = u[1:train_length]
    u_test = u[train_length-1:train_length+test_length-1]
    v_test = u[train_length:train_length+test_length]
    
    # Create the ESN.
    esn = rescompy.ESN(input_dimension=3, size=size, 
                       leaking_rate=leaking_rate,
                       seed=seed, bias_strength=bias_strength)
    
    # Train the ESN on the training signal.
    train_result = esn.train(transient_length=transient_length,
        inputs=u_train, target_outputs=v_train)
    
    # Predict autonomously and compare against the test signal.
    predict_result = esn.predict(train_result=train_result,
                                 predict_length=test_length,
                                 target_outputs=v_test)
    
    # Plot the results and report the unit valid length.
    rescompy.plotter.plot_actual(predict_result=predict_result)
    print (predict_result.unit_valid_length)
    
    ### ~~~ EXPERIMENT 2: Observation ~~~ ###
    # This time, we will observe the first 2 Lorenz variables and output the 3rd.
    
    # Partition into training and test signals.
    # Note the different dimensions of u_train/u_test versus v_train, v_test.
    u_train = u[:train_length, :2]
    v_train = u[:train_length, 2:]
    u_test = u[train_length:train_length+test_length, :2]
    v_test = u[train_length:train_length+test_length, 2:]
    
    # Create the ESN.
    # Note the different input dimension.
    esn = rescompy.ESN(input_dimension=2, size=size, 
                    leaking_rate=leaking_rate,
                    seed=seed, bias_strength=bias_strength)
    
    # Train the ESN on the training signal.
    train_result = esn.train(transient_length=transient_length,
                             inputs=u_train, target_outputs=v_train)
    
    # Define a mapper for the observation problem.
    # This function enforces no feedback connections by only returning the inputs.
    def observer_mapper(inputs, outputs):
        return inputs
    
    # Predict on the test signal.
    predict_result = esn.predict(train_result=train_result,
                                 predict_length=test_length,
                                 inputs=u_test,
                                 target_outputs=v_test,
                                 mapper=observer_mapper)
    
    # Plot the results and report the RMSE over the test period.
    rescompy.plotter.plot_actual(predict_result=predict_result)
    print (np.mean(predict_result.rmse))

    
if __name__ == '__main__':
    main()