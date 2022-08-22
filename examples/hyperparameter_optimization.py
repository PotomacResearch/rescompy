# Import statements.
import rescompy

# Fix a seed and set some experimental parameters.
seed = 14
size = 1000
connections = 10
spectral_radius = 0.99
input_strength = 1.0
bias_strength = 0.5
leaking_rate = 0.1
transient_length = 10000
train_length = 50000
validation_length = 100
test_length = 5000

# Create and normalize a long Lorenz signal.
u = rescompy.benchmarks.get_lorenz(seed=seed, return_length=train_length
                          + test_length)
u = rescompy.Standardizer(u).standardize(u)


def main():
    
    # Partition into training and test signals.
    u_train = u[:train_length-1]
    v_train = u[1:train_length]
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
        
    # Set training and prediction arguments for hyperparameter optimization.
    train_args = {'transient_length': transient_length,
                  'inputs': u_train[:-validation_length],
                  'target_outputs': v_train[:-validation_length]}
    predict_args = {'predict_length': validation_length,
                    'target_outputs': v_train[-validation_length:]}
    opt_esn = rescompy.optimize_hyperparameters(esn, train_args,
                                                predict_args)
    
    # Train the ESN on the training signal.
    train_result = opt_esn.train(transient_length=transient_length,
        inputs=u_train, target_outputs=v_train)
    
    # Predict autonomously and compare against the test signal.
    predict_result = opt_esn.predict(train_result=train_result,
                                 predict_length=test_length,
                                 target_outputs=v_test)
    
    # Plot the results and report the unit valid length.
    rescompy.plotter.plot_actual(predict_result=predict_result)
    print (predict_result.unit_valid_length)
    print (opt_esn)

    
if __name__ == '__main__':
    main()