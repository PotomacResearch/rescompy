# Import statements.
import rescompy
import logging


logger = logging.getLogger()
logger.setLevel(logging.ERROR)
SEED = 17


def main():

    # Grab a Lorenz signal, standardize it, and parse into inputs, outputs,
    # training, and testing.
    u = rescompy.benchmarks.get_lorenz(return_length=7000, seed=SEED)
    u = rescompy.Standardizer(u).standardize(u)
    inputs_train = u[:5000-1]
    target_outputs_train = u[1:5000]
    target_outputs_test = u[5000:]
    
    # Create an ESN with a fixed seed.
    esn = rescompy.ESN(3, 2000, 10, 0.99, 1.0, 0.5, 0.1, SEED)
    
    # Train the ESN on the training signal.
    train_result = esn.train(1000, inputs_train,
                             target_outputs_train)
            
    # Predict the signal in closed-loop configuration.
    predict_result = esn.predict(train_result,
                                 target_outputs=target_outputs_test)
    
    # Visualize the results a few different ways.
    rescompy.plotter.plot_actual(train_result, predict_result,
                                 y_labels=['x', 'y', 'z'],
                                 tau=0.009056,
                                 t_units='Lyapunov times')
    rescompy.plotter.plot_error(train_result, predict_result,
                                y_labels=['x', 'y', 'z'], tau=0.009056,
                                t_units='Lyapunov times')
    rescompy.plotter.plot_phase_space(train_result, predict_result)


if __name__ == '__main__':
    main()