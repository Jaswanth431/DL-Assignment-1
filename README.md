# DL-Assignment-1
## DL_Assignment_1.ipynb:
This document serves as a guide for the code implementing a Neural Network library with functionalities for training a neural network on image classification tasks.

## Key Components:

### NeuralNetwork Class:
- This class encapsulates the core functionalities of a neural network.
- It takes the number of input and output layer neurons, along with a configuration dictionary, as input during initialization.
- The configuration dictionary specifies various hyperparameters like the number of hidden layers, the number of neurons in each hidden layer, the activation function, the loss function, the optimizer, and other training parameters.
-The class provides methods for:
  - Forward propagation: This method calculates the activation values for each layer in the network for a given input.
  - Backpropagation: This method calculates the gradients of the loss function with respect to the weights and biases of each layer.
  - Various optimization algorithms: The class implements different optimization algorithms like stochastic gradient descent (SGD), momentum gradient descent, Nesterov accelerated gradient descent, and RMSprop to update the weights and biases of the network during training.
Code Structure:

The code is structured as follows:

1.Import Libraries:

    -Necessary libraries like pandas, numpy, wandb, matplotlib, and seaborn are imported.
2. Data Loading and Preprocessing:

    - The Fashion MNIST dataset is loaded using fashion_mnist.load_data.
    - A function plot_selected_images is defined to select and visualize a few class labels from the training data. This function utilizes wandb to log the image as an artifact.
    - The training data is flattened (converted from a 2D image matrix to a 1D vector) and normalized by dividing each pixel value by 255.
    - The data is then split into training and validation sets (90% for training and 10% for validation).
3. Neural Network Class Implementation:

  - The NeuralNetwork class is defined with the following functionalities:
    - Initialization:
        - Takes the number of input and output layer neurons, along with a configuration dictionary, as input.
        - Initializes the number of hidden layers, the number of neurons in each hidden layer, and other hyperparameters based on the configuration dictionary.
        - Initializes the weights and biases for all layers of the network. The weights can be initialized randomly or using the Xavier initialization method.
  - Forward Propagation (forward_propogate method):
      - Takes a flattened image as input.
      - Calculates the activation values for each layer in the network using the chosen activation function (e.g., ReLU, sigmoid).
      - Returns the activation vectors for both the output layer and all hidden layers.
  - Backpropagation (back_propagation method):
      - Takes the forward propagation outputs (activation vectors), the actual class label, and the input image as input.
      - Calculates the gradients of the loss function with respect to the weights and biases of each layer based on the chosen loss function (e.g., cross-entropy, mean squared error).
      - Returns the gradients for weights and biases of each layer.
4 .Optimization Algorithms:
- The class implements several optimization algorithms:
   - Stochastic Gradient Descent (SGD): This is the basic SGD implementation where the weights and biases are updated after processing each training example.
   - Momentum Gradient Descent: This extends SGD by incorporating momentum, which helps to accelerate convergence in certain cases.
   - Nesterov Accelerated Gradient Descent: This is a variant of momentum that can provide faster convergence compared to standard momentum.
   - RMSprop: This is an adaptive learning rate optimization algorithm that addresses the issue of oscillating gradients observed in SGD.
- Each optimization algorithm takes the training data, validation data, and hyperparameters like learning rate, batch size, and weight decay as input.
- It iterates through the training epochs, performing mini-batch updates on the weights and biases using the backpropagated gradients.
- Within each epoch, the algorithm might calculate and log the loss on both the training and validation sets to monitor the training progress.

5. Training the Network:

- The code  creates an instance of the NeuralNetwork class with a specified configuration dictionary containing the desired hyperparameters.
- An appropriate optimization algorithm method (e.g., stochastic_gradient_descent or momentum_gradient_descent) is called on the instance, providing the training -and validation data.
= This method performs the training process by iterating through epochs and updating the network weights and biases.

# Run DL_Assignment_1.ipynb
- Just update ```h_param_config_defaults``` with required configuration and run the next cell. It will create an instance of neural network with the given configuration.
- Gradient descent algorithm will be called and accuracy and loss will be logged to wandb

# Run train.py
- Open a terminal or command prompt.
- Run the script using the following command:
```python train.py [arguments]```
- Replace [arguments] with the desired command-line arguments (see below for available arguments).

### Command-line Arguments:
```
-wp or --wandb_project: Specifies the project name used to track experiments in the Weights & Biases dashboard.
-we or --wandb_entity: Specifies the WandB entity used to track experiments.
-d or --dataset: Chooses the dataset to be used (mnist or fashion_mnist).
-e or --epochs: Sets the number of epochs to train the neural network.
-b or --batch_size: Specifies the batch size used for training.
-l or --loss: Chooses the loss function (mean_squared_error or cross_entropy).
-o or --optimizer: Chooses the optimizer (sgd, momentum, nestrov, rmsprop, adam, or nadam).
-lr or --learning_rate: Sets the learning rate used to optimize model parameters.
-m or --momentum: Sets the momentum used by momentum and nag optimizers.
-beta or --beta: Sets the beta used by the rmsprop optimizer.
-beta1 or --beta1: Sets the beta1 used by adam and nadam optimizers.
-beta2 or --beta2: Sets the beta2 used by adam and nadam optimizers.
-eps or --epsilon: Sets the epsilon used by optimizers.
-w_d or --weight_decay: Sets the weight decay used by optimizers.
-w_i or --weight_init: Chooses the weight initialization method (random or xavier).
-nhl or --num_layers: Specifies the number of hidden layers used in the feedforward neural network.
-sz or --hidden_size: Sets the number of hidden neurons in a feedforward layer.
-a or --activation: Chooses the activation function (identity, sigmoid, tanh, or relu).
```
Example Usage:

```python train.py -wp myproject -we myname -d fashion_mnist -e 10 -b 32 -l cross_entropy -o adam -lr 0.0001 -m 0.9 -beta 0.5 -beta1 0.9 -beta2 0.999 -eps 0.000001 -w_d 0 -w_i xavier -nhl 3 -sz 128 -a relu```
- This command will train a neural network on the Fashion MNIST dataset for 10 epochs using the Adam optimizer with specified parameters, and track the experiments in the "myproject" project under the "myname" entity on WandB.


