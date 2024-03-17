#importing packages
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import pandas as pd
import numpy as np
import wandb
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


#creating wandb connection
# 62cfafb7157dfba7fdd6132ac9d757ccd913aaaf



#Creating neural network
class NeuralNetwork:
    def __init__(self, input_layer_neurons, output_layer_neurons, config):
        #initializing values for the neural network
        self.hidden_layers = config["hidden_layers"]
        self.hidden_layer_neurons = config["hl_size"]
        self.input_layer_neurons = input_layer_neurons
        self.output_layer_neurons = output_layer_neurons
        self.total_layers = self.hidden_layers+1
        self.output_layer_number = self.total_layers - 1;
        self.config = config

        #input weight and bias initilization
        # contains weights and biases of all layers
        self.w = []
        self.b = []

        #weight and biases initalization randomly
        if(self.config["initialization"] == "random"):
            for i in range(0, self.total_layers):
                if i == 0:
                    temp1 = np.random.randn(self.hidden_layer_neurons, self.input_layer_neurons)
                    temp2 = np.random.randn(self.hidden_layer_neurons,1)
                elif i==self.total_layers -1:
                    temp1 = np.random.randn(self.output_layer_neurons, self.hidden_layer_neurons)
                    temp2 = np.random.randn(self.output_layer_neurons,1)
                else:
                    temp1  =  np.random.randn( self.hidden_layer_neurons, self.hidden_layer_neurons)
                    temp2 = np.random.randn(self.hidden_layer_neurons,1)
                self.w.append(temp1*0.01)
                self.b.append(temp2*0.01)

        #weight and biases initalization using Xavier method
        elif(self.config["initialization"] == "xavier"):
            for i in range(0, self.total_layers):
                if i == 0:
                    temp1 = np.random.randn(self.hidden_layer_neurons, self.input_layer_neurons)*np.sqrt(2.0/(self.hidden_layer_neurons + self.input_layer_neurons))
                    temp2 = np.zeros((self.hidden_layer_neurons,1))
                elif i==self.total_layers -1:
                    temp1 = np.random.randn(self.output_layer_neurons, self.hidden_layer_neurons)*np.sqrt(2.0/(self.hidden_layer_neurons + self.output_layer_neurons))
                    temp2 = np.zeros((self.output_layer_neurons,1))
                else:
                    temp1  =  np.random.randn( self.hidden_layer_neurons, self.hidden_layer_neurons) *np.sqrt(2.0/(self.hidden_layer_neurons + self.hidden_layer_neurons))
                    temp2 = np.zeros((self.hidden_layer_neurons,1))
                self.w.append(temp1)
                self.b.append(temp2)

    #forward prop method, takes flattned image and returns activation and preactivation vectors for output layer
    def forward_propogate(self, input):
        h = [None] * self.total_layers
        a = [None] * self.total_layers

        for i in range(0, self.total_layers):
            if(i == 0):
              a[i] = np.matmul(self.w[i],input.reshape(self.input_layer_neurons,1) ) + self.b[i]
              h[i] = self.activation(self.config["activation"],a[i])

            elif i == self.total_layers-1:
              a[i] = np.matmul(self.w[i],h[i-1] ) + self.b[i]
              h[i] = self.softmax(a[i])
            else:
              a[i] = np.matmul(self.w[i],h[i-1] ) + self.b[i]
              h[i] = self.activation(self.config["activation"],a[i])
            # print(a[i].shape)
        return h, a

    #back propagation return the gradients for weights and biases of each layer
    def back_propagation(self, h, a, actual_class, input_pixels):
       d_h = [None] * self.total_layers
       d_a =  [None] * self.total_layers
       d_w =  [None] * self.total_layers
       d_b =  [None] * self.total_layers
       y_original = np.zeros((self.output_layer_neurons, 1))
       y_original[actual_class] = 1

      #  cheking the loss fucntion and finding derivative acccordingly
       if(self.config["loss"] == "cross_entropy"):
          d_a[self.total_layers-1] = -(y_original - h[self.total_layers-1])
       elif(self.config["loss"] == "mean_squared_error"):
           d_a[self.total_layers-1] = (h[self.total_layers-1] -y_original )*h[self.total_layers-1]*(1 - h[self.total_layers-1])
       for i in range(self.total_layers-1, -1, -1):
        if(i == 0):
          d_w[i] = np.matmul(d_a[i], input_pixels.reshape(1, -1))
        else:
          d_w[i] = np.matmul(d_a[i], h[i-1].T)
        d_b[i] = np.copy(d_a[i])

        if(i-1>=0):
          d_h[i-1]=np.matmul(self.w[i].T,d_a[i])
          d_a[i-1] = d_h[i-1] * self.activation_derivative(self.config["activation"],a[i-1])

        # returns weights and bias gradients
       return d_w, d_b
    
    #momentum gd
    def momentum_gradient_descent(self, x_train_data_all, y_train_data_all, x_validation_data, y_validation_data):
      previous_w = [np.zeros_like(weight) for weight in self.w]
      previous_b = [np.zeros_like(bias) for bias in self.b]
      temp_w = [np.zeros_like(weight) for weight in self.w]
      temp_b = [np.zeros_like(bias) for bias in self.b]
      beta =self.config["momentum_beta"]
      batch_size = self.config["batch_size"]
      weight_decay = self.config["weight_decay"]
      for p in range(0, self.config["epochs"]):
        # Batch size implementation logic, taking each batch once and updating the w and b 
        for q in range(0,len(x_train_data_all), batch_size):
            x_train_data = x_train_data_all[q:q+batch_size]
            y_train_data = y_train_data_all[q:q+batch_size]
            d_w = [np.zeros_like(weight) for weight in self.w]
            d_b = [np.zeros_like(bias) for bias in self.b]

            for j in range(0, len(x_train_data)):
                h,a = self.forward_propogate(x_train_data[j])
                d_w_temp, d_b_temp = self.back_propagation(h, a, y_train_data[j], x_train_data[j])
                for k in range(self.total_layers):
                    d_w[k] += d_w_temp[k]
                    d_b[k] += d_b_temp[k]
            for k in range(self.total_layers):
                    temp_w[k] = beta * previous_w[k] + self.config["learning_rate"]*d_w[k]
                    self.w[k] =  self.w[k] - temp_w[k] - weight_decay*self.w[k]
                    previous_w[k] = temp_w[k]
                    temp_b[k] = beta * previous_b[k] + self.config["learning_rate"]*d_b[k]
                    self.b[k] -= temp_b[k]
                    previous_b[k] = temp_b[k]
        #loss calculation and logging to wandb
        if((self.config["epochs"] == 10 and p%2==1) or self.config["epochs"] == 5):
            self.calculate_loss(x_train_data_all, y_train_data_all, x_validation_data, y_validation_data, p)

    #stochastic gradient descent
    def stochastic_gradient_descent(self, x_train_data_all, y_train_data_all, x_validation_data, y_validation_data):
      batch_size = self.config["batch_size"]
      weight_decay = self.config["weight_decay"]
      for p in range(0, self.config["epochs"]):
        for q in range(0, len(x_train_data_all), batch_size):
            x_train_data = x_train_data_all[q:q+batch_size]
            y_train_data = y_train_data_all[q:q+batch_size]
            d_w = [np.zeros_like(weight) for weight in self.w]
            d_b = [np.zeros_like(bias) for bias in self.b]

            for j in range(0, len(x_train_data)):
                h,a = self.forward_propogate(x_train_data[j])
                d_w_temp, d_b_temp = self.back_propagation(h, a, y_train_data[j], x_train_data[j])
                for k in range(self.total_layers):
                        d_w[k] += d_w_temp[k]
                        d_b[k] += d_b_temp[k]
            for k in range(self.total_layers):
                self.w[k] = self.w[k] -  self.config["learning_rate"]*d_w[k] - weight_decay*self.w[k]
                self.b[k] -= self.config["learning_rate"]*d_b[k]
        if((self.config["epochs"] == 10 and p%2==1) or self.config["epochs"] == 5):
            self.calculate_loss(x_train_data_all, y_train_data_all, x_validation_data, y_validation_data, p)
    

    #nestrov gradient descent
    def nestrov_gradient_descent(self, x_train_data_all, y_train_data_all, x_validation_data, y_validation_data):
      previous_w = [np.zeros_like(weight) for weight in self.w]
      previous_b = [np.zeros_like(bias) for bias in self.b]
      temp_w = [np.zeros_like(weight) for weight in self.w]
      temp_b = [np.zeros_like(bias) for bias in self.b]
      beta =self.config["momentum_beta"]
      batch_size = self.config["batch_size"]
      weight_decay = self.config["weight_decay"]
      for p in range(0, self.config["epochs"]):
        for q in range(0, len(x_train_data_all), batch_size):
            x_train_data = x_train_data_all[q:q+batch_size]
            y_train_data = y_train_data_all[q:q+batch_size]
            d_w = [np.zeros_like(weight) for weight in self.w]
            d_b = [np.zeros_like(bias) for bias in self.b]
            for k in range(self.total_layers):
                temp_w[k] = beta*previous_w[k]
                temp_b[k] = beta * previous_b[k]
            self.update_parameters(previous_w, previous_b, beta)

            for j in range(0, len(x_train_data)):
                h,a = self.forward_propogate(x_train_data[j])
                d_w_temp, d_b_temp = self.back_propagation(h, a, y_train_data[j], x_train_data[j])
                for k in range(self.total_layers):
                    d_w[k] += d_w_temp[k]
                    d_b[k] += d_b_temp[k]
            for k in range(self.total_layers):
                    previous_w[k] = temp_w[k] + self.config["learning_rate"]*d_w[k]
                    self.w[k] = self.w[k] - self.config["learning_rate"]*d_w[k] -weight_decay*self.w[k]
                    previous_b[k] = temp_b[k] + self.config["learning_rate"]*d_b[k]
                    self.b[k] -= self.config["learning_rate"]*d_b[k]
        if((self.config["epochs"] == 10 and p%2==1) or self.config["epochs"] == 5):
            self.calculate_loss(x_train_data_all, y_train_data_all, x_validation_data, y_validation_data, p)


    #rmpprop gradient descent
    def rmsprop_gradient_descent(self, x_train_data_all, y_train_data_all, x_validation_data, y_validation_data):
      v_w = [np.zeros_like(weight) for weight in self.w]
      v_b = [np.zeros_like(bias) for bias in self.b]
      beta =self.config["rms_beta"]
      epsilon = 1e-4
      batch_size = self.config["batch_size"]
      weight_decay = self.config["weight_decay"]
      for p in range(0, self.config["epochs"]):
        for q in range(0, len(x_train_data_all), batch_size):
            x_train_data = x_train_data_all[q:q+batch_size]
            y_train_data = y_train_data_all[q:q+batch_size]
            d_w = [np.zeros_like(weight) for weight in self.w]
            d_b = [np.zeros_like(bias) for bias in self.b]

            for j in range(0, len(x_train_data)):
                h,a = self.forward_propogate(x_train_data[j])
                d_w_temp, d_b_temp = self.back_propagation(h, a, y_train_data[j], x_train_data[j])
                for k in range(self.total_layers):
                        d_w[k] += d_w_temp[k]
                        d_b[k] += d_b_temp[k]
            for k in range(self.total_layers):
                v_w[k] = beta*v_w[k] + (1-beta)*d_w[k]**2
                v_b[k] = beta*v_b[k] + (1-beta)*d_b[k]**2
                self.w[k] = self.w[k] - self.config["learning_rate"]*d_w[k]/(np.sqrt(v_w[k])+epsilon) - weight_decay*self.w[k]
                self.b[k] = self.b[k] - self.config["learning_rate"]*d_b[k]/(np.sqrt(v_b[k])+epsilon)
        if((self.config["epochs"] == 10 and p%2==1) or self.config["epochs"] == 5):
            self.calculate_loss(x_train_data_all, y_train_data_all, x_validation_data, y_validation_data, p)

    #adam gradient descent
    def adam_gradient_descent(self, x_train_data_all, y_train_data_all, x_validation_data, y_validation_data):
      v_w = [np.zeros_like(weight) for weight in self.w]
      v_b = [np.zeros_like(bias) for bias in self.b]
      m_w = [np.zeros_like(weight) for weight in self.w]
      m_b = [np.zeros_like(bias) for bias in self.b]
      m_w_hat = [np.zeros_like(weight) for weight in self.w]
      m_b_hat = [np.zeros_like(bias) for bias in self.b]
      v_w_hat = [np.zeros_like(weight) for weight in self.w]
      v_b_hat = [np.zeros_like(bias) for bias in self.b]
      beta1 = self.config["beta1"]
      beta2 = self.config["beta2"]
      epsilon = self.config["eps"]
      batch_size = self.config["batch_size"]
      weight_decay = self.config["weight_decay"]
      for p in range(0, self.config["epochs"]):
        for q in range(0, len(x_train_data_all), batch_size):
            x_train_data = x_train_data_all[q:q+batch_size]
            y_train_data = y_train_data_all[q:q+batch_size]
            d_w = [np.zeros_like(weight) for weight in self.w]
            d_b = [np.zeros_like(bias) for bias in self.b]

            for j in range(0, len(x_train_data)):
                h,a = self.forward_propogate(x_train_data[j])
                d_w_temp, d_b_temp = self.back_propagation(h, a, y_train_data[j], x_train_data[j])
                for k in range(self.total_layers):
                        d_w[k] += d_w_temp[k]
                        d_b[k] += d_b_temp[k]

            for k in range(self.total_layers):
                m_w[k] = beta1*m_w[k] + (1-beta1)*d_w[k]
                m_b[k] = beta1*m_b[k] + (1-beta1)*d_b[k]
                v_w[k] = beta2*v_w[k] + (1-beta2)*d_w[k]**2
                v_b[k] = beta2*v_b[k] + (1-beta2)*d_b[k]**2

                m_w_hat[k] = m_w[k]/(1-np.power(beta1, p+1))
                m_b_hat[k] = m_b[k]/(1-np.power(beta1, p+1))
                v_w_hat[k] = v_w[k]/(1-np.power(beta2, p+1))
                v_b_hat[k] = v_b[k]/(1-np.power(beta2, p+1))

                self.w[k] = self.w[k] - self.config["learning_rate"]*m_w_hat[k]/(np.sqrt(v_w_hat[k]+epsilon)) - weight_decay*self.w[k]
                self.b[k] = self.b[k] - self.config["learning_rate"]*m_b_hat[k]/(np.sqrt(v_b_hat[k]+epsilon))

        if((self.config["epochs"] == 10 and p%2==1) or self.config["epochs"] == 5):
            self.calculate_loss(x_train_data_all, y_train_data_all, x_validation_data, y_validation_data, p)

    #nadam gradient descent
    def nadam_gradient_descent(self, x_train_data_all, y_train_data_all, x_validation_data, y_validation_data):
      v_w = [np.zeros_like(weight) for weight in self.w]
      v_b = [np.zeros_like(bias) for bias in self.b]
      m_w = [np.zeros_like(weight) for weight in self.w]
      m_b = [np.zeros_like(bias) for bias in self.b]
      m_w_hat = [np.zeros_like(weight) for weight in self.w]
      m_b_hat = [np.zeros_like(bias) for bias in self.b]
      v_w_hat = [np.zeros_like(weight) for weight in self.w]
      v_b_hat = [np.zeros_like(bias) for bias in self.b]
      beta1 = self.config["beta1"]
      beta2 = self.config["beta2"]
      epsilon = self.config["eps"]
      batch_size = self.config["batch_size"]
      weight_decay = self.config["weight_decay"]
      for p in range(0, self.config["epochs"]):
        for q in range(0, len(x_train_data_all), batch_size):
            x_train_data = x_train_data_all[q:q+batch_size]
            y_train_data = y_train_data_all[q:q+batch_size]
            d_w = [np.zeros_like(weight) for weight in self.w]
            d_b = [np.zeros_like(bias) for bias in self.b]

            for j in range(0, len(x_train_data)):
                h,a = self.forward_propogate(x_train_data[j])
                d_w_temp, d_b_temp = self.back_propagation(h, a, y_train_data[j], x_train_data[j])
                for k in range(self.total_layers):
                        d_w[k] += d_w_temp[k]
                        d_b[k] += d_b_temp[k]

            for k in range(self.total_layers):
                m_w[k] = beta1*m_w[k] + (1-beta1)*d_w[k]
                m_b[k] = beta1*m_b[k] + (1-beta1)*d_b[k]
                v_w[k] = beta2*v_w[k] + (1-beta2)*d_w[k]**2
                v_b[k] = beta2*v_b[k] + (1-beta2)*d_b[k]**2

                m_w_hat[k] = m_w[k]/(1-np.power(beta1, p+1))
                m_b_hat[k] = m_b[k]/(1-np.power(beta1, p+1))
                v_w_hat[k] = v_w[k]/(1-np.power(beta2, p+1))
                v_b_hat[k] = v_b[k]/(1-np.power(beta2, p+1))

                self.w[k] = self.w[k] - (self.config["learning_rate"]/np.sqrt(v_w_hat[k]+epsilon))*(beta1*m_w_hat[k] +(1-beta1)*d_w[k]/(1-beta1**(p+1))) - weight_decay*self.w[k]
                self.b[k] = self.b[k] - (self.config["learning_rate"]/np.sqrt(v_b_hat[k]+epsilon))*(beta1*m_b_hat[k] +(1-beta1)*d_b[k]/(1-beta1**(p+1)))

        if((self.config["epochs"] == 10 and p%2==1) or self.config["epochs"] == 5):
            self.calculate_loss(x_train_data_all, y_train_data_all, x_validation_data, y_validation_data, p)

    def gradient_descent(self,x_train_data, y_train_data, x_validation_data, y_validation_data):
        if(self.config["optimizer"] == "sgd"):
            self.stochastic_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "momentum"):
            self.momentum_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "nestrov"):
            self.nestrov_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "rmsprop"):
            self.rmsprop_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "adam"):
            self.adam_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)
        elif(self.config["optimizer"] == "nadam"):
            self.nadam_gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)

    #loss calculation
    def calculate_loss(self, x_train_data, y_train_data, x_validation_data, y_validation_data, epoch=0):
      train_count = 0
      train_loss = 0
      validation_loss = 0
      validation_count = 0
      epsilon = 1e-10

      # calculating train data loss
      for i in range(len(x_train_data)):
        h,a = self.forward_propogate(x_train_data[i])
        output_class = np.argmax(h[self.total_layers-1])
        actual_class = y_train_data[i]
        if(output_class == actual_class):
          train_count+=1
        if(self.config["loss"] == "cross_entropy"):
          log_val = max(h[self.total_layers -1][actual_class, 0], epsilon)
          train_loss+=-math.log10(log_val)
        elif(self.config["loss"] == "mean_squared_error"):
          e_l = np.zeros((10, 1))
          e_l[actual_class] = 1
          train_loss+=np.sum((h[self.total_layers -1] - e_l)**2)
        
      # calculating validation data loss
      for i in range(len(x_validation_data)):
        h,a = self.forward_propogate(x_validation_data[i])
        output_class = np.argmax(h[self.total_layers-1])
        actual_class = y_validation_data[i]
        if(output_class == actual_class):
          validation_count+=1
        if(self.config["loss"] == "cross_entropy"):
          log_val = max(h[self.total_layers -1][actual_class, 0], epsilon)
          validation_loss+=-math.log10(log_val)
        elif(self.config["loss"] == "mean_squared_error"):
          e_l = np.zeros((10, 1))
          e_l[actual_class] = 1
          validation_loss+=np.sum((h[self.total_layers -1] - e_l)**2)
      train_accuracy = train_count/len(x_train_data)
      validation_accuracy = validation_count/len(x_validation_data)
      train_loss = train_loss/(len(x_train_data))
      validation_loss = validation_loss/(len(x_validation_data))
      print("Epoch: ", epoch,"train acc:",train_accuracy, "train_loss:", train_loss,"validation acc:", validation_accuracy, "validation loss:",validation_loss)
      if((self.config["epochs"] == 10 and epoch%2==1) or self.config["epochs"] == 5):
        wandb.log({"train_accuracy":train_accuracy, "train_loss":train_loss, "val_accuracy":validation_accuracy, "val_loss":validation_loss, "epoch":epoch})

    def plot_confusion_matrix(self,x_test_data, y_test_data):
      predicted_y = []
      for i in range(len(x_test_data)):
        h,a = self.forward_propogate(x_test_data[i])
        output_class = np.argmax(h[self.total_layers-1])
        predicted_y.append(output_class)
      confusion_matrix = np.zeros((10,10))
      for i in range(len(y_test_data)):
        confusion_matrix[y_test_data[i]][predicted_y[i]]+=1
      # print(confusion_matrix)
      classes = ['Ankle boot', 'T-shirt/top', 'Dress', 'Pullover', 'Sneaker', 'Sandal', 'Trouser', 'Shirt', 'Coat', 'Bag']
      plt.figure(figsize=(10, 8))
      sns.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues, fmt='.2f', xticklabels=classes, yticklabels=classes)
      plt.title("Confusion matrix")
      plt.xlabel('Predicted class')
      plt.ylabel('True class')
      plt.savefig('Confusing matrix.png') 
      wandb.log({"Confusing matrix": wandb.Image('Confusing matrix.png')})
      plt.show()

    def update_parameters(self,d_w,d_b, eta):
      for i in range(0, self.total_layers):
        self.w[i] -= eta*d_w[i]
        self.b[i] -= eta*d_b[i]

    def sigmoid(self,arr):
        # print(arr)
        clip_arr = np.clip(arr,-500, 500)
        return 1. / (1.+np.exp(-clip_arr))
    def sigmoid_derivative(self, arr):
        return self.sigmoid(arr) * (1-self.sigmoid(arr))

    def tanh(self, arr):
       clipped_arr = np.clip(arr, -100, 100)
       return np.tanh(clipped_arr)

    def tanh_derivative(self, arr):
       return 1-self.tanh(arr)**2

    def relu(self,arr):
       return np.maximum(0, arr)

    def relu_derivative(self, arr):
       return np.where(arr > 0, 1, 0)

    def softmax(self, arr):
        max_val = np.max(arr)
        exp_arr = np.exp(arr - max_val)
        return exp_arr / np.sum(exp_arr, axis=0)

    def activation(self, fun, arr):
       if(fun=="sigmoid"):
          return self.sigmoid(arr)
       elif(fun == "relu"):
          return self.relu(arr)
       elif(fun == "tanh"):
          return self.tanh(arr)

    def activation_derivative(self, fun, arr):
       if(fun=="sigmoid"):
          return self.sigmoid_derivative(arr)
       elif(fun == "relu"):
          return self.relu_derivative(arr)
       elif(fun == "tanh"):
          return self.tanh_derivative(arr)


def main():
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("-wp", "--wandb_project", default="myproject", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to be used")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("-o", "--optimizer", default="adam", choices=["sgd", "momentum", "nestrov", "rmsprop", "adam", "nadam"], help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", default="xavier", choices=["random", "xavier"], help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", default="relu", choices=["identity", "sigmoid", "tanh", "relu"], help="Activation function")
    args = parser.parse_args()

    
    config = {
        "wandb_project":args.wandb_project,
        "wandb_entity":args.wandb_entity,
        "dataset":args.dataset,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "loss": args.loss,
        "optimizer":args.optimizer,
        "learning_rate":args.learning_rate,
        "momentum_beta":args.momentum,
        "rms_beta":args.beta,
        "beta1":args.beta1,
        "beta2":args.beta2,
        "eps":args.epsilon,
        "weight_decay":args.weight_decay,
        "initialization":args.weight_init,
        "hl_size":args.hidden_size,
        "activation":args.activation,
        "hidden_layers":args.num_layers
    }

    print(config)
    wandb.login()
    loss_type = "ce" if config["loss"] == "cross_entropy" else "mse"
    run = wandb.init(project=config["wandb_project"], name = f"{config['optimizer']}_hl_{config['hidden_layers']}_hlsize_{config['hl_size']}_bs_{config['batch_size']}_ac_{config['activation']}_init_{config['initialization']}_loss_{loss_type}", config=config)

    if(config["dataset"] == "fashion_mnist"):
        [(x_total_train_data, y_total_train_data), (x_test_data, y_test_data)] = fashion_mnist.load_data()
    elif(config["dataset"] == "mnist"):
        [(x_total_train_data, y_total_train_data), (x_test_data, y_test_data)] = mnist.load_data()

    #getting training and test data
    total_train_len = len(x_total_train_data);
    train_count = int(total_train_len*.90)
    flattened_train_data = []
    #flattening the 28*28 pixel matrix
    for i in range(0, total_train_len):
        flattened_train_data.append(x_total_train_data[i].flatten()/255.0)
        # print(flattened_train_data[i])

    x_train_data = np.array(flattened_train_data[0:train_count])
    y_train_data = y_total_train_data[0:train_count]
    x_validation_data = np.array(flattened_train_data[train_count:])
    y_validation_data = y_total_train_data[train_count:]

    n_network = NeuralNetwork(784, 10, config)
    n_network.gradient_descent(x_train_data, y_train_data, x_validation_data, y_validation_data)

main()
