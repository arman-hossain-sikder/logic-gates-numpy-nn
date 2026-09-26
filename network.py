import numpy as np

class Network:
    def __init__(self, input_size, hidden_layers, output_size):
        self.architecture = [input_size] + hidden_layers + [output_size]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(self.architecture) - 1):
            inputs_to_layer = self.architecture[i]
            outputs_from_layer = self.architecture[i+1]
            
            w = np.random.uniform(-1, 1, (inputs_to_layer, outputs_from_layer))
            b = np.random.uniform(-1, 1, (1, outputs_from_layer))
            
            self.weights.append(w)
            self.biases.append(b)


    def forward(self, inputs):   
        current_activation = np.array(inputs).reshape(1, -1)

        for i in range(len(self.weights) - 1):
            pre_activation = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = np.where(pre_activation < 0, 0.1 * pre_activation, pre_activation)
            
        final_pav = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        prob = (1 / (1 + np.exp(-final_pav)))

        return prob


    def train(self, inputs, answer, learning_rate=0.1):
        inputs = np.array(inputs).reshape(-1, self.architecture[0])
        batch_size = len(inputs)
        
        activations = [inputs]
        pre_activations = [] 
        
        current_activation = inputs
        for i in range(len(self.weights) - 1):
            pre_activation = np.dot(current_activation, self.weights[i]) + self.biases[i]
            pre_activations.append(pre_activation)

            current_activation = np.where(pre_activation < 0, 0.1 * pre_activation, pre_activation)
            activations.append(current_activation)
            
        final_pav = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        pre_activations.append(final_pav)
        
        probability = 1 / (1 + np.exp(-final_pav))
        answer = np.array(answer).reshape(-1, 1)

        error = (answer - probability) * (probability * (1 - probability))
        current_gradient = error
        

        for i in reversed(range(len(self.weights))):
            layer_input = activations[i]

            weight_update = np.dot(layer_input.T, current_gradient) / batch_size
            bias_update = np.mean(current_gradient, axis=0, keepdims=True)

            self.weights[i] += weight_update * learning_rate
            self.biases[i]  += bias_update * learning_rate
            
            if i > 0:
                slope = np.where(pre_activations[i-1] < 0, 0.1, 1)
                current_gradient = np.dot(current_gradient, self.weights[i].T) * slope

        return float(np.mean(np.abs(error)))
        