import time
import numpy as np
from network import Network
from logic_gates import Logic_gates
from logger import TrainingLog, Visualizer

def main():
    np.random.seed(42)

    hidden = [4] # Hidden layers is required to train XOR 
    outputs = 1
    quantity = 16  # Size of training batch
    runs_per_gate = 20000

    # Initializing tools
    gates_dataset = Logic_gates()
    logger = TrainingLog()
    visualizer = Visualizer(window_title="Gate Architecture Benchmark")

    start_lr = 1
    end_lr = 0.1
    lr_decay = (start_lr - end_lr) / runs_per_gate

    # Group all our gates into a list of tuples (name, function_pointer, input_size)
    gates_to_test = [
        ("and",  gates_dataset.and_gate, 2),
        ("nand", gates_dataset.nand_gate, 2),
        ("or",   gates_dataset.or_gate, 2),
        ("nor",  gates_dataset.nor_gate, 2),
        ("not",  gates_dataset.not_gate, 1),
        ("xor",  gates_dataset.xor_gate, 2),
    ]
    print("Starting Training Experiment...")
    
    # Iterate through each logical operation as a training step
    for step_idx, (gate_name, gate_function, input_dim) in enumerate(gates_to_test):
        avg_loss = 0
        current_lr = start_lr

        # Network instantiates here—clearing weights for the new gate type
        network = Network(input_dim, hidden, outputs)
        print(f"\nEvaluating: {gate_name.upper()} gate across {runs_per_gate} runs...")

        for run in range(1, runs_per_gate+1):
            current_lr = max(end_lr, start_lr - (run * lr_decay))

            # Generate dataset for this gate
            inputs, targets = gate_function(quantity)
            
            # Benchmark training run
            avg_loss += network.train(inputs, targets, current_lr)

            if run > 0 and run % 500 == 0:
                metric_key = f"{gate_name}_loss"
                
                logger.record(run, **{metric_key: avg_loss / 500})
                avg_loss = 0
                visualizer.update(logger.history)
                logger.save("training_log.npz")

        if gate_name == 'not':
            for x in [[0], [1]]:
                print(x, network.forward(x))
        else:
            for x in [[0,0],[0,1],[1,0],[1,1]]:
                print(x, network.forward(x))


    print("\nTraining completely finished!")
    visualizer.keep_open()

if __name__ == "__main__":
    main()
