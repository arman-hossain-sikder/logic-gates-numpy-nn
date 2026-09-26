from network import Network
import numpy as np


class Logic_gates:
    def __init__(self):
        pass

    def and_gate(self, quantity):
        inputs = np.random.randint(0, 2, size=(quantity, 2))
        outputs = (inputs[:, 0] & inputs[:, 1]).reshape(-1, 1)
        return inputs, outputs

    def nand_gate(self, quantity):
        inputs = np.random.randint(0, 2, size=(quantity, 2))
        outputs = 1 - (inputs[:, 0] & inputs[:, 1]).reshape(-1, 1)
        return inputs, outputs

    def or_gate(self, quantity):
        inputs = np.random.randint(0, 2, size=(quantity, 2))
        outputs = (inputs[:, 0] | inputs[:, 1]).reshape(-1, 1)
        return inputs, outputs

    def nor_gate(self, quantity):
        inputs = np.random.randint(0, 2, size=(quantity, 2))
        outputs = 1 - (inputs[:, 0] | inputs[:, 1]).reshape(-1, 1)
        return inputs, outputs

    def not_gate(self, quantity):
        inputs = np.random.randint(0, 2, size=(quantity, 1))
        outputs = 1 - (inputs[:, 0]).reshape(-1, 1)
        return inputs, outputs

    def xor_gate(self, quantity):
        inputs = np.random.randint(0, 2, size=(quantity, 2))
        outputs = (inputs[:, 0] ^ inputs[:, 1]).reshape(-1, 1)
        return inputs, outputs


if __name__ == "__main__":
    gates = Logic_gates()
    gates.not_gate(100)