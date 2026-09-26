import numpy as np
import matplotlib.pyplot as plt

class TrainingLog:
    def __init__(self):
        self.history = {}

    def record(self, step, **metrics):
        """Record one row of metrics at a given step."""
        self.history.setdefault("step", []).append(step)
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)

    def save(self, path="training_log.npz"):
        # Convert lists to NumPy arrays securely
        np.savez_compressed(path, **{k: np.array(v) for k, v in self.history.items()})

    def load(self, path="training_log.npz"):
        with np.load(path, allow_pickle=True) as data:
            self.history = {k: data[k].tolist() for k in data.files}
        return self.history


class Visualizer:
    def __init__(self, window_title="Monitor"):
        plt.ion() 
        self.window_title = window_title
        self.fig = None
        self.ax = None
        self.lines = {}
        # Pre-defined nice colors for the 5 different gates
        self.colors = {'and': '#E53935', 'nand': '#43A047', 'or': '#1E88E5', 'nor': '#FB8C00', 'xor': '#8E24AA'}
        
    def _initialize_plots(self):
        """Builds one unified graph canvas for all loss curves."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title(self.window_title)
        
        self.ax.set_title("Logic Gate Convergence")
        self.ax.set_xlabel("Training Steps")
        self.ax.set_ylabel("Mean gradient magnitude")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

    def update(self, history):
        """Safely updates lines dynamically as new gates begin training."""
        if "step" not in history or len(history["step"]) == 0:
            return

        # Check if the user manually closed the window to avoid errors
        if self.fig is not None and not plt.fignum_exists(self.fig.number):
            return
            
        if self.fig is None:
            self._initialize_plots()

        steps = history["step"]
        metric_names = [k for k in history.keys() if k != "step"]
        
        for name in metric_names:
            # Dynamically create the line the first time the gate appears in history
            if name not in self.lines:
                gate_prefix = name.split('_')[0]
                color = self.colors.get(gate_prefix, '#757575')
                self.lines[name], = self.ax.plot([], [], color=color, linewidth=2, label=gate_prefix.upper())
                self.ax.legend(loc="upper right")

            # Fix for sequential steps: map matching metrics cleanly
            # We must truncate the 'steps' array to match how many items this specific metric has!
            y_data = history[name]
            x_data = steps[:len(y_data)]
            
            self.lines[name].set_data(x_data, y_data)
                
        # Re-scale graph frame borders automatically
        self.ax.relim()
        self.ax.autoscale_view()
                
        # Flush frame updates smoothly
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def keep_open(self):
        """Locks the UI window open at script end."""
        plt.ioff()
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.show()
