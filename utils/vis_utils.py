import matplotlib.pyplot as plt

class LossVisualizer:
    """
    Realtime visualizer for loss of a neural network model. 
    """

    def __init__(self, dark_mode=True):
        
        self.fig, self.ax = plt.subplots()

        if not dark_mode:
            self.ax.set_facecolor('white')
            self.ax.set_title("Real-time Loss")
            self.ax.set_xlabel("Iterations")
            self.ax.set_ylabel("Loss")
        else:
            self.fig.patch.set_facecolor('black')  # Change the background color to black
            self.ax.set_facecolor('black')  # Change the background color to black

            self.ax.set_title("Real-time Loss", color='white')  # Change title color to white
            self.ax.set_xlabel("Iterations", color='white')  # Change x-label color to white
            self.ax.set_ylabel("Loss", color='white')  # Change y-label color to white

            self.ax.spines['bottom'].set_color('white')  # Change x-axis color to white
            self.ax.spines['top'].set_color('white')  # Change x-axis color to white
            self.ax.spines['left'].set_color('white')  # Change y-axis color to white
            self.ax.spines['right'].set_color('white')  # Change y-axis color to white

            self.ax.xaxis.label.set_color('white')  # Change x-axis label color to white
            self.ax.yaxis.label.set_color('white')  # Change y-axis label color to white

            self.ax.tick_params(axis='x', colors='white')  # Change x-axis ticks color to white
            self.ax.tick_params(axis='y', colors='white')  # Change y-axis ticks color to white

        self.loss_line, = self.ax.plot([], [], 'r-')
        self.losses = []

        plt.ion()
        plt.show()

    def update(self, loss: float):
        """
        This function is called at every time step to update the visualization.
        It takes a loss value and appends it to the loss history.
        """

        self.losses.append(loss)
        self.loss_line.set_xdata(range(len(self.losses)))
        self.loss_line.set_ydata(self.losses)

        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
