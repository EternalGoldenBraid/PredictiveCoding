from typing import Dict, Any
import matplotlib.pyplot as plt

import numpy as np

def print_summary(params: dict):
    print("Final states of updated parameters:")
    print(f"phi: {params['phi']}")
    print(f"e_prior: {params['e_prior']}")
    print(f"e_u: {params['e_u']}")
    print(f"mean_prior: {params['mean_prior']}")
    print(f"sigma_prior: {params['sigma_prior']}")
    print(f"sigma_u: {params['sigma_u']}")
    
    
class FastPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(2, 1)

        self.ln1 = dict()
        self.ln2 = dict()

    def save_background_(self):
        self.bg1 = self.fig.canvas.copy_from_bbox(self.ax[0].bbox) 
        self.bg2 = self.fig.canvas.copy_from_bbox(self.ax[1].bbox)
        
    def plot_distributions(self, v_prior, v_posterior, v_bounds):
        vs = np.linspace(*v_bounds, 1000)
        self.ax[0].plot(vs, v_prior(vs), label='Prior', color='C0')
        self.ax[0].plot(vs, v_posterior(vs), label='Posterior', color='C2')
        self.ax[0].legend()
        self.ax[1].legend()
        
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, params, i):
        self.fig.canvas.restore_region(self.bg1)
        self.fig.canvas.restore_region(self.bg2)

        for label, param in params.items():
            if label not in self.ln1:
                self.ln1[label], = self.ax[0].plot([], [], param['color'], linestyle='--', label=label)
                self.ln2[label], = self.ax[1].plot([i], [param['phi']], param['color'], linestyle='--', label=label)
            else:
                xdata, ydata = self.ln2[label].get_data() # TODO: this and append is slow
                self.ln2[label].set_data(np.append(xdata, i), np.append(ydata, param['phi']))

            self.ln1[label].set_xdata([param['phi'], param['phi']])
            self.ln1[label].set_ydata([0, 1])

            self.ax[0].draw_artist(self.ln1[label])
            self.ax[1].draw_artist(self.ln2[label])

        self.ax[0].legend() # TODO Not doing anything?
        self.ax[1].legend()

        self.fig.canvas.blit(self.ax[0].bbox)
        self.fig.canvas.blit(self.ax[1].bbox)
