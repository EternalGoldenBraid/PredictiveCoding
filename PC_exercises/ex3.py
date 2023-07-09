from typing import List, Tuple, Dict, Callable, Union, Optional, Any
import time

import torch as pt
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import lines

from PC_exercises.utils.vis_utils import print_summary, FastPlotter
from PC_exercises.utils.utils import euler_step, neuron_step, neuron_step_update_params

plt.style.use('dark_background')

def normal_pdf(x: float, mu: float, sigma: float) -> float:
    return 1.0 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def run():

    u: float = 2.0 # Observed light intensity
    sigma_u: float = 1.0 # Uncertainty of sensors

    mean_prior: float = 3.0 # Mean of the prior distribution of food size
    sigma_prior: float = 1.0
    v_prior = lambda v: normal_pdf(v, mean_prior, sigma_prior)
    
    # Food size to light intensity relationship
    g = lambda v: v**2.0
    g_inv = lambda u: np.sqrt(u)
    d_g = lambda v: 2.0 * v

    # Likelihood of observing the observed light intensity given a food size
    u_likelihood = lambda v, u=u, sigma_u=sigma_u: normal_pdf(u, g(v), sigma_u)
    
    v_bounds: Tuple[float, float] = (0.01, 5.0)
    
    # The integral of the numerator of Bayes' rule
    p_u: float = sp.integrate.quad(lambda v: u_likelihood(v) * v_prior(v), *(-100.0, 100.0))[0]
    
    # The posterior distribution of food size
    v_posterior = lambda v: u_likelihood(v) * v_prior(v) / p_u
    
    # Check that posterior is a valid probability distribution
    cdf = sp.integrate.quad(v_posterior, *(-100.0, 100.0))[0]
    assert np.isclose(cdf, 1.0)


    # Iterative gradient ascent
    
    params: Dict[str, Any] = {
        'euler': {
            'phi': mean_prior, 'u': u,
            'sigma_prior': sigma_prior, 'sigma_u': sigma_u,
            'mean_prior': mean_prior,
            'e_prior': None, 'e_u': None,
            'g': g, 'g_inv': g_inv, 'd_g': d_g,
            'lr': 0.01, 'line': None, 'color': 'C0',
        },
        'neural': {
            'phi': mean_prior, 'u': u,
            'sigma_prior': sigma_prior, 'sigma_u': sigma_u,
            'mean_prior': mean_prior,
            'e_prior': 0.0, 'e_u': 0.0,
            'g': g, 'g_inv': g_inv, 'd_g': d_g,
            'lr': 0.01, 'line': None, 'color': 'C1',
        },
        'neural_updated_params': {
            'phi': mean_prior, 'u': u,
            'sigma_prior': sigma_prior, 'sigma_u': sigma_u,
            'mean_prior': mean_prior,
            'e_prior': 0.0, 'e_u': 0.0,
            'g': g, 'g_inv': g_inv, 'd_g': d_g,
            'lr': 0.01, 'line': None, 'color': 'C2',
        },
    }
    
    euler_params: Dict[str, Any] = params['euler']
    neural_static_params: Dict[str, Any] = params['neural']
    neural_updated_params: Dict[str, Any] = params['neural_updated_params']
    
    plotter = FastPlotter()
    plotter.plot_distributions(v_prior, v_posterior, v_bounds)
    plotter.save_background_()

    plotter.ax[0].set_xlabel('v')
    plotter.ax[0].set_ylabel('p(v)')
    # plotter.fig.set_size_inches(12, 4)
    
    plotter.ax[1].set_xlabel('Iteration')
    plotter.ax[1].set_ylabel('phi')
    iters = 900
    plotter.ax[1].set_xlim(0, iters)
    plotter.ax[1].set_ylim(0, 10)

    frame_time: float = 1.0 / 30.0
    for i in range(iters):
        start_time = time.time()
        euler_params = euler_step(params=euler_params)
        neural_static_params = neuron_step(params=neural_static_params)
        neural_updated_params = neuron_step_update_params(params=neural_updated_params)

        # Update plots
        plotter.update(params, i)
        # plotter.fig.canvas.draw()
        # plotter.fig.canvas.flush_events()

        end_time = time.time()
        time.sleep(max(0.0, frame_time - (end_time - start_time)))

    print("Most probable food size given the sensory observation", u, ":") 
    print_summary(params=params['euler'])
    print_summary(params=params['neural'])
    print_summary(params=params['neural_updated_params'])
    print("True:", g_inv(u))
