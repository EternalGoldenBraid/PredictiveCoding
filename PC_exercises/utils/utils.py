def euler_step(params: dict):
    """
    Compute the next value of phi using Euler's method.
    """
    dF_dphi= lambda phi, u=params['u']: ((u - params['g'](phi))/params['sigma_u'])*params['d_g'](phi) + (params['mean_prior'] - phi)/params['sigma_prior']
    params['phi'] += params['lr'] * dF_dphi(params['phi'])
    
    return params

def neuron_step(params: dict):
    """
    Compute the next value of phi using neural dynamics:
    e_p = phi - mean_prior - v_sigma_prior * e_prior
    e_u = u - g(phi) - sigma_u * e_u
    """

    params['phi'] += params['lr'] * (-params['e_prior'] + params['e_u']*params['d_g'](params['phi']))

    params['e_prior'] += params['lr'] * (params['phi'] - params['mean_prior'] - params['sigma_prior']*params['e_prior'])
    params['e_u'] += params['lr'] * (params['u'] - params['g'](params['phi']) - params['sigma_u']*params['e_u'])
    
    return params

def neuron_step_update_params(params: dict):
    """
    Compute the next value of phi using neural dynamics:
    e_p = phi - mean_prior - v_sigma_prior * e_prior
    e_u = u - g(phi) - sigma_u * e_u
    """

    params['phi'] += params['lr'] * (-params['e_prior'] + params['e_u']*params['d_g'](params['phi']))

    params['e_prior'] += params['lr'] * (params['phi'] - params['mean_prior'] - params['sigma_prior']*params['e_prior'])
    params['e_u'] += params['lr'] * (params['u'] - params['g'](params['phi']) - params['sigma_u']*params['e_u'])
    
    params['mean_prior'] += params['lr'] * params['e_prior']
    params['sigma_prior'] += params['lr'] * 0.5*(params['e_prior']**2 - 1/params['sigma_prior'])
    params['sigma_u'] += params['lr'] * 0.5*(params['e_u']**2 - 1/params['sigma_u'])
    
    return params