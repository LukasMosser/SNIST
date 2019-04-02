import torch

def sample_model_N_times(dVel0, dVel, v0, dv_l_const, v_max, n_layers, dz, dx, ny, nz, N):
    """Sample N earth models according to the generative model defined in Roeth and Tarantola 1994
    
    Arguments:
        dVel0 {torch.distribution} -- Pytorch Distribution for the random velocity contribution of the initial layer velocity
        dVel {torch.distribution} -- Pytorch Distribution for the random velocity contribution of the next layer
        v0 {float} -- Initial Velocity
        dv_l_const {float} -- Constant velocity contribution per next layer
        v_max {float} -- Largest possible velocity in the model
        n_layers {int} -- Number of layers to generate
        dz {float} -- Depth increase [m]
        dx {float} -- Horizontal increase [m]
        ny {int} -- Number of offset gridblocks
        nz {int} -- Number of gridblocks in depth
        N {int} -- Number of earth models to generate
    
    Returns:
        models_th -- 2D Velocity Models
        velocities_th -- 1D Velocities per layer
    """

    models = []
    velocities = []
    for i in range(N):
        model_true = torch.ones(nz, ny)

        vl = v0+dVel0.sample()
        vel_temp = []
        for i in range(0, n_layers):
            vel_temp.append(vl)
            model_true[dz*i:dz*(i+1), :] = vl
            vl =(vl+dv_l_const)+dVel.sample()
            if vl >= v_max:
                vl = torch.ones(1)*v_max
                vl = vl[0]
        velocities.append(torch.stack(vel_temp, 0))
        models.append(model_true)
        
    models_th = torch.stack(models, 0)
    velocities_th = torch.stack(velocities, 0) 
    return models_th, velocities_th
