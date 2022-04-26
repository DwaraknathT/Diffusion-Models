import torch
import numpy as np
from torchdiffeq import odeint


def prior_likelihood(z):
    """The likelihood of a Gaussian distribution with mean zero and
    standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
    return logps


def get_loglikelihood(
    args,
    sde,
    score_model,
    data,
    noise_type="gaussian",
    is_train=True,
):
    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    device = data.device
    shape = data.shape

    if noise_type == "gaussian":
        epsilon = lambda x: torch.randn_like(x)
    elif noise_type == "rademacher":
        epsilon = lambda x: torch.randint_like(x, low=0, high=2).float() * 2 - 1.0
    else:
        raise NotImplementedError(f"Hutchinson type {noise_type} unknown.")

    nfe_counter = 0

    def ode_func(t, state):
        nonlocal nfe_counter
        nfe_counter = nfe_counter + 1

        x, _ = state
        noise = epsilon(x)
        t = torch.max(t, torch.tensor(args.train_time_eps))
        time_steps = torch.ones((shape[0],), device=device) * t

        def drift_fn(input):
            drift = sde.drift(input, time_steps)
            g2 = torch.square(sde.diffusion(input, time_steps))

            score = score_model(input, time_steps)
            dx_dt = drift - 0.5 * g2 * score
            return dx_dt

        dx_dt, vjp = torch.autograd.functional.vjp(
            drift_fn, x, noise, create_graph=is_train
        )
        trJ = torch.sum(vjp * noise, dim=[1, 2, 3])
        dlogp_x_dt = trJ.view(x.shape[0], 1)

        return (dx_dt, dlogp_x_dt)

    init = (data, torch.zeros((shape[0], 1), device=device))

    # Black-box ODE solver
    (z_final, logp_final) = odeint(
        ode_func,
        init,
        torch.tensor([args.train_time_eps, 1.0], device=device),
        atol=args.ode_solver_tol,
        rtol=args.ode_solver_tol,
        method="rk4",
        options=dict(step_size=0.1),
    )

    z, delta_logp = z_final[-1], logp_final[-1]
    prior_logp = prior_likelihood(z)
    loglikelihood = prior_logp + delta_logp

    return loglikelihood, nfe_counter
