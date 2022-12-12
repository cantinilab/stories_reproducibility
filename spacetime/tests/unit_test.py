from spacetime import data
from jax import random

def test_euler_maruyama():
    # Define the hyperparameters.
    key = random.PRNGKey(42)
    n, d = 25, 2
    dt, n_steps = 0.06, 6
    sd = 0.5
    u0 = random.normal(key, (n, d))

    # Generate an iterator for the populations.
    populations = data.euler_maruyama(
        potential=data.styblinski_potential,
        u0=u0,
        dt=dt,
        sd=sd,
        n_steps=n_steps,
        key=key,
    )

    # Iterate over the populations.
    for p in populations:
        pass
