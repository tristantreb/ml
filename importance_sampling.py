import jax
import jax.numpy as jnp
from jax import random, vmap, jit

# 1. Define our distributions and function
def target_log_pdf(x):
    # p(x): Narrow Gaussian centered at 2.0
    return -0.5 * jnp.sum(((x - 2.0) / 0.5)**2)

def proposal_log_pdf(x):
    # q(x): Wide Gaussian centered at 0.0 (easier to sample)
    return -0.5 * jnp.sum((x / 2.0)**2)

def f(x):
    # The function we want the expectation of
    return jnp.sum(x**2)

# 2. Vectorized Importance Sampling logic
@jit
def importance_sampling_estimate(key, num_samples=1000):
    # Sample from the Proposal q(x)
    # Note: In a real scenario, q is chosen because it's easy to sample.
    samples = random.normal(key, (num_samples, 1)) * 2.0
    
    # Calculate log weights: log(p/q) = log(p) - log(q)
    # We do this in log-space for numerical stability!
    log_p = vmap(target_log_pdf)(samples)
    log_q = vmap(proposal_log_pdf)(samples)
    log_w = log_p - log_q
    
    # Convert back to weights and normalize
    weights = jnp.exp(log_w)
    weights = weights / jnp.sum(weights)
    
    # Weighted expectation: E[f(x)] ≈ Σ w_i * f(x_i)
    fx_values = vmap(f)(samples)
    return jnp.sum(weights * fx_values)

def main():
    key = random.PRNGKey(42)
    estimate = importance_sampling_estimate(key)
    print(f"Estimated Expectation: {estimate:.4f}")

if __name__ == "__main__":
    main()