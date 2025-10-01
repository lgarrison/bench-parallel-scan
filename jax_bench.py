import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def jax_matrix_multiply(x, y):
    return jnp.matmul(x, y)


@jax.jit
def scan_fn(mats):
    return jax.lax.associative_scan(jax_matrix_multiply, mats)


def benchmark(matrices):
    """Run associative scan on an array of n 2x2 matrices"""
    result = scan_fn(matrices)
    jax.block_until_ready(result)
    return result
