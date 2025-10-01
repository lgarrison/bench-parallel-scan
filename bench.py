import timeit

import click
import cuda.cccl.parallel.experimental as parallel
import cupy as cp
import numba as nb
import numpy as np
from astropy.table import Table

dtype = np.float64


@parallel.gpu_struct
class Matrix2x2:
    a11: dtype
    a12: dtype
    a21: dtype
    a22: dtype


def cupy_matrix_multiply(x: Matrix2x2, y: Matrix2x2) -> Matrix2x2:
    return Matrix2x2(
        x.a11 * y.a11 + x.a12 * y.a21,
        x.a11 * y.a12 + x.a12 * y.a22,
        x.a21 * y.a11 + x.a22 * y.a21,
        x.a21 * y.a12 + x.a22 * y.a22,
    )


@nb.njit(parallel=True, fastmath=True)
def numba_inclusive_cpu_scan(array, op):
    """Inclusive parallel scan on the CPU using Numba with a custom op"""
    n = len(array)
    element_shape = array[0].shape

    # Step 1: Compute the prefix sums for each block
    nthread = nb.get_num_threads()
    num_blocks = nthread
    block_size = (n + num_blocks - 1) // num_blocks
    block_sums = np.empty((num_blocks,) + element_shape, dtype=array.dtype)

    for b in nb.prange(num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        for i in range(start + 1, end):
            op(array[i-1], array[i], array[i])
        block_sums[b] = array[end - 1]

    # Step 2: Compute the prefix sums of the block sums
    for b in range(1, num_blocks):
        op(block_sums[b - 1], block_sums[b], block_sums[b])

    # Step 3: Add the block sums to each element in the blocks
    for b in nb.prange(1, num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        add_value = block_sums[b - 1]
        for i in range(start, end):
            op(add_value, array[i], array[i])

    return array


@nb.njit
def numba_op(x, y, out):
    a11 = x[0, 0] * y[0, 0] + x[0, 1] * y[1, 0]
    a12 = x[0, 0] * y[0, 1] + x[0, 1] * y[1, 1]
    a21 = x[1, 0] * y[0, 0] + x[1, 1] * y[1, 0]
    a22 = x[1, 0] * y[0, 1] + x[1, 1] * y[1, 1]
    out[0, 0] = a11
    out[0, 1] = a12
    out[1, 0] = a21
    out[1, 1] = a22


def create_matrix_array(length):
    rng = np.random.default_rng(42)
    return rng.random(size=(length, 2, 2), dtype=dtype)


def benchmark_thrust(scanner, d_input, d_output, d_temp, length, h_init):
    """Run associative scan on an array of n 2x2 matrices using Thrust"""
    scanner(d_temp, d_input, d_output, length, h_init)
    cp.cuda.Stream.null.synchronize()
    return d_output


def custom_autorange(timer, min_time=0.1):
    """Custom autorange function for timeit that accepts a min_time parameter"""
    number = 1
    total_time = 0

    while True:
        # Measure time for current number of iterations
        time_taken = timer.timeit(number)
        total_time += time_taken

        # If we've exceeded min_time, we're done
        if total_time >= min_time:
            return number, total_time

        # Otherwise, increase number of iterations
        # Using a doubling strategy for efficiency
        number *= 2


# Benchmark for different array lengths
def bench_one_jax(matrices, get_result=False):
    import jax_bench

    jax_matrices = jax_bench.jnp.asarray(matrices)
    warm_res = jax_bench.benchmark(jax_matrices)
    if get_result:
        return warm_res
    timer = timeit.Timer(lambda: jax_bench.benchmark(jax_matrices))
    number, time_taken = custom_autorange(timer, min_time=0.1)
    return time_taken / number


def bench_one_thrust(matrices, get_result=False):
    matrices = cp.asarray(matrices)
    N = len(matrices)

    h_init = Matrix2x2(1.0, 0.0, 0.0, 1.0)
    d_input = cp.asarray(matrices.reshape(N, -1)).view(Matrix2x2.dtype).reshape(-1)
    d_output = cp.empty_like(d_input)
    thrust_scanner = parallel.algorithms.make_inclusive_scan(
        d_input, d_output, cupy_matrix_multiply, h_init
    )
    temp_size = thrust_scanner(None, d_input, d_output, len(d_input), h_init)
    d_temp = cp.empty(temp_size, dtype=np.uint8)
    warm_res = benchmark_thrust(thrust_scanner, d_input, d_output, d_temp, N, h_init)
    if get_result:
        return warm_res.view(dtype).reshape(N, 2, 2).get()
    timer = timeit.Timer(
        lambda: benchmark_thrust(thrust_scanner, d_input, d_output, d_temp, N, h_init)
    )
    number, time_taken = custom_autorange(timer, min_time=0.1)
    return time_taken / number


def bench_one_numba_cpu(matrices, get_result=False):
    warm_res = numba_inclusive_cpu_scan(matrices, numba_op)
    if get_result:
        return warm_res
    timer = timeit.Timer(lambda: numba_inclusive_cpu_scan(matrices, numba_op))
    number, time_taken = custom_autorange(timer, min_time=0.1)
    return time_taken / number


def do_compare(method, N=10**6):
    for m in method_dispatch:
        if m == method:
            continue
        print(f"Comparing {method} against {m}")
        bench_func_1 = method_dispatch[method]
        bench_func_2 = method_dispatch[m]

        matrices = create_matrix_array(N)
        result_1 = bench_func_1(matrices, get_result=True)
        matrices = create_matrix_array(N)
        result_2 = bench_func_2(matrices, get_result=True)

        np.testing.assert_allclose(result_1, result_2)


method_dispatch = {
    "jax": bench_one_jax,
    "thrust": bench_one_thrust,
    "numba_cpu": bench_one_numba_cpu,
}


@click.command()
@click.option(
    "-m",
    "--method",
    type=click.Choice(["jax", "thrust", "numba_cpu"], case_sensitive=False),
    default="thrust",
    help="Implementation to benchmark",
)
@click.option(
    "-l",
    "--label",
    type=str,
    default=None,
    help="Label for the benchmark results",
)
@click.option(
    "-c",
    "--compare",
    is_flag=True,
    help="Instead of benchmarks, check the result of the given method against the others",
)
def main(method="jax", label=None, compare=False):
    results = []
    implementation = method
    if not label:
        label = implementation

    if compare:
        do_compare(method)
        return

    bench_func = method_dispatch[method]

    for exp in range(4, 9):
        N = 10**exp
        matrices = create_matrix_array(N)
        time_per_run = bench_func(matrices)
        results.append({"method": label, "N": N, "time": time_per_run})
        print(f"{label} N=10^{exp}: {time_per_run * 1e3:9.3f} ms")

    # Convert results to Astropy Table
    results_table = Table(results)

    # Write results to ECSV
    ecsv_filename = f"{label}_benchmark_results.ecsv"
    results_table.write(ecsv_filename, format="ascii.ecsv", overwrite=True)

    print(f"Results saved to {ecsv_filename}")


if __name__ == "__main__":
    main()
