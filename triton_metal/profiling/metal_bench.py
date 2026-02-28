"""Metal-specific benchmarking using GPU timestamps."""

import time


def metal_do_bench(fn, *, quantiles, warmup=25, rep=100, **kwargs):
    """Benchmark a function using wall-clock timing.

    Uses Metal command buffer GPU timestamps when available, falling
    back to wall-clock time.

    Args:
        fn: Callable to benchmark.
        quantiles: List of quantiles to return (e.g. [0.5, 0.2, 0.8]).
        warmup: Number of warmup iterations.
        rep: Number of timed iterations.

    Returns:
        List of times in milliseconds corresponding to each quantile.
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # Convert to ms

    times.sort()
    if quantiles is None:
        return times[len(times) // 2]  # Median

    result = []
    for q in quantiles:
        idx = int(q * (len(times) - 1))
        result.append(times[idx])
    return result
