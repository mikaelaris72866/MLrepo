"""Small timing helper shared by the benchmarks: warm-up, then median of repeated calls."""
import statistics
import time


def bench(fn, runs, warmup=3):
    """Return (median, q1, q3) of `runs` timed calls of fn(), in seconds, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    q = statistics.quantiles(ts, n=4) if len(ts) >= 4 else (ts[0], ts[0], ts[-1])
    return statistics.median(ts), q[0], q[2]
