"""CUE end/cut convention check (numerics-review follow-up; reproduces the review's numbers).

cue.py cuts each circle at theta_1 and treats F = int Z_U as a path; analyze.barcode drops bars
born at the two path endpoints, so lobe 0 and the wrap-around lobe N-1 can never give short bars.
This script measures how atypical those two positions are:
  - the wrap-around gap (it contains angle 0) is length-biased;
  - the share of short lobes (|area| below x * median) sitting at the two excluded positions,
    against the naive 2/N;
  - Hill exponents for all lobes against interior lobes only (exponent sensitivity).
Reads results/raw/cue12.npz and cue128.npz; stdout saved as results/stats/cue_endpoint_check.out.
"""
import numpy as np

from stats_paths import raw_path


def hill(x, c):
    m = x < c
    return m.sum() / np.sum(np.log(c / x[m])), int(m.sum())


for N in (12, 128):
    d = np.load(raw_path(f"cue{N}.npz"))
    area, gap = d["area"], d["gap"]
    pos = np.arange(area.size) % N
    A = np.abs(area) / np.median(np.abs(area))
    inter = (pos != 0) & (pos != N - 1)
    print(f"CUE N={N}: wrap-around gap mean {gap[pos == N - 1].mean():.3f}; "
          f"P(gap<0.2): wrap {np.mean(gap[pos == N - 1] < 0.2):.4f}, interior {np.mean(gap[inter] < 0.2):.4f}, "
          f"lobe 0 {np.mean(gap[pos == 0] < 0.2):.4f}, lobe N-2 {np.mean(gap[pos == N - 2] < 0.2):.4f}")
    for c in (0.03, 0.01, 0.003):
        s_all, n_all = hill(A, c)
        s_int, n_int = hill(A[inter], c)
        print(f"   lobes below {c:g}*median: share at the two excluded positions {np.mean(~inter[A < c]):.4f} "
              f"(naive 2/N = {2 / N:.4f}); lobe Hill all {s_all:.3f} (n={n_all}) vs interior {s_int:.3f} (n={n_int})")
    for c in (0.4, 0.2):
        print(f"   gap Hill cut {c:g}: all {hill(gap, c)[0]:.3f} vs interior {hill(gap[inter], c)[0]:.3f}")
