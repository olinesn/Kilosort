"""Checks that Kilosort's stack runs natively on Windows ARM64 with one OpenMP runtime.

The faiss call after torch's threads are running is the step that hangs on
Apple Silicon when torch, faiss and scikit-learn each load their own libomp
(Kilosort issue #950).
"""
import os
import platform
import sys
import time

print('python', sys.version)
print('machine', platform.machine(), '|', platform.platform())
assert platform.machine().upper() == 'ARM64', 'not a native ARM64 Python'

import numpy, scipy, sklearn, numba, faiss, torch, kilosort, psutil  # noqa: E401
for mod in (numpy, scipy, sklearn, numba, faiss, torch, kilosort):
    print(f'{mod.__name__:10s} {mod.__version__}')
print('torch threads', torch.get_num_threads(), '| cuda', torch.cuda.is_available())
print(torch.__config__.parallel_info())

x = torch.randn(3000, 3000)
for _ in range(3):
    x @ x
print('torch matmul OK')

from kilosort import clustering_qr
Xd = numpy.random.randn(20000, 60).astype('float32')
t = time.time()
kn, M = clustering_qr.neigh_mat(Xd, nskip=1, n_neigh=10)
print(f'faiss neighbour search after torch OK in {time.time() - t:.2f} s', kn.shape)

from sklearn.cluster import KMeans
KMeans(8, n_init=1).fit(Xd[:5000])
print('sklearn KMeans OK')

from numba import njit, prange


@njit(parallel=True)
def total(a):
    s = 0.0
    for i in prange(a.size):
        s += a[i]
    return s


assert total(numpy.ones(10**6)) == 10**6
print('numba parallel OK, threading layer:', numba.threading_layer())

names = ('omp', 'vcomp', 'openblas', 'armpl')
proc = psutil.Process()
if not hasattr(proc, 'memory_maps'):  # macOS
    sys.exit(0)
loaded = sorted({m.path for m in proc.memory_maps()
                 if any(k in os.path.basename(m.path).lower() for k in names)})
print('OpenMP / BLAS libraries loaded:')
for path in loaded:
    print('  ', path)
vcomp = [p for p in loaded if 'vcomp' in os.path.basename(p).lower()
         or 'omp' in os.path.basename(p).lower()]
print('OpenMP runtimes loaded:', len(vcomp))
