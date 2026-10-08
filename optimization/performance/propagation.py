"""CPU microbenchmark: dense pupil propagation versus a separable chirp Z transform.

This is an investigation prototype, not a production PSF implementation.
Run: OPENBLAS_NUM_THREADS=1 python -m optimization.performance.propagation
"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.signal import CZT
from utils.psf import Microscope, Centered_Square_Grid, _convolution_grid
from utils.rw import get_bfp_grid, bfp_coord_convert


def main():
    results = []
    rng = np.random.default_rng(0)
    for size in [32, 64, 128, 256]:
        m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, size)
        kg = _convolution_grid(Centered_Square_Grid(.0065, size, 0))
        _, bx, by = get_bfp_grid(m.L_bfp, size)
        mask, _, _, sx, sy, _ = bfp_coord_convert(m.f, m.n, m.alpha, bx, by)
        x, y = kg.get_xy()
        sx, sy = sx[:, 0], sy[0]
        ax = np.exp(1j * m.k * np.outer(x, sx))
        ay = np.exp(1j * m.k * np.outer(sy, y))
        tx = CZT(size, len(x), w=np.exp(1j*m.k*(x[1]-x[0])*(sx[1]-sx[0])),
                 a=np.exp(-1j*m.k*x[0]*(sx[1]-sx[0])))
        ty = CZT(size, len(y), w=np.exp(1j*m.k*(y[1]-y[0])*(sy[1]-sy[0])),
                 a=np.exp(-1j*m.k*y[0]*(sy[1]-sy[0])))
        phase = np.exp(1j*m.k*x*sx[0])[:, None] * np.exp(1j*m.k*y*sy[0])[None, :]
        pupil = (rng.normal(size=(4, 3, size, size)) + 1j*rng.normal(size=(4, 3, size, size))) * mask
        def dense():
            return ax @ pupil @ ay
        def czt():
            return ty(tx(pupil, axis=-2), axis=-1) * phase
        ref, other = dense(), czt()
        relative_error = np.linalg.norm(ref-other)/np.linalg.norm(ref)
        assert relative_error < 1e-10, relative_error
        timing = {}
        for label, fn in [('dense', dense), ('czt', czt)]:
            fn()
            samples=[]
            for _ in range(5):
                start=perf_counter(); fn(); samples.append(perf_counter()-start)
            timing[label]=float(np.median(samples))
        row=dict(size=size, relative_field_error=float(relative_error), **timing)
        results.append(row)
        print(row, flush=True)
    Path('optimization/performance/propagation.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
