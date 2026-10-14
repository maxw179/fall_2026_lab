"""Minimized regularized loss without retained object or Hessian factors."""
from __future__ import annotations


def linear_screening_loss(matrices, data, rho, xp):
    """Guarded fast path: solve the smaller regularized primal/dual system.

    T=S/sqrt(rho). For K<=P, y=(I+T T^H)^-1 D is the residual and L=D^H y.
    For K>P, x=(I+T^H T)^-1 T^H D is the scaled object, and
    L=||D-T x||^2+||x||^2. Both are exactly the original minimized loss.
    The caller must guard Gram-matrix conditioning; weak rho uses SVD instead.
    Explicit singleton RHS dimensions work with both NumPy 2 and CuPy 14.
    """
    rows, columns = matrices.shape[-2:]
    t = matrices / rho**.5
    adjoint = t.conj().swapaxes(-1, -2)
    if rows <= columns:
        gram = t @ adjoint
        diagonal = xp.arange(rows)
        gram[:, diagonal, diagonal] += 1
        y = xp.linalg.solve(gram, data[..., None])[..., 0]
        return xp.real(xp.sum(data.conj() * y, axis=1))
    gram = adjoint @ t
    diagonal = xp.arange(columns)
    gram[:, diagonal, diagonal] += 1
    x = xp.linalg.solve(gram, adjoint @ data[..., None])[..., 0]
    residual = data - (t @ x[..., None])[..., 0]
    return xp.sum(abs(residual)**2, axis=1) + xp.sum(abs(x)**2, axis=1)


def spectral_screening_loss(matrices, data, rho, svd, xp):
    """Stable SVD fallback; includes all left-null-space residual components."""
    rows, columns = matrices.shape[-2:]
    u, singular, _ = svd(matrices, full_matrices=rows > columns)
    projected = (u.conj().swapaxes(-1, -2) @ data[..., None])[..., 0]
    rank = singular.shape[1]
    value = xp.sum(rho / (singular**2 + rho) * abs(projected[:, :rank])**2, axis=1)
    if rows > rank:
        value += xp.sum(abs(projected[:, rank:])**2, axis=1)
    return value
