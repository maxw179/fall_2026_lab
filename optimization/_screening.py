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


def augmented_qr_screening_loss(matrices, data, rho, xp):
    """Minimized ridge loss from batched Householder QR, without a Gram or SVD.

    For K>P, min_f ||D-S f||^2 + rho ||f||^2 has A=[S; sqrt(rho) I]
    and b=[D; 0]. Apply the same Householder reflectors to A and b; the
    trailing entries of Q^H b give the loss without subtracting squared norms.
    For K<=P, QR of [S^H; sqrt(rho) I_K] gives R^H R=S S^H+rho I_K.
    One batched forward substitution R^H z=sqrt(rho) D gives L=||z||^2.
    This avoids taking a tiny interpolating residual by cancellation, and
    keeps the number of reflector columns at min(K,P).

    Each reflector operates on the whole frequency/candidate batch. Dividing
    all of A by a per-matrix scale preserves its column space and the minimized
    residual, hence preserves rho in the original objective. D is not used to
    choose that scale. Reflector vectors are normalized before their norms are
    formed so large raw optical units do not overflow the QR calculation.
    """
    batch, k, p = matrices.shape
    dual = k <= p
    operator = matrices.conj().swapaxes(-1, -2) if dual else matrices
    rows, columns = operator.shape[-2:]
    root = rho**.5
    scale = xp.maximum(xp.max(abs(matrices), axis=(1, 2)), root)
    dtype = xp.result_type(matrices.dtype, data.dtype, xp.float64)
    a = xp.zeros((batch, rows+columns, columns), dtype=dtype)
    a[:, :rows] = operator/scale[:, None, None]
    diagonal = xp.arange(columns)
    a[:, rows+diagonal, diagonal] = (root/scale)[:, None]
    if not dual:
        b = xp.zeros((batch, rows+columns), dtype=dtype)
        b[:, :rows] = data
    for j in range(columns):
        v = a[:, j:, j].copy()
        column_scale = xp.max(abs(v), axis=1)
        v /= xp.where(column_scale > 0, column_scale, 1)[:, None]
        norm = xp.sqrt(xp.sum(abs(v)**2, axis=1))
        first_abs = abs(v[:, 0])
        phase = v[:, 0]/xp.where(first_abs > 0, first_abs, 1)
        phase = xp.where(first_abs > 0, phase, 1)
        v[:, 0] += phase*norm
        v_norm = xp.sqrt(xp.sum(abs(v)**2, axis=1))
        v /= xp.where(v_norm > 0, v_norm, 1)[:, None]
        block = a[:, j:, j:]
        projection = xp.sum(v.conj()[:, :, None]*block, axis=1)
        block -= 2*v[:, :, None]*projection[:, None, :]
        if not dual:
            rhs = b[:, j:]
            rhs -= 2*v*xp.sum(v.conj()*rhs, axis=1)[:, None]
        a[:, j, j] = -phase*norm*column_scale
        a[:, j+1:, j] = 0
    if not dual:
        return xp.sum(abs(b[:, columns:])**2, axis=1)
    z = xp.empty((batch, columns), dtype=dtype)
    rhs = (root/scale)[:, None]*data
    for j in range(columns):
        z[:, j] = (rhs[:, j]-xp.sum(a[:, :j, j].conj()*z[:, :j], axis=1))/a[:, j, j].conj()
    return xp.sum(abs(z)**2, axis=1)
