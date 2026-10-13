"""QR reduction for stacks of rectangular SVDs, without forming a Gram matrix.

CuPy uses a genuinely batched SVD only when both matrix dimensions are <=32.
Householder operations here run across the entire frequency batch, so a 24x40
operator reduces to a 24x24 SVD without a cuSOLVER call for each frequency.
NumPy execution is supported for independent numerical validation.
"""
from __future__ import annotations


def _householder_qr(matrices, full_matrices, xp):
    """Return Q and square R for a stack of tall real/complex matrices.

    Unlike calling a single-matrix QR in a Python loop, each operation processes
    every frequency. Complete Q retains a stable explicit null-space basis for
    the reduced Hessian, even at small regularization. Column scaling prevents
    overflow/underflow in the reflector norms; zero columns use the identity.
    """
    batch, rows, columns = matrices.shape
    if rows < columns:
        raise ValueError('Householder QR requires rows >= columns.')
    r = matrices.copy()
    reflectors = []
    for j in range(columns):
        v = r[:, j:, j].copy()
        scale = xp.max(abs(v), axis=1)
        v /= xp.where(scale > 0, scale, 1)[:, None]
        norm = xp.sqrt(xp.sum(abs(v)**2, axis=1))
        first_abs = abs(v[:, 0])
        phase = v[:, 0] / xp.where(first_abs > 0, first_abs, 1)
        phase = xp.where(first_abs > 0, phase, 1)
        diagonal = -phase * norm
        v[:, 0] -= diagonal
        v_norm = xp.sqrt(xp.sum(abs(v)**2, axis=1))
        v /= xp.where(v_norm > 0, v_norm, 1)[:, None]
        block = r[:, j:, j:]
        projection = xp.sum(v.conj()[:, :, None] * block, axis=1)
        block -= 2 * v[:, :, None] * projection[:, None, :]
        r[:, j, j] = diagonal * scale
        r[:, j + 1:, j] = 0
        reflectors.append(v)

    q_columns = rows if full_matrices else columns
    q = xp.broadcast_to(xp.eye(rows, q_columns, dtype=matrices.dtype),
                        (batch, rows, q_columns)).copy()
    for j in range(columns - 1, -1, -1):
        v = reflectors[j]
        block = q[:, j:, :]
        projection = xp.sum(v.conj()[:, :, None] * block, axis=1)
        block -= 2 * v[:, :, None] * projection[:, None, :]
    return q, r[:, :columns, :]


def qr_reduced_svd(matrices, full_matrices, xp):
    """Return U, singular values, Vh with the usual batched SVD shapes.

    For wide matrices factor A^H=QR, then take the SVD of R^H. For tall
    matrices factor A=QR. No singular values or physical planes are discarded.
    Singular vectors can differ from a direct SVD within degenerate subspaces.
    """
    if matrices.ndim != 3:
        raise ValueError('Expected a three-dimensional matrix batch.')
    rows, columns = matrices.shape[-2:]
    wide = rows < columns
    tall = matrices.conj().swapaxes(-1, -2) if wide else matrices
    q, r = _householder_qr(tall, full_matrices, xp)
    small = r.conj().swapaxes(-1, -2) if wide else r
    scale = xp.max(abs(small), axis=(-2, -1))
    divisor = xp.where(scale > 0, scale, 1)
    u_small, singular, vh_small = xp.linalg.svd(
        xp.ascontiguousarray(small / divisor[:, None, None]), full_matrices=False)
    singular *= divisor[:, None]
    rank = min(rows, columns)
    if wide:
        vh = vh_small @ q[:, :, :rank].conj().swapaxes(-1, -2)
        if full_matrices:
            vh = xp.concatenate((vh, q[:, :, rank:].conj().swapaxes(-1, -2)), axis=1)
        return u_small, singular, vh
    u = q[:, :, :rank] @ u_small
    if full_matrices and rows > columns:
        u = xp.concatenate((u, q[:, :, rank:]), axis=2)
    return u, singular, vh_small
