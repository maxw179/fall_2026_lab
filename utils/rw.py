from __future__ import annotations
from typing import Callable, Sequence
from utils.zernike import Aberration
import sys
import numpy as np

sys.path.append("../")
sys.path.append("../utils")


"""
Gets endpoint-inclusive focal coordinates with at least two samples.
Params:
    length (float): extent of the focal axis [mm]
    samples (int): integer number of samples, at least two
    offset (float): center of the focal axis [mm]
Returns:
    coordinates: one-dimensional focal coordinates [mm]
"""
def get_ffp_axis(length: float, samples: int, offset: float):
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("Focal-plane sample counts must be integers >= 2.")
    return offset + np.linspace(-length / 2, length / 2, samples)


"""
Gets the amplitude of the Gaussian beam across the back focal plane.
Params:
    mag (float): the magnification
    w_0 (float): the beam waist [mm]
    f (float): the focal length [mm]
    n (float): the refractive index
    s_perp (float | np.ndarray): the transverse direction coordinate
Returns:
    gaussian_amplitude: the Gaussian amplitude at each value of s_perp
"""
def gaussian_amplitude_s_perp(mag: float, w_0: float, f: float, n: float, s_perp: float | np.ndarray):
    s_waist = (mag * w_0) / (f * n)
    gaussian_amplitude = np.exp(-1 * ((s_perp / s_waist) ** 2))
    return gaussian_amplitude


"""
Gets the angular strength factors in the Richards-Wolf integral.
Params:
    theta (np.ndarray): the polar angle, measured from the optical axis [rad]
    phi (np.ndarray): the azimuthal angle [rad]
Returns:
    a_x: the x-component of the angular strength
    a_y: the y-component of the angular strength
    a_z: the z-component of the angular strength
"""
def strength_angular(theta: np.ndarray, phi: np.ndarray):
    cosT = np.cos(theta)
    sinT = np.sin(theta)
    cosP = np.cos(phi)
    sinP = np.sin(phi)

    a_x = cosT + (1 - cosT) * sinP**2
    a_y = (cosT - 1) * cosP * sinP
    a_z = -sinT * cosP

    return a_x, a_y, a_z


"""
Creates a square coordinate grid across the back focal plane.
Params:
    L_bfp (float): the side length of the back focal plane grid [mm]
    grid_bfp (int): the number of samples along each grid dimension
Returns:
    dxy_bfp: the grid spacing [mm]
    x_bfp: the x coordinates across the grid [mm]
    y_bfp: the y coordinates across the grid [mm]
"""
def get_bfp_grid(L_bfp: float, grid_bfp: int):
    dxy_bfp = L_bfp / grid_bfp
    x_bfp_1d = (np.arange(grid_bfp) - grid_bfp / 2) * dxy_bfp
    y_bfp_1d = (np.arange(grid_bfp) - grid_bfp / 2) * dxy_bfp
    x_bfp, y_bfp = np.meshgrid(x_bfp_1d, y_bfp_1d, indexing="ij")
    return dxy_bfp, x_bfp, y_bfp


"""
Converts back focal plane positions to ray directions and pupil angles.
Params:
    f (float): the focal length [mm]
    n (float): the refractive index
    alpha (float): the maximum pupil polar angle [rad]
    x_bfp (np.ndarray): the x coordinates across the back focal plane [mm]
    y_bfp (np.ndarray): the y coordinates across the back focal plane [mm]
Returns:
    mask: positions inside the pupil
    theta: the polar angle inside the pupil [rad]
    phi: the azimuthal angle [rad]
    sx: the x component of the ray direction
    sy: the y component of the ray direction
    sz: the z component of the ray direction inside the pupil
"""
def bfp_coord_convert(f: float, n: float, alpha: float, x_bfp: np.ndarray, y_bfp: np.ndarray):
    if not 0 < alpha < np.pi / 2:
        raise ValueError("Pupil angle must satisfy 0 < alpha < pi/2 (0 < NA < n).")
    #divide by f*n to express pupil positions as transverse ray directions
    sx = x_bfp / (f * n)
    sy = y_bfp / (f * n)

    s_perp2 = sx**2 + sy**2
    s_max2 = np.sin(alpha) ** 2
    mask = s_perp2 <= s_max2

    #the ray directions have unit length within the pupil
    sz = np.zeros_like(sx)
    sz[mask] = np.sqrt(1.0 - s_perp2[mask])

    theta = np.zeros_like(sx)
    theta[mask] = np.arccos(sz[mask])

    phi = np.mod(np.arctan2(sy, sx), 2 * np.pi)
    return mask, theta, phi, sx, sy, sz


"""
Evaluates an aberration phase map on the back focal plane grid.
Params:
    alpha (float): the maximum pupil polar angle [rad]
    f (float): the focal length [mm]
    n (float): the refractive index
    L_bfp (float): the side length of the back focal plane grid [mm]
    aberration (Aberration): the aberration used to construct the phase map
    grid_bfp (int): the number of samples along each grid dimension
Returns:
    Z: the phase map, set to zero outside the pupil [rad]
"""
def get_phase_map(alpha: float, f: float, n: float, L_bfp: float, aberration: Aberration, grid_bfp: int):
    z_map = aberration.construct_map(alpha)
    dxy_bfp, x_bfp, y_bfp = get_bfp_grid(L_bfp, grid_bfp)

    mask, theta, phi, sx, sy, sz = bfp_coord_convert(
        f, n, alpha, x_bfp, y_bfp
    )

    Z = np.zeros_like(x_bfp)
    phase = z_map(theta[mask], phi[mask])
    Z[mask] = phase
    return Z


"""
Constructs the complex pupil field from an aberration phase map.
Params:
    alpha (float): the maximum pupil polar angle [rad]
    mag (float): the magnification
    w_0 (float): the beam waist [mm]
    f (float): the focal length [mm]
    n (float): the refractive index
    L_bfp (float): the side length of the back focal plane grid [mm]
    aberration (Aberration): the aberration used to construct the phase map
    grid_bfp (int): the number of samples along each grid dimension
    gaussian (bool): whether to apply the Gaussian amplitude profile
Returns:
    U: the complex pupil field, set to zero outside the pupil
"""
def get_pupil_function(
    alpha: float, mag: float, w_0: float, f: float, n: float, L_bfp: float, aberration: Aberration, grid_bfp: int, gaussian: bool=True
):
    z_map = aberration.construct_map(alpha)
    dxy_bfp, x_bfp, y_bfp = get_bfp_grid(L_bfp, grid_bfp)

    mask, theta, phi, sx, sy, sz = bfp_coord_convert(
        f, n, alpha, x_bfp, y_bfp
    )

    U = np.zeros_like(x_bfp, dtype=complex)
    U[mask] = np.exp(1j * z_map(theta[mask], phi[mask]))

    if gaussian:
        gauss = gaussian_amplitude_s_perp(
            mag,
            w_0,
            f,
            n,
            np.sqrt(sx**2 + sy**2),
        )
        U[mask] *= gauss[mask]

    return U


"""
Computes the scalar focal field by integrating over the back focal plane.
Params:
    L_ffp_x (float): the x extent of the focal plane grid [mm]
    L_ffp_y (float): the y extent of the focal plane grid [mm]
    grid_ffp_x (int): the number of focal plane samples along x
    grid_ffp_y (int): the number of focal plane samples along y
    x_offset (float): the x offset of the focal plane grid [mm]
    y_offset (float): the y offset of the focal plane grid [mm]
    alpha (float): the maximum pupil polar angle [rad]
    k (float): the wave number used in the propagation phase [rad/mm]
    f (float): the focal length [mm]
    n (float): the refractive index
    mag (float): the magnification
    w_0 (float): the beam waist [mm]
    L_bfp (float): the side length of the back focal plane grid [mm]
    aberration (Aberration): the aberration used to construct the phase map
    grid_bfp (int): the number of samples along each back focal plane dimension
    z (float): the axial focal plane position [mm]
Returns:
    h: the complex scalar field on the focal plane grid
"""
def get_scalar_h(
    L_ffp_x: float,
    L_ffp_y: float,
    grid_ffp_x: int,
    grid_ffp_y: int,
    x_offset: float,
    y_offset: float,
    alpha: float,
    k: float,
    f: float,
    n: float,
    mag: float,
    w_0: float,
    L_bfp: float,
    aberration: Aberration,
    grid_bfp: int,
    z: float=0,
):
    z_map = aberration.construct_map(alpha)
    dxy_bfp, x_bfp, y_bfp = get_bfp_grid(L_bfp, grid_bfp)

    mask, theta, phi, sx, sy, sz = bfp_coord_convert(
        f, n, alpha, x_bfp, y_bfp
    )

    U = np.zeros_like(x_bfp, dtype=complex)
    gauss = gaussian_amplitude_s_perp(
        mag,
        w_0,
        f,
        n,
        np.sqrt(sx**2 + sy**2),
    )

    phase = np.exp(1j * z_map(theta[mask], phi[mask]))
    z_phase = np.exp(1j * k * z * sz[mask])
    U[mask] = gauss[mask] * phase * z_phase

    x_ffp = get_ffp_axis(L_ffp_x, grid_ffp_x, x_offset)
    y_ffp = get_ffp_axis(L_ffp_y, grid_ffp_y, y_offset)

    #each back focal plane grid axis supplies one direction cosine
    sx_1d = sx[:, 0]
    sy_1d = sy[0, :]
    dsx = sx_1d[1] - sx_1d[0]
    dsy = sy_1d[1] - sy_1d[0]

    #the two matrix products apply the x and y propagation phases separately
    Ax = np.exp(1j * k * np.outer(x_ffp, sx_1d))
    Ay = np.exp(1j * k * np.outer(sy_1d, y_ffp))
    h = Ax @ U @ Ay * dsx * dsy

    return h


"""
Computes a vector Richards-Wolf intensity map and applies multiphoton order.
Params:
    L_ffp_x (float): the x extent of the focal plane grid [mm]
    L_ffp_y (float): the y extent of the focal plane grid [mm]
    grid_ffp_x (int): the number of focal plane samples along x
    grid_ffp_y (int): the number of focal plane samples along y
    x_offset (float): the x offset of the focal plane grid [mm]
    y_offset (float): the y offset of the focal plane grid [mm]
    alpha (float): the maximum pupil polar angle [rad]
    k (float): the wave number used in the propagation phase [rad/mm]
    f (float): the focal length [mm]
    n (float): the refractive index
    mag (float): the magnification
    w_0 (float): the beam waist [mm]
    L_bfp (float): the side length of the back focal plane grid [mm]
    aberration (Aberration): the aberration used to construct the phase map
    grid_bfp (int): the number of samples along each back focal plane dimension
    N_order (int): the multiphoton excitation order
    z (float): the axial focal plane position [mm]
Returns:
    x_ffp: the focal plane x coordinates [mm]
    y_ffp: the focal plane y coordinates [mm]
    I: the intensity raised to N_order
"""
def rw_fast(
    L_ffp_x: float,
    L_ffp_y: float,
    grid_ffp_x: int,
    grid_ffp_y: int,
    x_offset: float,
    y_offset: float,
    alpha: float,
    k: float,
    f: float,
    n: float,
    mag: float,
    w_0: float,
    L_bfp: float,
    aberration: Aberration,
    grid_bfp: int,
    N_order: int,
    z: float,
):
    z_map = aberration.construct_map(alpha)
    dxy_bfp, x_bfp, y_bfp = get_bfp_grid(L_bfp, grid_bfp)

    mask, theta, phi, sx, sy, sz = bfp_coord_convert(
        f, n, alpha, x_bfp, y_bfp
    )

    U = np.zeros_like(x_bfp, dtype=complex)
    gauss = gaussian_amplitude_s_perp(
        mag,
        w_0,
        f,
        n,
        np.sqrt(sx**2 + sy**2),
    )
    phase = np.exp(1j * z_map(theta[mask], phi[mask]))
    U[mask] = gauss[mask] * phase

    #the angular factors distribute the pupil field among three field components
    a_x, a_y, a_z = strength_angular(theta, phi)

    z_phase = np.ones_like(U, dtype=complex)
    z_phase[mask] = np.exp(1j * k * z * sz[mask])

    P_x = np.zeros_like(U, dtype=complex)
    P_y = np.zeros_like(U, dtype=complex)
    P_z = np.zeros_like(U, dtype=complex)

    #apply the angular weighting only inside the pupil
    inv_sqrt_sz = np.zeros_like(sz)
    inv_sqrt_sz[mask] = 1.0 / np.sqrt(sz[mask])

    P_x[mask] = U[mask] * a_x[mask] * inv_sqrt_sz[mask] * z_phase[mask]
    P_y[mask] = U[mask] * a_y[mask] * inv_sqrt_sz[mask] * z_phase[mask]
    P_z[mask] = U[mask] * a_z[mask] * inv_sqrt_sz[mask] * z_phase[mask]

    x_ffp = get_ffp_axis(L_ffp_x, grid_ffp_x, x_offset)
    y_ffp = get_ffp_axis(L_ffp_y, grid_ffp_y, y_offset)

    sx_1d = sx[:, 0]
    sy_1d = sy[0, :]

    #factor the transverse propagation phase into two matrix products
    Ax = np.exp(1j * k * np.outer(x_ffp, sx_1d))
    Ay = np.exp(1j * k * np.outer(sy_1d, y_ffp))

    C = -1j * k * f / (2 * np.pi)
    dsx = dxy_bfp / (f * n)
    dsy = dxy_bfp / (f * n)
    scale = C * dsx * dsy

    E_x = scale * (Ax @ P_x @ Ay)
    E_y = scale * (Ax @ P_y @ Ay)
    E_z = scale * (Ax @ P_z @ Ay)

    #sum the component intensities before applying the excitation order
    I1 = np.abs(E_x) ** 2 + np.abs(E_y) ** 2 + np.abs(E_z) ** 2
    I = I1**N_order

    return x_ffp, y_ffp, I
