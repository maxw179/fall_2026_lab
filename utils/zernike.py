from __future__ import annotations
from typing import Callable, Sequence
import math
import numpy as np


"""
Stores Zernike modes and their strengths.
Params:
    modes (Sequence[Sequence[int]] | np.ndarray): array of modes, with each row ordered as [m, n]
    strengths (Sequence[float] | np.ndarray): strength of each mode in waves
Returns:
    An Aberration object
"""
class Aberration:
    """
    Stores the modes and strengths of an aberration.
    Params:
        modes (Sequence[Sequence[int]] | np.ndarray): array of modes, with each row ordered as [m, n]
        strengths (Sequence[float] | np.ndarray): strength of each mode in waves
    Returns:
        None
    """
    def __init__(self, modes: Sequence[Sequence[int]] | np.ndarray, strengths: Sequence[float] | np.ndarray):
        if len({tuple(mode) for mode in modes}) != len(modes):
            raise ValueError("Zernike modes must be unique.")
        self.modes = modes
        self.strengths = np.array(strengths, dtype=float)

    """
    Formats the modes and strengths as a string.
    Params:
        None
    Returns:
        s: one line for each mode and its rounded strength
    """
    def __str__(self):
        s = ""
        for i, mode in enumerate(self.modes):
            s += f"n={mode[1]}, m={mode[0]}: {np.round(self.strengths[i], 3)}\n"
        return s.rstrip()

    """
    Adds the strengths of matching modes in two aberrations.
    Params:
        other (Aberration): aberration to add
    Returns:
        Aberration: combined modes and strengths
    """
    def __add__(self, other: Aberration):
        new_modes = np.copy(self.modes)
        new_strengths = np.copy(self.strengths)

        for i, mode in enumerate(np.array(other.modes)):
            matches = np.all(new_modes == mode, axis=1)

            if np.any(matches):
                new_strengths[matches] += other.strengths[i]
            else:
                new_modes = np.concatenate([new_modes, [mode]], axis=0)
                new_strengths = np.concatenate(
                    [new_strengths, [other.strengths[i]]]
                )

        return Aberration(new_modes, new_strengths)

    """
    Subtracts the strengths of matching modes in two aberrations.
    Params:
        other (Aberration): aberration to subtract
    Returns:
        Aberration: resulting modes and strengths
    """
    def __sub__(self, other: Aberration):
        new_modes = np.copy(self.modes)
        new_strengths = np.copy(self.strengths)

        for i, mode in enumerate(np.array(other.modes)):
            matches = np.all(new_modes == mode, axis=1)

            if np.any(matches):
                new_strengths[matches] -= other.strengths[i]
            else:
                new_modes = np.concatenate([new_modes, [mode]], axis=0)
                new_strengths = np.concatenate(
                    [new_strengths, [-other.strengths[i]]]
                )

        return Aberration(new_modes, new_strengths)

    """
    Gets the number of modes in the aberration.
    Params:
        None
    Returns:
        int: number of modes
    """
    def __len__(self):
        return len(self.modes)

    """
    Constructs a function that evaluates the aberration's phase map.
    Params:
        alpha (float): maximum polar angle [rad]
    Returns:
        function: phase map evaluated from theta and phi [rad]
    """
    def construct_map(self, alpha: float):
        return create_zernike_function(self.modes, self.strengths, alpha)


"""
Creates an aberration with zero strength for every specified mode.
Params:
    modes (Sequence[Sequence[int]] | np.ndarray): modes to include, with each row ordered as [m, n]
Returns:
    An EmptyAberration object
"""
class EmptyAberration(Aberration):
    """
    Initializes an aberration with zero strengths.
    Params:
        modes (Sequence[Sequence[int]] | np.ndarray): modes to include, with each row ordered as [m, n]
    Returns:
        None
    """
    def __init__(self, modes: Sequence[Sequence[int]] | np.ndarray = [[0, 0]]):
        super().__init__(modes, [0] * len(modes))


"""
Creates an aberration scaled to a specified RMS wavefront value.
Params:
    modes (Sequence[Sequence[int]] | np.ndarray): modes to include, with each row ordered as [m, n]
    raw_strengths (Sequence[float] | np.ndarray): initial strength of each mode in waves
    RMS_desired (float): desired RMS wavefront value in waves
    alpha (float): maximum polar angle [rad]
Returns:
    An RMSAberration object
"""
class RMSAberration(Aberration):
    """
    Scales the initial mode strengths to the desired RMS.
    Params:
        modes (Sequence[Sequence[int]] | np.ndarray): modes to include, with each row ordered as [m, n]
        raw_strengths (Sequence[float] | np.ndarray): initial strength of each mode in waves
        RMS_desired (float): desired RMS wavefront value in waves
        alpha (float): maximum polar angle [rad]
    Returns:
        None
    """
    def __init__(self, modes: Sequence[Sequence[int]] | np.ndarray, raw_strengths: Sequence[float] | np.ndarray, RMS_desired: float, alpha: float):
        super().__init__(modes, raw_strengths)
        RMS_val = zernike_RMS(self, alpha)
        if RMS_val == 0:
            if RMS_desired != 0:
                raise ValueError("Cannot scale a zero wavefront to a nonzero RMS.")
            return
        self.strengths = np.array(
            [s * (RMS_desired / RMS_val) for s in self.strengths]
        )


"""
Calculates the radial component of a Zernike mode.
Params:
    m (int): absolute azimuthal order
    n (int): radial order
    rho (float | np.ndarray): normalized pupil radius
Returns:
    total: radial Zernike value at rho, or zero when n - m is odd
"""
def radial_zernike(m: int, n: int, rho: float | np.ndarray):
    if (n - m) % 2 == 1:
        return 0

    total = 0
    for k in range(0, int((n - m) / 2 + 1)):
        numerator = (-1) ** k * math.factorial(n - k)
        denominator = (
            math.factorial(k)
            * math.factorial(int((n + m) / 2 - k))
            * math.factorial(int((n - m) / 2 - k))
        )
        total += (numerator / denominator) * rho ** (n - 2 * k)

    return total


"""
Gets the normalization factor for a Zernike mode.
Params:
    m (int): azimuthal order
    n (int): radial order
Returns:
    float: normalization factor
"""
def zernike_norm_factor(m: int, n: int):
    if m == 0:
        return np.sqrt(n + 1)
    return np.sqrt(2 * (n + 1))


"""
Evaluates a normalized Zernike mode at pupil coordinates.
Params:
    m (int): azimuthal order
    n (int): radial order
    rho (float | np.ndarray): normalized pupil radius
    phi (np.ndarray): pupil azimuthal angle [rad]
Returns:
    array or scalar: normalized Zernike value
"""
def zernike_mode(m: int, n: int, rho: float | np.ndarray, phi: np.ndarray):
    radial_component = radial_zernike(np.abs(m), n, rho)

    if m == 0:
        Z = radial_component
    elif m < 0:
        Z = radial_component * np.sin(np.abs(m) * phi)
    else:
        Z = radial_component * np.cos(np.abs(m) * phi)

    return zernike_norm_factor(m, n) * Z


"""
Converts pupil polar angles to normalized Zernike coordinates.
Params:
    theta_grid (np.ndarray): pupil polar angles [rad]
    phi_grid (np.ndarray): pupil azimuthal angles [rad]
    alpha (float): maximum polar angle [rad]
Returns:
    rho: normalized pupil radii
    psi: pupil azimuthal angles [rad]
"""
def pupil_polar_coords(theta_grid: np.ndarray, phi_grid: np.ndarray, alpha: float):
    rho = np.sin(theta_grid) / np.sin(alpha)
    psi = phi_grid
    return rho, psi


"""
Creates a function that evaluates the total Zernike phase map.
Params:
    modes (Sequence[Sequence[int]] | np.ndarray): modes ordered as [[m_1, n_1], [m_2, n_2], ...]
    strengths (Sequence[float] | np.ndarray): strength of each mode in waves
    alpha (float): maximum polar angle [rad]
Returns:
    total_zernike_map: function mapping theta and phi to phase [rad]
"""
def create_zernike_function(modes: Sequence[Sequence[int]] | np.ndarray, strengths: Sequence[float] | np.ndarray, alpha: float):
    modes = tuple(tuple(m) for m in modes)
    strengths = tuple(strengths)
    alpha = float(alpha)

    """
    Evaluates the combined phase inside the pupil.
    Params:
        theta_grid (np.ndarray): pupil polar angles [rad]
        phi_grid (np.ndarray): pupil azimuthal angles [rad]
    Returns:
        phase: combined phase [rad], set to zero outside the pupil
    """
    def total_zernike_map(theta_grid: np.ndarray, phi_grid: np.ndarray):
        rho, psi = pupil_polar_coords(theta_grid, phi_grid, alpha)
        phase = np.zeros_like(theta_grid, dtype=float)

        mask = rho <= 1.0
        rho_inside = rho[mask]
        psi_inside = psi[mask]

        if rho_inside.size == 0:
            return phase

        slm_waves = np.zeros_like(rho_inside)
        for i in range(len(modes)):
            m = modes[i][0]
            n = modes[i][1]
            strength = strengths[i]

            zernike = zernike_mode(m, n, rho_inside, psi_inside)
            slm_waves += strength * zernike

        phase[mask] = 2 * np.pi * slm_waves
        return phase

    return total_zernike_map


"""
Lists Zernike modes within a range of radial orders.
Params:
    min_order (int): first radial order to include
    max_order (int): first radial order to exclude
Returns:
    modes (Sequence[Sequence[int]] | np.ndarray): array of modes, with each row ordered as [m, n]
"""
def get_allowed_modes(min_order: int, max_order: int):
    modes = []
    for n in range(min_order, max_order):
        if n == 0:
            modes.append([0, 0])
        else:
            for m in range(-n, n + 1, 2):
                modes.append([m, n])
    return np.array(modes)


"""
Estimates Zernike strengths from a sampled phase map.
Params:
    z_map (Callable[[np.ndarray, np.ndarray], np.ndarray]): function mapping theta and phi to phase [rad]
    alpha (float): maximum polar angle [rad]
    decomp_order (int): first radial order to exclude
    n (int): number of samples along each Cartesian grid dimension
Returns:
    modes (Sequence[Sequence[int]] | np.ndarray): array of fitted modes, with each row ordered as [m, n]
    strengths (Sequence[float] | np.ndarray): estimated strength of each mode in waves
"""
def decompose_wavefront(z_map: Callable[[np.ndarray, np.ndarray], np.ndarray], alpha: float, decomp_order: int, n: int=100):
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)

    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    rho_grid = np.sqrt(x_grid**2 + y_grid**2)
    phi_grid = np.arctan2(y_grid, x_grid)

    mask = rho_grid <= 1.0
    theta_grid = np.arcsin(np.minimum(rho_grid, 1.0) * np.sin(alpha))
    wavefront = np.zeros_like(theta_grid)
    wavefront[mask] = z_map(theta_grid[mask], phi_grid[mask])

    modes = get_allowed_modes(0, decomp_order)
    strengths = []

    for mode in modes:
        Z = np.zeros_like(rho_grid)
        Z[mask] = zernike_mode(*mode, rho_grid[mask], phi_grid[mask])

        coeff = np.sum(wavefront[mask] * Z[mask]) / np.sum(Z[mask] ** 2)
        strengths.append(coeff / (2 * np.pi))

    return modes, strengths


"""
Creates a copy of an aberration scaled to a desired RMS.
Params:
    aberration (Aberration): aberration to rescale
    RMS_desired (float): desired RMS wavefront value in waves
    alpha (float): maximum polar angle [rad]
Returns:
    RMSAberration: aberration with rescaled strengths
"""
def rescale_aberration(aberration: Aberration, RMS_desired: float, alpha: float):
    return RMSAberration(
        aberration.modes,
        aberration.strengths,
        RMS_desired,
        alpha,
    )


"""
Calculates the RMS difference between two aberration phase maps.
Params:
    a_1 (Aberration): first aberration
    a_2 (Aberration): second aberration
    alpha (float): maximum polar angle [rad]
    n (int): number of samples along each Cartesian grid dimension
Returns:
    float: RMS phase difference in waves
"""
def zernike_RMS_difference(a_1: Aberration, a_2: Aberration, alpha: float, n: int=100):
    z_1 = a_1.construct_map(alpha)
    z_2 = a_2.construct_map(alpha)
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)

    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    rho_grid = np.sqrt(x_grid**2 + y_grid**2)
    phi_grid = np.arctan2(y_grid, x_grid)

    mask = rho_grid <= 1.0
    theta_grid = np.arcsin(np.minimum(rho_grid, 1.0) * np.sin(alpha))

    z_1_out = z_1(theta_grid[mask], phi_grid[mask])
    z_2_out = z_2(theta_grid[mask], phi_grid[mask])
    dz = (z_1_out - z_2_out) / (2 * np.pi)

    return np.sqrt(np.mean(dz**2))


"""
Calculates the RMS of an aberration relative to zero phase.
Params:
    a_1 (Aberration): aberration to measure
    alpha (float): maximum polar angle [rad]
    n (int): number of samples along each Cartesian grid dimension
Returns:
    float: RMS phase in waves
"""
def zernike_RMS(a_1: Aberration, alpha: float, n: int=100):
    return zernike_RMS_difference(a_1, EmptyAberration(), alpha)
