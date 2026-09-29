from __future__ import annotations
from typing import Callable, Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from utils.psf import Image_Mask_3D


def animate_reconstruction(info: dict, modes, microscope, *,
                           interval: float = 500, threshold: float | None = None,
                           percentile: float = 98, stride: int = 1,
                           res: int = 150, dpi: int = 100,
                           **reconstruction_kwargs):
    """Animate recorded optimizer iterates using zernike_plot and plot_sample_3D.

    Pass info from optimize_aberration_3D, its modes in the same order, and
    the original microscope. Supply grid, images, focal_z_levels,
    sample_z_levels, rho, diversities, and any nondefault mode/padding settings
    as reconstruction_kwargs. Objects are refitted once per selected iterate;
    rendering and replay do not rerun reconstruction. No optimization is run.

    Returns (aberration_animation, object_animation), two FuncAnimations.
    In notebooks use display(HTML(animation.to_jshtml())). Keep references
    to both animations; save with animation.save("name.gif", writer="pillow").
    Frames represent iterations, not elapsed wall time; interval is in ms.
    stride subsamples history while always including its final entry.

    By default a fixed threshold is the given percentile of the final selected
    object's values, matching the notebooks' percentile-based voxel plots.
    An explicit threshold overrides this. Object colors share one scale across
    all frames; phase colors retain zernike_plot's [-pi, pi] rad scale.
    Frames are rendered to RGB snapshots to retain the existing plot styles
    without rebuilding voxel artists during playback. Memory grows with frame
    count and dpi; use stride and dpi to reduce it. Figures are closed so only
    the requested notebook animations are displayed.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from optimization.reconstruction import estimate_sample_3D
    from utils.zernike import Aberration

    history = info.get("history", [])
    if not history:
        raise ValueError("info must contain a nonempty optimizer history.")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be finite and positive.")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if res < 2 or dpi <= 0:
        raise ValueError("res must be at least 2 and dpi must be positive.")
    if "aberration" in reconstruction_kwargs or "return_info" in reconstruction_kwargs:
        raise ValueError("aberration and return_info are managed by this helper.")
    indices = sorted(set(range(0, len(history), stride)) | {len(history) - 1})
    entries = [history[i] for i in indices]
    aberrations = []
    for entry in entries:
        strengths = np.asarray(entry["strengths"], dtype=float)
        if strengths.shape != (len(modes),) or not np.all(np.isfinite(strengths)):
            raise ValueError("History strengths must match modes and be finite.")
        aberrations.append(Aberration(modes, strengths))
    samples = [estimate_sample_3D(microscope=microscope, aberration=a,
                                  **reconstruction_kwargs) for a in aberrations]
    if threshold is None:
        threshold = float(np.percentile(samples[-1].image_mask, percentile))
    visible = [s.image_mask[s.image_mask > threshold] for s in samples]
    occupied = [v for v in visible if v.size]
    norm = (plt.Normalize(min(v.min() for v in occupied),
                          max(v.max() for v in occupied))
            if occupied else plt.Normalize(0, 1))
    del visible, occupied

    def snapshot(fig, title):
        try:
            fig.set_dpi(dpi)
            fig.suptitle(title, fontsize=9)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            return np.asarray(canvas.buffer_rgba())[..., :3].copy()
        finally:
            plt.close(fig)

    phase_frames, object_frames = [], []
    with plt.ioff():
        for entry, aberration, sample in zip(entries, aberrations, samples):
            title = f"Iteration {entry['iteration']}\nLoss = {entry['loss']:.3e}"
            fig, _ = zernike_plot(aberration.construct_map(microscope.alpha),
                                  microscope.alpha, res=res, show=False)
            phase_frames.append(snapshot(fig, title))
            fig, _ = plot_sample_3D(sample, threshold=threshold, norm=norm, show=False)
            object_frames.append(snapshot(fig, title))

        def make_animation(frames):
            height, width = frames[0].shape[:2]
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            artist = ax.imshow(frames[0])

            def update(index):
                artist.set_data(frames[index])
                return (artist,)

            animation = FuncAnimation(fig, update, frames=len(frames),
                                      interval=interval, blit=False, repeat=True)
            plt.close(fig)
            return animation

        return make_animation(phase_frames), make_animation(object_frames)


"""
Plots every plane of an image stack with a shared intensity scale and colorbar.
Params:
    x (np.ndarray): one-dimensional x coordinates [mm]
    y (np.ndarray): one-dimensional y coordinates [mm]
    z (np.ndarray): one-dimensional plane positions [mm]
    images (np.ndarray): intensity stack with shape (planes, x, y)
    ncols (int): maximum number of panels per row
    normalize (bool): whether to divide the whole stack by its positive maximum
    file_name (str | None): optional PNG filename in the project figures directory
Returns:
    fig (Figure): figure containing the stack and shared colorbar
    axs (np.ndarray): one-dimensional array of image axes in plane order
"""
def plot_image_stack(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     images: np.ndarray, ncols: int = 4,
                     normalize: bool = True, file_name: str | None = None):
    x, y, z = (np.asarray(axis) for axis in (x, y, z))
    images = np.asarray(images)
    if any(axis.ndim != 1 for axis in (x, y, z)) or len(x) < 2 or len(y) < 2 or len(z) == 0:
        raise ValueError("Provide 1D coordinates with at least two x/y samples and one plane.")
    if images.shape != (len(z), len(x), len(y)):
        raise ValueError("images must have shape (len(z), len(x), len(y)).")
    if not all(np.all(np.isfinite(a)) for a in (x, y, z, images)):
        raise ValueError("Coordinates and intensities must be finite.")
    if not isinstance(ncols, (int, np.integer)) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    values = _normalize_intensity(images) if normalize else images
    columns = min(ncols, len(z))
    rows = (len(z) + columns - 1) // columns
    fig, panels = plt.subplots(rows, columns, squeeze=False, sharex=True, sharey=True,
                               figsize=(3 * columns, 3 * rows), layout="constrained")
    axs = panels.ravel()[:len(z)]
    for ax, level, plane in zip(axs, z, values):
        artist = ax.imshow(plane.T, extent=_image_extent(x, y), origin="lower",
                           aspect="equal", cmap="Greys_r",
                           vmin=values.min(), vmax=values.max())
        ax.set_title(f"z = {level:.4g} mm")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.grid(False)
    for ax in panels.ravel()[len(z):]:
        fig.delaxes(ax)
    label = "Intensity (stack normalized)" if normalize else "Intensity (raw)"
    fig.colorbar(artist, ax=list(axs), label=label, shrink=0.8)
    if file_name is not None:
        fig.savefig(_figure_path(file_name), bbox_inches="tight")
    plt.show()
    return fig, axs


"""
Plots occupied voxels of a 3D sample at their physical coordinates.
Colors indicate sample values. Values at or below threshold are omitted.
Params:
    sample (Image_Mask_3D): sample containing planes on a shared lateral grid
    threshold (float): minimum sample value to display, exclusive
    marker_size (float): retained for compatibility; voxel sizes follow grid spacing
    alpha (float): voxel face and edge opacity between zero and one (default 0.3)
    file_name (str | None): optional PNG filename in the project figures directory
    cmap (str): colormap mapping visible sample values to voxel colors
    show (bool): display immediately; False allows animation rendering
    norm: optional shared Matplotlib color normalization
Returns:
    fig (Figure): figure containing the plot
    ax (Axes): three-dimensional axes
"""
def plot_sample_3D(sample: Image_Mask_3D, threshold: float = 0.0,
                   marker_size: float = 8.0, alpha: float = 0.3,
                   file_name: str | None = None, cmap: str = "Greys_r",
                   *, show: bool = True, norm=None):
    values = sample.image_mask
    if not np.all(np.isfinite(values)):
        raise ValueError("Sample must contain only finite values.")
    x, y = sample.get_xy()
    order = np.argsort(sample.z_levels)
    z = sample.z_levels[order]
    # Matplotlib expects (x, y, z), while samples store (z, x, y).
    values = values[order].transpose(1, 2, 0)
    def edges(centers, single_width):
        if len(centers) == 1:
            return np.array([centers[0] - single_width / 2,
                             centers[0] + single_width / 2])
        return np.r_[centers[0] - (centers[1] - centers[0]) / 2,
                     (centers[:-1] + centers[1:]) / 2,
                     centers[-1] + (centers[-1] - centers[-2]) / 2]

    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])
    xe, ye, ze = edges(x, dx), edges(y, dy), edges(z, min(dx, dy))
    visible = values > threshold
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if np.any(visible):
        colors = plt.cm.ScalarMappable(
            norm=norm if norm is not None else plt.Normalize(values[visible].min(), values[visible].max()),
            cmap=cmap)
        ax.voxels(xe[:, None, None], ye[None, :, None], ze[None, None, :],
                  visible, facecolors=colors.to_rgba(values.ravel(), alpha=alpha).reshape(values.shape + (4,)),
                  edgecolors=(0.3, 0.3, 0.3, alpha), linewidth=0.2, shade=False)
        fig.colorbar(colors, ax=ax, label="Sample value", shrink=0.7)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_xlim(xe.min(), xe.max())
    ax.set_ylim(ye.min(), ye.max())
    ax.set_zlim(ze.min(), ze.max())
    ax.set_box_aspect((np.ptp(xe), np.ptp(ye), np.ptp(ze)))
    ax.set_proj_type("ortho")
    if file_name is not None:
        fig.savefig(_figure_path(file_name), bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


"""
Normalizes finite intensity values by their positive maximum.
Params:
    intensity (np.ndarray): array of intensity values
Returns:
    normalized: scaled array, or zeros when the maximum is nonpositive
"""
def _normalize_intensity(intensity: np.ndarray):
    if not np.all(np.isfinite(intensity)):
        raise ValueError("Intensity must contain only finite values.")
    peak = np.max(intensity)
    return intensity / peak if peak > 0 else np.zeros_like(intensity)


"""
Gets image boundaries extending half a pixel beyond the sample centers.
Params:
    x (np.ndarray): uniformly spaced x coordinates [mm]
    y (np.ndarray): uniformly spaced y coordinates [mm]
Returns:
    extent: left, right, bottom, and top image boundaries [mm]
"""
def _image_extent(x: np.ndarray, y: np.ndarray):
    dx = (x[1] - x[0]) / 2
    dy = (y[1] - y[0]) / 2
    return [x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy]


"""
Gets a PNG output path and creates the project figures directory if needed.
Params:
    file_name (str | None): output filename without the PNG extension
Returns:
    path: output path in the project figures directory
"""
def _figure_path(file_name: str | None):
    directory = Path(__file__).resolve().parent.parent / "figures"
    directory.mkdir(exist_ok=True)
    return directory / f"{file_name}.png"


"""
Plots an intensity map with axis labels and a colorbar.
Params:
    x (np.ndarray): x coordinates [mm]
    y (np.ndarray): y coordinates [mm]
    intensity_map (np.ndarray): two-dimensional intensity values
    file_name (str | None): optional filename for a PNG saved in the figures directory
    normalize (bool): whether to divide the intensity map by its maximum
    vmax (float | bool): upper color limit; False uses the map's maximum
Returns:
    fig (Figure): figure containing the plot
    ax (Axes): main plot axes
"""
def plot_intensity(x: np.ndarray, y: np.ndarray, intensity_map: np.ndarray, file_name: str | None=None, normalize: bool=True, vmax: float | bool=False):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

    if normalize:
        intensity_map = _normalize_intensity(intensity_map)
    if not vmax:
        vmax = np.max(intensity_map)

    plt.imshow(
        intensity_map.T,
        extent=_image_extent(x, y),
        origin="lower",
        aspect="equal",
        vmin=np.min(intensity_map),
        vmax=vmax,
        cmap="Greys_r",
    )
    ax.grid(False)

    x_step = (np.max(x) - np.min(x)) / 5
    y_step = (np.max(y) - np.min(y)) / 5
    x_ticks = np.arange(np.min(x), np.max(x) + x_step, x_step)
    y_ticks = np.arange(np.min(y), np.max(y) + y_step, y_step)
    plt.xticks(x_ticks, rotation=45)
    plt.yticks(y_ticks, rotation=45)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    if normalize:
        plt.colorbar(label="Intensity (Normalized)")
    else:
        plt.colorbar(label="Intensity (Raw)")

    plt.gca().set_aspect("equal")
    if file_name is not None:
        plt.savefig(_figure_path(file_name), bbox_inches="tight")
    plt.show()
    return fig, ax


"""
Plots an intensity map without ticks, axis labels, or a colorbar.
Params:
    x (np.ndarray): x coordinates [mm]
    y (np.ndarray): y coordinates [mm]
    intensity_map (np.ndarray): two-dimensional intensity values
    file_name (str | None): optional filename for a PNG saved in the figures directory
    normalize (bool): whether to divide the intensity map by its maximum
    vmax (float | bool): upper color limit; False uses the map's maximum
Returns:
    fig (Figure): figure containing the plot
    ax (Axes): main plot axes
"""
def plot_intensity_clean(x: np.ndarray, y: np.ndarray, intensity_map: np.ndarray, file_name: str | None=None, normalize: bool=True, vmax: float | bool=False):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

    if normalize:
        intensity_map = _normalize_intensity(intensity_map)
    if not vmax:
        vmax = np.max(intensity_map)

    plt.imshow(
        intensity_map.T,
        extent=_image_extent(x, y),
        origin="lower",
        aspect="equal",
        vmin=np.min(intensity_map),
        vmax=vmax,
        cmap="Greys_r",
    )
    ax.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect("equal")

    if file_name is not None:
        plt.savefig(_figure_path(file_name), bbox_inches="tight")
    plt.show()
    return fig, ax


"""
Plots an unnormalized intensity map with axis labels and a colorbar.
Params:
    x (np.ndarray): x coordinates [mm]
    y (np.ndarray): y coordinates [mm]
    intensity_map (np.ndarray): two-dimensional intensity values
    file_name (str | None): optional filename for a PNG saved in the figures directory
Returns:
    fig (Figure): figure containing the plot
    ax (Axes): main plot axes
"""
def plot_intensity_unnorm(x: np.ndarray, y: np.ndarray, intensity_map: np.ndarray, file_name: str | None=None):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=300)

    plt.imshow(
        intensity_map.T,
        extent=_image_extent(x, y),
        origin="lower",
        aspect="equal",
        cmap="Greys_r",
    )
    ax.grid(False)

    x_step = (np.max(x) - np.min(x)) / 5
    y_step = (np.max(y) - np.min(y)) / 5
    x_ticks = np.arange(np.min(x), np.max(x) + x_step, x_step)
    y_ticks = np.arange(np.min(y), np.max(y) + y_step, y_step)
    plt.xticks(x_ticks, rotation=45)
    plt.yticks(y_ticks, rotation=45)

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.colorbar(label="Intensity (Unnormalized)")
    plt.gca().set_aspect("equal")

    if file_name is not None:
        plt.savefig(_figure_path(file_name), bbox_inches="tight")
    plt.show()
    return fig, ax


"""
Plots multiple intensity maps, each normalized by its own maximum.
Params:
    axs (Sequence[Axes] | np.ndarray): axes on which to draw the maps
    x (np.ndarray): x coordinates [mm]
    y (np.ndarray): y coordinates [mm]
    intensity_maps (Sequence[np.ndarray]): intensity map for each axis
    is_edge (bool): whether to label axes along the outer edge
    is_horizontal (bool): whether the axes are arranged horizontally
Returns:
    fig (Figure): figure containing the plot
    axs (np.ndarray): array of panel axes
"""
def plot_many_intensity(axs: Sequence[Axes] | np.ndarray, x: np.ndarray, y: np.ndarray, intensity_maps: Sequence[np.ndarray], is_edge: bool=False, is_horizontal: bool=True):
    axs = np.ravel(axs)
    for i, ax in enumerate(axs):
        if is_horizontal:
            if is_edge:
                ax.set_xlabel("x [mm]")
            if i == 0:
                ax.set_ylabel("y [mm]")
        else:
            if is_edge:
                ax.set_ylabel("y [mm]")
            if i == len(axs) - 1:
                ax.set_xlabel("x [mm]")

        ax.imshow(
            _normalize_intensity(intensity_maps[i]).T,
            extent=_image_extent(x, y),
            origin="lower",
            aspect="equal",
            vmin=0,
            vmax=1,
            cmap="Greys_r",
        )
        ax.grid(False)
        ax.tick_params(axis="both", labelrotation=45)
        ax.set_xlabel("x [mm]")
        if i == 0:
            ax.set_ylabel("y [mm]")
    return axs[0].figure, axs


"""
Plots a phase map on polar axes.
Params:
    z_map (Callable[[np.ndarray, np.ndarray], np.ndarray]): function that evaluates the phase from theta and phi
    alpha (float): maximum polar angle used to calculate theta [rad]
    file_name (str | None): optional filename for a PNG saved in the figures directory
    res (int): number of samples along each polar grid dimension
Returns:
    fig (Figure): figure containing the plot
    ax (Axes): polar phase-map axes
"""
def zernike_plot(z_map: Callable[[np.ndarray, np.ndarray], np.ndarray], alpha: float, file_name: str | None=None, res: int=500,
                 *, show: bool = True):
    fig = plt.figure(figsize=(2, 3), dpi=600)

    rho = np.linspace(0, 1, res)
    phi = np.linspace(0, 2 * np.pi, res)
    rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
    theta_grid = np.arcsin(rho_grid * np.sin(alpha))
    z = z_map(theta_grid, phi_grid)

    polar_ax = fig.add_axes([0.1, 0.25, 0.8, 0.7], projection="polar")
    pcm = polar_ax.pcolormesh(
        phi,
        rho,
        z,
        edgecolors="face",
        vmin=-np.pi,
        vmax=np.pi,
        cmap="plasma",
    )
    polar_ax.grid(False)
    polar_ax.set_xticklabels([])
    polar_ax.set_yticklabels([])

    cbar = fig.colorbar(
        pcm,
        ax=polar_ax,
        orientation="horizontal",
        pad=0.15,
        fraction=0.08,
    )
    cbar.set_label("Phase (rad)")

    if file_name is not None:
        plt.savefig(_figure_path(file_name), bbox_inches="tight")
    if show:
        plt.show()
    return fig, polar_ax


"""
Plots a normalized intensity map with an inset polar phase map.
Params:
    x (np.ndarray): x coordinates [mm]
    y (np.ndarray): y coordinates [mm]
    intensity_map (np.ndarray): two-dimensional intensity values
    z_map (Callable[[np.ndarray, np.ndarray], np.ndarray]): function that evaluates the phase from theta and phi
    alpha (float): maximum polar angle used to calculate theta [rad]
    file_name (str | None): optional filename for a PNG saved in the figures directory
Returns:
    fig (Figure): figure containing the plot
    ax (Axes): main plot axes
"""
def composite_plot(x: np.ndarray, y: np.ndarray, intensity_map: np.ndarray, z_map: Callable[[np.ndarray, np.ndarray], np.ndarray], alpha: float, file_name: str | None=None):
    fig, ax = plt.subplots(dpi=300)

    plt.imshow(
        _normalize_intensity(intensity_map).T,
        extent=_image_extent(x, y),
        origin="lower",
        aspect="equal",
        cmap="Greys_r",
    )
    ax.grid(False)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    rho = np.linspace(0, 1, 500)
    phi = np.linspace(0, 2 * np.pi, 500)
    rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
    theta_grid = np.arcsin(rho_grid * np.sin(alpha))
    z = z_map(theta_grid, phi_grid)

    size_param = 0.05
    polar_ax = fig.add_axes(
        [
            0.58 - size_param,
            0.68 - size_param,
            0.2 + size_param,
            0.2 + size_param,
        ],
        projection="polar",
    )
    polar_ax.pcolormesh(
        phi,
        rho,
        z,
        edgecolors="face",
        vmin=-2 * np.pi,
        vmax=2 * np.pi,
        cmap="plasma",
    )
    polar_ax.grid(False)
    polar_ax.set_xticklabels([])
    polar_ax.set_yticklabels([])

    plt.colorbar(label="Intensity (Normalized)")
    if file_name is not None:
        plt.savefig(_figure_path(file_name), bbox_inches="tight")
    plt.show()
    return fig, ax


"""
Plots a normalized intensity map with an inset polar phase map and no axis labels.
Params:
    x (np.ndarray): x coordinates [mm]
    y (np.ndarray): y coordinates [mm]
    intensity_map (np.ndarray): two-dimensional intensity values
    z_map (Callable[[np.ndarray, np.ndarray], np.ndarray]): function that evaluates the phase from theta and phi
    alpha (float): maximum polar angle used to calculate theta [rad]
    file_name (str | None): optional filename for a PNG saved in the figures directory
Returns:
    fig (Figure): figure containing the plot
    ax (Axes): main plot axes
"""
def composite_plot_clean(x: np.ndarray, y: np.ndarray, intensity_map: np.ndarray, z_map: Callable[[np.ndarray, np.ndarray], np.ndarray], alpha: float, file_name: str | None=None):
    fig, ax = plt.subplots(dpi=300)

    plt.imshow(
        _normalize_intensity(intensity_map).T,
        extent=_image_extent(x, y),
        origin="lower",
        aspect="equal",
        cmap="Greys_r",
    )
    ax.grid(False)
    plt.xticks([])
    plt.yticks([])

    rho = np.linspace(0, 1, 500)
    phi = np.linspace(0, 2 * np.pi, 500)
    rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
    theta_grid = np.arcsin(rho_grid * np.sin(alpha))
    z = z_map(theta_grid, phi_grid)

    size_param = 0.05
    polar_ax = fig.add_axes(
        [
            0.58 - size_param,
            0.68 - size_param,
            0.2 + size_param,
            0.2 + size_param,
        ],
        projection="polar",
    )
    polar_ax.pcolormesh(
        phi,
        rho,
        z,
        edgecolors="face",
        vmin=-2 * np.pi,
        vmax=2 * np.pi,
        cmap="plasma",
    )
    polar_ax.grid(False)
    polar_ax.set_xticklabels([])
    polar_ax.set_yticklabels([])

    if file_name is not None:
        plt.savefig(_figure_path(file_name), bbox_inches="tight")
    plt.show()
    return fig, ax


"""
Plots multiple intensity maps with inset polar phase maps.
Params:
    fig (Figure): figure containing the axes
    axs (Sequence[Axes] | np.ndarray): axes on which to draw the intensity maps
    x (np.ndarray): x coordinates [mm]
    y (np.ndarray): y coordinates [mm]
    intensity_maps (Sequence[np.ndarray]): intensity map for each axis
    z_maps (Sequence[Callable[[np.ndarray, np.ndarray], np.ndarray]]): phase-map function for each axis
    alpha (float): maximum polar angle used to calculate theta [rad]
    vmin (float): lower color limit for the phase maps
    vmax (float | bool): upper color limit for the phase maps
    is_edge (bool): whether to label axes along the outer edge
    is_horizontal (bool): whether the axes are arranged horizontally
Returns:
    fig (Figure): figure containing the plot
    axs (np.ndarray): array of panel axes
"""
def many_composite(
    fig: Figure,
    axs: Sequence[Axes] | np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    intensity_maps: Sequence[np.ndarray],
    z_maps: Sequence[Callable[[np.ndarray, np.ndarray], np.ndarray]],
    alpha: float,
    vmin: float=-2 * np.pi,
    vmax: float | bool=2 * np.pi,
    is_edge: bool=False,
    is_horizontal: bool=True,
):
    axs = np.ravel(axs)
    intensity_vmax = max(np.max(i_map) for i_map in intensity_maps)

    for i, ax in enumerate(axs):
        ax.imshow(
            intensity_maps[i].T,
            vmax=intensity_vmax,
            extent=_image_extent(x, y),
            origin="lower",
            aspect="equal",
            cmap="Greys_r",
        )
        ax.grid(False)
        ax.tick_params(axis="x", labelrotation=45)
        ax.tick_params(axis="y", labelrotation=45)

        if is_horizontal:
            if is_edge:
                ax.set_xlabel("x [mm]")
            if i == 0:
                ax.set_ylabel("y [mm]")
        else:
            if is_edge:
                ax.set_ylabel("y [mm]")
            if i == len(axs) - 1:
                ax.set_xlabel("x [mm]")

        rho = np.linspace(0, 1, 500)
        phi = np.linspace(0, 2 * np.pi, 500)
        rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
        theta_grid = np.arcsin(rho_grid * np.sin(alpha))
        z = z_maps[i](theta_grid, phi_grid)

        inset = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc="upper right",
            borderpad=0,
        )
        fig.canvas.draw()
        bbox = inset.get_window_extent(
            fig.canvas.get_renderer()
        ).transformed(fig.transFigure.inverted())
        inset.remove()

        polar_ax = fig.add_axes(
            [bbox.x0, bbox.y0, bbox.width, bbox.height],
            projection="polar",
        )
        polar_ax.pcolormesh(
            phi,
            rho,
            z,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        polar_ax.grid(False)
        polar_ax.set_xticklabels([])
        polar_ax.set_yticklabels([])
    return fig, axs
