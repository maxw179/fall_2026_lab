"""Read MATLAB bead acquisitions without losing signed averaged intensities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat


@dataclass
class Acquisition:
    images: np.ndarray                 # (image, x/column, y/row)
    modes: np.ndarray                  # (mode, [m,n])
    total: np.ndarray                  # total commanded phase, waves
    diversities: np.ndarray
    baseline: np.ndarray
    imposed: np.ndarray
    focal_z: np.ndarray                # mm
    plane_index: np.ndarray            # MATLAB 1-based
    names: np.ndarray
    is_pd: np.ndarray
    acquired: np.ndarray
    optics: dict
    status: str
    source: Path
    config: dict
    cumulative: np.ndarray
    loop_number: int
    run_id: str
    system_ao: np.ndarray


def _records(value):
    if isinstance(value, dict):
        return [value]
    return list(np.asarray(value, dtype=object).reshape(-1))


def load_acquisition(folder, exclude_planes=None, exclude_images=None):
    folder = Path(folder).resolve()
    meta = loadmat(folder/'metadata.mat', simplify_cells=True)['metadata']
    try:
        data = loadmat(folder/'acquisition.mat', simplify_cells=True)
        raw = np.asarray(data['images'], dtype=float)
        acquired = np.zeros(len(np.atleast_1d(meta['names'])), dtype=bool)
        for i, entry in enumerate(_records(data.get('log', []))):
            if i >= len(acquired):
                raise ValueError('Acquisition log exceeds the planned image count.')
            acquired[i] = bool(entry['acquired'])
    except NotImplementedError:
        # MATLAB v7.3 stores numeric array dimensions in reverse order.
        try:
            import h5py
        except ImportError as exc:
            raise ImportError('MATLAB v7.3 input requires h5py. Install it in phase_diversity_env, '
                              'or re-save acquisition.mat using MATLAB save(..., "-v7").') from exc
        with h5py.File(folder/'acquisition.mat','r') as data:
            stored = np.asarray(data['images'], dtype=float)
            raw = stored.transpose(2,1,0) if stored.ndim == 3 else stored.T[:,:,None]
            acquired = np.zeros(len(np.atleast_1d(meta['names'])), dtype=bool)
            if 'log' in data and isinstance(data['log'], h5py.Group):
                for i, ref in enumerate(np.asarray(data['log']['acquired']).reshape(-1)):
                    if i >= len(acquired):
                        raise ValueError('Acquisition log exceeds the planned image count.')
                    acquired[i] = bool(np.asarray(data[ref]).item())
    names = np.asarray(meta['names'], dtype=str).reshape(-1)
    count = len(names)
    raw = raw.reshape((*np.asarray(meta['config']['imageSize'],dtype=int), count))
    mode_values = np.asarray(meta['modes_mn'],dtype=float).reshape(-1,2)
    if not np.all(np.isfinite(mode_values)) or not np.all(mode_values == np.round(mode_values)):
        raise ValueError('Zernike orders must be finite integers.')
    modes = mode_values.astype(int)
    nmode = len(modes)
    if meta.get('coefficientUnits') != 'waves':
        raise ValueError('Expected Zernike coefficients in waves.')
    if len(set(map(tuple, modes))) != nmode:
        raise ValueError('Zernike modes must be unique.')
    for m,n in modes:
        if n < 0 or abs(m) > n or (n-abs(m)) % 2:
            raise ValueError(f'Invalid Zernike mode {(m,n)}.')
    is_pd = np.asarray(meta['isPDImage'],dtype=bool).reshape(count)
    baseline = np.asarray(meta['baselineWaves'],dtype=float).reshape(nmode)
    imposed = np.asarray(meta['imposedWaves'],dtype=float).reshape(nmode)
    diversity = np.asarray(meta['diversityWaves'],dtype=float).reshape(count,nmode)
    total = np.asarray(meta['totalCommandWaves'],dtype=float).reshape(count,nmode)
    cumulative = np.asarray(meta.get('cumulativeCorrectionWaves',np.zeros(nmode)),dtype=float).reshape(nmode)
    expected = baseline + diversity + is_pd[:,None]*(imposed+cumulative)
    if not np.allclose(total,expected,rtol=1e-12,atol=1e-12):
        raise ValueError('Total commands do not match baseline + imposed + cumulative correction + diversity metadata.')
    if 'focalPositionsMm' in meta:
        focus = np.asarray(meta['focalPositionsMm'],dtype=float).reshape(count)
    else:
        focus = np.full(count,float(meta['config']['optics']['focalPositionUm'])/1000)
    plane = np.asarray(meta.get('planeIndex', np.ones(count)),dtype=int).reshape(count)
    system_ao = np.asarray(meta.get('systemAOPhaseSLM',[]),dtype=float)
    if system_ao.size:
        if system_ao.ndim != 2 or not np.all(np.isfinite(system_ao)) or np.any((system_ao < 0) | (system_ao > 255)):
            raise ValueError('System AO must be a finite 2D phase map in raw SLM units 0..255.')
        if 'slmSize' in meta and tuple(system_ao.shape) != tuple(np.asarray(meta['slmSize'],dtype=int)):
            raise ValueError('System AO dimensions do not match the SLM size.')
    if exclude_planes is not None:
        acquired[np.isin(plane,exclude_planes)] = False
    if exclude_images is not None:
        numbers=np.asarray(exclude_images)
        if (numbers.ndim!=1 or (numbers.size and not np.issubdtype(numbers.dtype,np.integer))
                or np.any(numbers<1) or np.any(numbers>count)):
            raise ValueError(f'Excluded image numbers must be integers from 1 to {count} (MATLAB 1-based).')
        acquired[numbers.astype(int)-1]=False
    images = raw.transpose(2,1,0)  # MATLAB row,col,image -> model image,x,y
    if not np.all(np.isfinite(images[acquired])):
        raise ValueError('An acquired image contains nonfinite values.')
    if not all(np.all(np.isfinite(a)) for a in [focus,total,diversity,baseline,imposed,cumulative]):
        raise ValueError('Acquisition metadata contains nonfinite values.')
    return Acquisition(images,modes,total,diversity,baseline,imposed,focus,plane,names,
                       is_pd,acquired,meta['config']['optics'],str(meta['status']),folder,meta['config'],
                       cumulative,int(meta.get('loopNumber',1)),str(meta.get('runId',folder.name)),system_ao)


def block_average(images, size):
    """Average nonoverlapping pixel blocks, maintaining physical FOV and signal units."""
    _,nx,ny = images.shape
    if size is None:
        return images.copy()
    if size < 2 or nx != ny or nx % size or size > nx:
        raise ValueError('Size must divide the acquired square image size (e.g. 64, 128, 256, 512).')
    factor = nx//size
    return images.reshape(len(images),size,factor,size,factor).mean(axis=(2,4))


def subtract_border_background(images, width):
    """Subtract one outer-border median per image at final analysis resolution."""
    images=np.asarray(images,dtype=float)
    if (images.ndim!=3 or not isinstance(width,(int,np.integer)) or isinstance(width,(bool,np.bool_))
            or width<1 or 2*width>=min(images.shape[-2:]) or not np.all(np.isfinite(images))):
        raise ValueError('Border width must be a positive integer smaller than half the finite image size.')
    border=np.ones(images.shape[-2:],bool);border[width:-width,width:-width]=False
    offsets=np.median(images[:,border],axis=1)
    return images-offsets[:,None,None],offsets


def optics_parameters(optics, pupil_size):
    return dict(N_order=int(optics['nonlinearOrder']),
                lambd=float(optics['excitationWavelengthNm'])*1e-6,
                n=float(optics['refractiveIndex']),num_apt=float(optics['NA']),
                f=float(optics['focalLengthMm']),mag=float(optics['magnification']),
                w_0=float(optics['beamWaistMm']),L_bfp=float(optics['backFocalPlaneWidthMm']),
                grid_bfp=int(pupil_size))


def write_json(path, value):
    def convert(item):
        if isinstance(item,np.ndarray): return convert(item.tolist())
        if isinstance(item,np.generic): return convert(item.item())
        if isinstance(item,Path): return str(item)
        if isinstance(item,dict): return {key:convert(value) for key,value in item.items()}
        if isinstance(item,(list,tuple)): return [convert(value) for value in item]
        if isinstance(item,float) and not np.isfinite(item): return None
        return item
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(convert(value),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def analysis_settings(acquisition, **overrides):
    """MATLAB is the settings source; explicit Python overrides remain optional."""
    fields = dict(size=('size',128),pupil_size=('pupilSize',64),background=('background',0.),
                  z_sign=('zSign',1),bead_center_um=('beadCenterUm',(0.,0.,0.)),
                  bead_planes=('beadPlanes',21),use_tex=('useTex',True))
    saved = acquisition.config.get('analysis',{})
    values = {key:saved.get(field,default) for key,(field,default) in fields.items()}
    for key,value in overrides.items():
        if value is not None: values[key] = value
    for key in ('size','pupil_size','z_sign','bead_planes'): values[key] = int(values[key])
    values['background'] = float(values['background']); values['use_tex'] = bool(values['use_tex'])
    return values


def solver_settings(acquisition, **overrides):
    fields = dict(rho=('rho',1e46),sample_planes=('samplePlanes',21),
                  sample_extent_um=('sampleExtentUm',4.),max_iterations=('maxIterations',50),
                  random_starts=('randomStarts',100))
    saved = acquisition.config.get('solver',{})
    values = {key:saved.get(field,default) for key,(field,default) in fields.items()}
    values.update({key:value for key,value in overrides.items() if value is not None})
    for key in ('sample_planes','max_iterations','random_starts'):values[key] = int(values[key])
    for key in ('rho','sample_extent_um'):values[key] = float(values[key])
    return values


def performance_settings(**overrides):
    """Execution tuning, independent of the microscope/model settings in MATLAB.

    Match the GPU demo's bounded batches and QR path. These settings change how
    the objective is evaluated, not the data, precision or convergence rules.
    """
    values = dict(svd_method='qr',screening_batch_size=8,psf_batch_size=8,
                  derivative_batch_size=8,frequency_batch_size=4096)
    unknown = overrides.keys()-values.keys()
    if unknown: raise ValueError(f'Unknown performance settings: {sorted(unknown)}')
    values.update({key:value for key,value in overrides.items() if value is not None})
    if values['svd_method'] not in ('qr','direct'):
        raise ValueError('svd_method must be qr or direct.')
    for key in values.keys()-{'svd_method'}:
        value = values[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < 1:
            raise ValueError(f'{key} must be a positive integer.')
        values[key] = int(value)
    return values
