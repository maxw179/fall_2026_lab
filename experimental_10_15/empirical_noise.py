"""Effective Poisson/Gaussian noise in final saved, binned image units.

Repeat differences remove static scene structure. Smooth brightness/first- and
second-order derivative nuisance terms reduce drift/bleaching contamination;
no measured image is aligned or modified. Effective gain is a moment match,
not an absolute photon/read-noise calibration from raw exposures.
"""
import numpy as np
from scipy import ndimage,optimize

MAD_NORMAL=.6744897501960817


def background_noise(images):
    images=np.asarray(images,dtype=float)
    width=max(2,int(np.ceil(min(images.shape[-2:])*.15)))
    border=np.ones(images.shape[-2:],bool);border[width:-width,width:-width]=False
    backgrounds=np.median(images[:,border],axis=1)
    sigma=np.median(abs(images[:,border]-backgrounds[:,None]),axis=1)/MAD_NORMAL
    return backgrounds,sigma


def _variance_fit(records):
    signal=np.concatenate([r['signal'] for r in records]);residual=np.concatenate([r['residual'] for r in records])
    edges=np.r_[-np.inf,0.,10.*2.**np.arange(10),np.inf]
    bins=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        keep=(signal>=lo)&(signal<hi);n=int(keep.sum())
        if n<30:continue
        values=residual[keep]
        variance=float((np.median(abs(values-np.median(values)))/MAD_NORMAL)**2)
        bins.append(dict(signal=max(float(np.median(signal[keep])),0.),variance=variance,pixels=n))
    if len(bins)<3 or max(row['signal'] for row in bins)<30:
        raise ValueError('Insufficient repeated bright pixels to estimate a signal-dependent variance.')
    x=np.array([row['signal'] for row in bins]);y=np.array([row['variance'] for row in bins])
    n=np.array([row['pixels'] for row in bins])
    weights=np.sqrt(n)/np.maximum(y,1.)
    gain,floor=optimize.nnls(np.c_[x,np.ones(len(x))]*weights[:,None],y*weights)[0]
    return float(gain),float(floor),bins


def calibrate(acquisition,images,excluded_images,*,bootstrap=100,seed=20261015):
    images=np.asarray(images,dtype=float)
    backgrounds,sigmas=background_noise(images)
    usable=acquisition.acquired&~np.isin(np.arange(len(images))+1,excluded_images)
    records=[];pairs=[]
    for plane in np.unique(acquisition.plane_index):
        for base in ['reference','aberrated_zero']:
            names=[base+'_before' if base=='reference' else base,base+'_after']
            indices=[np.flatnonzero(usable&(acquisition.plane_index==plane)&(acquisition.names==name)) for name in names]
            if any(len(i)!=1 for i in indices):continue
            i,j=[int(ids[0]) for ids in indices]
            if not np.allclose(acquisition.total[i],acquisition.total[j],rtol=0,atol=1e-12):
                raise ValueError('Repeated image commands differ.')
            u,v=images[[i,j]]-backgrounds[[i,j],None,None]
            signal=ndimage.gaussian_filter((u+v)/2,1.)
            difference=(u-v)/np.sqrt(2)
            gx,gy=np.gradient(signal);gxx,gxy=np.gradient(gx);_,gyy=np.gradient(gy)
            design=np.stack([signal,gx,gy,gxx,gxy,gyy],axis=-1).reshape(-1,6)
            design/=np.maximum(np.linalg.norm(design,axis=0),1.)
            weights=np.ones(design.shape[0]);target=difference.ravel()
            for _ in range(8):
                coefficients=np.linalg.lstsq(design*np.sqrt(weights[:,None]),target*np.sqrt(weights),rcond=None)[0]
                residual=target-design@coefficients
                noise=max(float(np.median(abs(residual-np.median(residual)))/MAD_NORMAL),1.)
                weights=np.minimum(1.,2.5*noise/np.maximum(abs(residual),1e-12))
            residual=residual.reshape(signal.shape)
            stable=np.hypot(gx,gy)<.1*np.maximum(signal,30.)
            records.append(dict(signal=signal[stable],residual=residual[stable]))
            pairs.append(dict(first=i+1,second=j+1,plane=int(plane),kind=base,pixels=int(stable.sum())))
    if len(records)<3:raise ValueError('Need at least three usable repeated-command pairs.')
    gain,floor,bins=_variance_fit(records)
    rng=np.random.default_rng(seed);gains=[]
    for _ in range(bootstrap):
        sample=[records[i] for i in rng.integers(0,len(records),len(records))]
        try:gains.append(_variance_fit(sample)[0])
        except ValueError:pass
    pd=usable&acquisition.is_pd
    return dict(effective_poisson_gain=gain,fitted_repeat_gaussian_variance=floor,
        gaussian_sigma_by_image=sigmas,background_by_image=backgrounds,
        gaussian_sigma_pd_quantiles=np.quantile(sigmas[pd],[.1,.5,.9]),pairs=pairs,variance_bins=bins,
        gain_bootstrap_5_95=np.quantile(gains,[.05,.95]) if gains else [],bootstrap_pairs=len(gains),
        calibration_resolution=list(images.shape[-2:]),
        model='Y = clean + gain*(Poisson(signal/gain)-signal/gain) + Normal(0, sigma_image)',
        variance='gain * positive fluorescence signal + sigma_image**2',
        units='Effective variance parameters of saved 64x64 averaged images; do not divide again by exposures/bin area.',
        nuisance_treatment='Robust smooth brightness and spatial-derivative regression on repeat differences only; no image warping.',
        limitations='Approximate repeat-based moment fit. Residual drift, bleaching and structured noise may affect gain. Independent noise does not reproduce stripes/artifacts.')


def add_noise(clean,background,gain,sigmas,*,seed=20261015):
    clean=np.asarray(clean,dtype=float);sigmas=np.asarray(sigmas,dtype=float)
    if clean.ndim!=3 or sigmas.shape!=(len(clean),) or not np.isfinite(clean).all():
        raise ValueError('Require finite image stacks and one Gaussian sigma per image.')
    if gain<0 or not np.isfinite(gain) or np.any(sigmas<0) or not np.isfinite(sigmas).all():
        raise ValueError('Noise parameters must be finite and nonnegative.')
    signal=np.maximum(clean-background,0.)
    rng=np.random.default_rng(seed);noisy=clean.copy()
    if gain>0:noisy+=gain*rng.poisson(signal/gain)-signal
    noisy+=rng.normal(size=clean.shape)*sigmas[:,None,None]
    return noisy
