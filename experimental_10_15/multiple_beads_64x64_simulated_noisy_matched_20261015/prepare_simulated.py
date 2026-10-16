"""Freeze one empirically noise-matched realization of the previous simulation."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from scipy.io import loadmat,savemat
from experimental_10_15.bead_pd_io import load_acquisition,block_average,write_json
from experimental_10_15.empirical_noise import calibrate,add_noise
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.reference_bead_simulation import FLAGGED

HERE=Path(__file__).resolve().parent
SEED=20261015


def prepare():
    clean_folder=HERE.parent/'multiple_beads_64x64_simulated_matched_20261015/simulated_acquisition'
    if not (clean_folder/'acquisition.mat').exists():
        from experimental_10_15.multiple_beads_64x64_simulated_matched_20261015.prepare_simulated import prepare as prepare_clean
        prepare_clean()
    folder=HERE/'simulated_acquisition';folder.mkdir(parents=True,exist_ok=True)
    clean_hash=hashlib.sha256((clean_folder/'acquisition.mat').read_bytes()).hexdigest()
    experiment_hash=hashlib.sha256((SOURCE/'acquisition.mat').read_bytes()).hexdigest()
    provenance_path=folder/'noise_provenance.json'
    if provenance_path.exists() and (folder/'acquisition.mat').exists():
        previous=json.loads(provenance_path.read_text())
        if (previous['seed']!=SEED or previous['clean_sha256']!=clean_hash
                or previous['experimental_sha256']!=experiment_hash or previous['excluded_images']!=FLAGGED):
            raise ValueError('Cached noisy realization has different sources/settings; use a separate batch.')
        return folder
    acquisition=load_acquisition(SOURCE)
    measured=block_average(acquisition.images,64)[:,::-1,:]
    calibration=calibrate(acquisition,measured,FLAGGED,seed=SEED)
    write_json(folder/'noise_calibration.json',calibration)
    clean=load_acquisition(clean_folder)
    reference_fit=json.loads((clean_folder/'reference_fit.json').read_text())
    noisy=add_noise(clean.images,reference_fit['background'],calibration['effective_poisson_gain'],
                    calibration['gaussian_sigma_by_image'],seed=SEED)
    meta=loadmat(clean_folder/'metadata.mat',simplify_cells=True)['metadata']
    meta['runId']=str(meta['runId'])+'_noise_'+str(SEED)
    meta['noiseModel']='Empirical post-binning Poisson/Gaussian; see noise_calibration.json'
    savemat(folder/'metadata.mat',dict(metadata=meta),do_compression=True,long_field_names=True)
    logs=np.array([dict(acquired=True,name=str(name),dataOrigin='simulation') for name in clean.names],dtype=object)
    savemat(folder/'acquisition.mat',dict(images=noisy.transpose(2,1,0),log=logs),do_compression=True,long_field_names=True)
    for name in ['source_metadata.mat','reference_fit.json','simulation_provenance.json']:
        shutil.copy2(clean_folder/name,folder/name)
    provenance=json.loads((folder/'simulation_provenance.json').read_text())
    provenance['noise']='Effective Poisson–Gaussian matched to repeat differences and per-frame background noise; seed '+str(SEED)
    write_json(folder/'simulation_provenance.json',provenance)
    write_json(provenance_path,dict(seed=SEED,clean_sha256=clean_hash,experimental_sha256=experiment_hash,
        clean_source=str(clean_folder.resolve()),experimental_source=str(SOURCE.resolve()),excluded_images=FLAGGED,
        calibration_file='noise_calibration.json',mean='Unchanged noiseless prediction and fitted signed background.',
        sampling='Noise added once at final64 averaged-image resolution; no clipping or second flip.',
        realization_sha256=hashlib.sha256(noisy.tobytes()).hexdigest()))
    np.testing.assert_array_equal(load_acquisition(folder).images,noisy)
    print('Frozen noise realization:',calibration['effective_poisson_gain'],
          'effective gain;',calibration['gaussian_sigma_pd_quantiles'],'Gaussian sigma 10/50/90%',flush=True)
    return folder
