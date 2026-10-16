"""Package the existing optical-coordinate simulation without applying a second flip."""
from pathlib import Path
import hashlib
import shutil
import numpy as np
from scipy.io import loadmat,savemat
from experimental_10_15.bead_pd_io import load_acquisition,write_json
from experimental_10_15.plot_many_bead_images import SOURCE

HERE=Path(__file__).resolve().parent


def prepare():
    simulation=HERE.parent/'reference_bead_orientation_scan/flip_x'
    folder=HERE/'simulated_acquisition';folder.mkdir(exist_ok=True)
    with np.load(simulation/'comparison.npz',allow_pickle=False) as data:
        images=data['simulated'].copy();indices=data['source_indices'].copy()
        assert str(data['orientation_key'])=='flip_x'
    assert images.shape==(162,64,64)
    np.testing.assert_array_equal(indices,np.arange(162))
    original=load_acquisition(SOURCE)
    meta=loadmat(SOURCE/'metadata.mat',simplify_cells=True)['metadata']
    meta['config']['imageSize']=np.array([64,64])
    meta['config']['analysis']['size']=64
    meta['config']['dataOrigin']='simulation'
    meta['runId']=str(meta.get('runId',SOURCE.name))+'_simulated_flipx'
    meta['status']='complete'
    meta['simulationCoordinates']='Optical model x/y from flip_x reference fit; do not flip predictions again.'
    savemat(folder/'metadata.mat',dict(metadata=meta),do_compression=True,long_field_names=True)
    logs=np.array([dict(acquired=True,name=str(name),dataOrigin='simulation') for name in original.names],dtype=object)
    savemat(folder/'acquisition.mat',dict(images=images.transpose(2,1,0),log=logs),
            do_compression=True,long_field_names=True)
    shutil.copy2(SOURCE/'metadata.mat',folder/'source_metadata.mat')
    shutil.copy2(simulation/'reference_fit.json',folder/'reference_fit.json')
    write_json(folder/'simulation_provenance.json',dict(
        origin='simulation',source_comparison=str((simulation/'comparison.npz').resolve()),
        source_sha256=hashlib.sha256((simulation/'comparison.npz').read_bytes()).hexdigest(),
        raw_acquisition=str(SOURCE.resolve()),
        forward_model='Three reference-fitted uniform 1-um beads; vector 3P; native128 then 2x2 block means64.',
        image_orientation='Predictions already in optical coordinates. No additional flip or transpose.',
        noise='No added noise/artifacts; reference-fitted constant signed background retained.',
        calibration='Positions and brightnesses inferred using references only; all fitting modes/object remain free in optimizer.',
        comparison_limit='Forward simulation uses native128 finite-sphere sampling; inverse uses original64 object grid, so discretization/model mismatch is retained.'))
    reloaded=load_acquisition(folder)
    np.testing.assert_array_equal(reloaded.images,images)
    np.testing.assert_array_equal(reloaded.total,original.total)
    np.testing.assert_array_equal(reloaded.focal_z,original.focal_z)
    return folder
