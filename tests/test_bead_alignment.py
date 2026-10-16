"""Independent motion fixtures for reference-only acquisition preprocessing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.io import savemat
from scipy.ndimage import shift

from experimental_10_15.align_stack import register_translation,align_run,render_alignment_preview
from experimental_10_15.bead_pd_io import load_acquisition
from experimental_10_15.bead_pd import prepare_run
from experimental_10_15.submit_bead_pd import bundle_run


def bead_field():
    x,y=np.indices((64,64))
    return (100*np.exp(-((x-21.)**2+(y-26.)**2)/8)+
            70*np.exp(-((x-43.)**2+(y-40.)**2)/12)+
            35*np.exp(-((x-40.)**2+(y-16.)**2)/5)-12)


def acquisition_fixture(root):
    names=np.array(['reference_before','aberrated_zero','probe','aberrated_zero_after','reference_after']*2,dtype=object)
    is_pd=np.tile([False,True,True,True,False],2)
    diversity=np.zeros((10,2));diversity[[2,7],0]=.1
    imposed=np.array([.03,.06]);baseline=np.zeros(2)
    optics=dict(excitationWavelengthNm=1300,nonlinearOrder=3,NA=1.05,refractiveIndex=1.333,
        focalLengthMm=7.2,magnification=4,beamWaistMm=3.5,backFocalPlaneWidthMm=15.12,
        fieldOfViewUm=6.5,beadDiameterUm=1,focalPositionUm=0)
    metadata=dict(config=dict(imageSize=[64,64],optics=optics,analysis=dict(size=32,pupilSize=8)),
        coefficientUnits='waves',modes_mn=np.array([[2,2],[0,4]]),baselineWaves=baseline,
        imposedWaves=imposed,diversityWaves=diversity,totalCommandWaves=diversity+is_pd[:,None]*imposed,
        isPDImage=is_pd,names=names,planeIndex=np.repeat([1,2],5),
        focalPositionsMm=np.repeat([0,.001],5),status='complete')
    fractions=np.array([0,1,4,7,8])/8
    images=[];logs=[]
    for plane in range(2):
        base=np.array([0,0]) if plane==0 else np.array([3,-2])
        for j,fraction in enumerate(fractions):
            image=shift(bead_field(),base+fraction*np.array([2,-1]),order=3,mode='constant',cval=-12)
            if j==2:
                # A probe-specific displacement must NOT determine registration.
                image=shift(image,[4,0],order=1,mode='constant',cval=-12)
            images.append(image)
            logs.append(dict(acquired=True,name=str(names[plane*5+j]),
                finishedAt=f'20261015T1200{plane*20+int(fraction*8):02d}',filename='source.tif'))
    images=np.array(images)
    savemat(root/'metadata.mat',dict(metadata=metadata),long_field_names=True)
    savemat(root/'acquisition.mat',dict(images=images.transpose(2,1,0),log=np.array(logs,dtype=object)))
    return images,fractions


class AlignmentTests(unittest.TestCase):
    def test_subpixel_sign_gain_and_background(self):
        reference=bead_field()
        motion=np.array([2.35,-1.65])
        moving=2.1*shift(reference,motion,order=3,mode='constant',cval=-12)+50
        applied,quality=register_translation(reference,moving,max_shift=6)
        np.testing.assert_allclose(applied,-motion,atol=.08)
        self.assertGreater(quality['correlation_after'],.99)
        self.assertGreater(quality['correlation_after'],quality['correlation_before'])

    def test_reference_drift_is_interpolated_without_registering_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'raw';source.mkdir()
            raw,fractions=acquisition_fixture(source)
            before=(source/'acquisition.mat').read_bytes()
            out=align_run(source,root/'aligned',max_shift_pixels=6)
            with np.load(out/'alignment.npz') as d:
                np.testing.assert_allclose(d['shifts_xy_pixels'][:5],fractions[:,None]*[-2,1],atol=.02)
                np.testing.assert_allclose(d['shifts_xy_pixels'][5:],fractions[:,None]*[-2,1],atol=.02)
                self.assertFalse(d['valid_mask'][4,-1,0])
                self.assertTrue(d['common_valid_mask'][32,32])
            loaded=load_acquisition(out)
            self.assertEqual(loaded.images.shape,(10,64,64))
            self.assertLess(loaded.images.min(),0)
            np.testing.assert_allclose(loaded.images[0],raw[0])
            # Per-Z references are independent by default.
            np.testing.assert_allclose(loaded.images[5],raw[5])
            # Probe-specific motion remains; reference drift alone is removed.
            self.assertGreater(np.linalg.norm(loaded.images[2]-loaded.images[0]),10)
            self.assertEqual(before,(source/'acquisition.mat').read_bytes())
            self.assertEqual(loaded.optics['fieldOfViewUm'],6.5)
            path=prepare_run(out,root/'prepared')
            with np.load(path) as d:self.assertEqual(d['images'].shape,(6,32,32))
            report=json.loads((out/'alignment.json').read_text())
            self.assertIn('timestamps',report['planes'][0]['interpolation_time_source'])
            with self.assertRaises(ValueError):align_run(source,out)
            with self.assertRaises(ValueError):align_run(source,source)
            with self.assertRaisesRegex(ValueError,'already an aligned'):align_run(out,root/'double')

    def test_optional_global_anchor_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'raw';source.mkdir();acquisition_fixture(source)
            out=align_run(source,root/'global',align_planes=True,anchor_plane=1,max_shift_pixels=6)
            with np.load(out/'alignment.npz') as d:
                np.testing.assert_allclose(d['shifts_xy_pixels'][5],[-3,2],atol=.02)
                np.testing.assert_allclose(d['shifts_xy_pixels'][9],[-5,3],atol=.02)
            excluded=align_run(source,root/'excluded',exclude_planes=[2],max_shift_pixels=6)
            a=load_acquisition(excluded)
            self.assertEqual(a.acquired.sum(),5)
            self.assertTrue(np.all(np.isnan(a.images[5:])))
            archive,_=bundle_run(excluded,root/'bundle',size=32,pupil_size=8)
            with zipfile.ZipFile(archive) as z:
                for name in ['alignment.json','alignment.npz','alignment_shifts.csv','source_metadata.mat']:
                    self.assertIn('original_acquisition/'+name,z.namelist())

    def test_preview_and_quantized_timestamp_fallback(self):
        from experimental_10_15.align_stack import _interpolation_positions
        fraction,method=_interpolation_positions([dict(finishedAt='20261015T120000')]*5,
                                                np.arange(5),0,4)
        np.testing.assert_allclose(fraction,np.linspace(0,1,5))
        self.assertIn('acquisition index',method)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'raw';source.mkdir();acquisition_fixture(source)
            with patch('experimental_10_15.align_stack.render_alignment_preview') as preview:
                output=align_run(source,root/'aligned',max_shift_pixels=6,preview=True,use_tex=False)
                preview.assert_called_once_with(source.resolve(),output,use_tex=False)

    def test_invalid_registration_and_missing_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'contrast'):
            register_translation(np.ones((64,64)),np.ones((64,64)))
        with self.assertRaisesRegex(ValueError,'bound'):
            register_translation(bead_field(),shift(bead_field(),[8,0],cval=-12),max_shift=4)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);acquisition_fixture(root)
            from scipy.io import loadmat
            d=loadmat(root/'acquisition.mat',simplify_cells=True)
            d['log'][4]['acquired']=False
            savemat(root/'acquisition.mat',{k:v for k,v in d.items() if not k.startswith('__')},long_field_names=True)
            with self.assertRaisesRegex(ValueError,'reference_after'):
                align_run(root,root/'aligned',max_shift_pixels=6)
            self.assertFalse((root/'aligned').exists())


if __name__=='__main__':unittest.main()
