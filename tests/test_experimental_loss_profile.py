"""Truth scaling preserves known cumulative/diversity offsets and profiles the object."""
import tempfile
from pathlib import Path
import unittest
import json
import importlib.util
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from tests.test_experimental_bead_pd import fixture
from experimental_10_15.bead_pd import prepare_run
from experimental_10_15.profile_loss import profile_case


class LossProfileTests(unittest.TestCase):
    def test_probe_filter_preserves_baseline_and_higher_order_modes(self):
        script=Path(__file__).resolve().parents[1]/'experimental_10_15/multiple_beads_z_probe_loss_profiles_20261015/manage.py'
        spec=importlib.util.spec_from_file_location('subset_profile_test',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modes=np.array([[-1,3],[3,3],[0,4],[1,5]])
        baseline=np.array([.1,.2,.3,.4])
        probes=np.vstack([np.zeros(4),.1*np.eye(4)])
        data=dict(modes=modes,baseline=baseline,diversity_strengths=np.tile(probes,(9,1))+baseline,
                  focal_z=np.repeat(np.arange(-2,2.1,.5)/1000,5))
        for limit,full,filtered in [(2.,45,27),(1.5,35,21),(1.,25,15)]:
            self.assertEqual(len(module.select_images(data,limit,False)),full)
            indices=module.select_images(data,limit,True)
            self.assertEqual(len(indices),filtered)
            self.assertEqual(set(indices%5),{0,3,4})

    def test_saved_estimate_is_reevaluated_on_subset_without_comparing_old_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            results=root/'results';results.mkdir()
            with np.load(path) as source:
                values={key:source[key] for key in ['modes','focal_z','cumulative_correction','imposed_for_validation']}
            strengths=np.array([.05,-.03])
            np.savez(results/'solution.npz',**values,sample_z=np.linspace(-.002,.002,3),estimated_strengths=strengths)
            (results/'result.json').write_text(json.dumps(dict(rho=1e46,sample_planes=3,sample_extent_um=4.,random_starts=100)))
            (results/'optimizer.json').write_text(json.dumps(dict(history=[dict(strengths=strengths.tolist(),loss=999.)])))
            captured={}
            class Workspace:
                def __init__(self,*args,**kwargs):captured['images']=args[2];captured['focus']=args[3]
                def loss(self,point):return float(1+np.sum(point**2))
                def screen_losses(self,candidates,on_completed=None):return np.array([self.loss(point) for point in candidates])
            references=[dict(label='Saved estimate',results=results)]
            with patch('experimental_10_15.profile_loss.reconstruction.ReconstructionWorkspace',Workspace):
                data=profile_case(path,references,fractions=[0,1],backend='cpu',image_indices=[0,2,3])
                self.assertEqual(len(captured['images']),3)
                np.testing.assert_array_equal(captured['focus'],values['focal_z'][[0,2,3]])
                self.assertFalse(data['references'][0]['objective_matches_original'])
                self.assertIsNone(data['references'][0]['relative_replay_error'])
                self.assertAlmostEqual(data['references'][0]['loss'],1+np.sum(strengths**2))
                with self.assertRaisesRegex(ValueError,'saved optimizer objective'):
                    profile_case(path,references,fractions=[0,1],backend='cpu')

    def test_fraction_scales_only_truth_and_reports_sampled_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,_=fixture(root)
            cumulative=np.array([.03,-.02])
            meta['cumulativeCorrectionWaves']=cumulative
            meta['totalCommandWaves']+=meta['isPDImage'][:,None]*cumulative
            savemat(root/'metadata.mat',dict(metadata=meta))
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            target=.75*meta['imposedWaves']+cumulative
            captured={}
            class Workspace:
                def __init__(self,*args,**kwargs):captured['known']=kwargs['diversities']
                def loss(self,point):return 1.+np.sum((point-target)**2)
                def screen_losses(self,candidates,on_completed=None):
                    captured['candidates']=candidates.copy()
                    return np.array([self.loss(point) for point in candidates])
            with patch('experimental_10_15.profile_loss.reconstruction.ReconstructionWorkspace',Workspace):
                result=profile_case(path,[],fractions=[-.5,.5,.75,1.5],backend='cpu')
            self.assertEqual(result['best_fraction'],.75)
            np.testing.assert_allclose(captured['candidates'],
                np.asarray(result['fractions'])[:,None]*meta['imposedWaves']+cumulative)
            np.testing.assert_allclose(captured['known'][0].strengths,meta['baselineWaves'])
            truth_index=np.flatnonzero(result['fractions']==1)[0]
            self.assertEqual(result['loss_over_truth'][truth_index],1.)
            self.assertTrue(all(row['relative_error']==0 for row in result['canonical_checks']))

    def test_zero_imposed_wavefront_is_not_a_meaningful_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,_=fixture(root)
            meta['totalCommandWaves']-=meta['isPDImage'][:,None]*meta['imposedWaves']
            meta['imposedWaves']=np.zeros(2)
            savemat(root/'metadata.mat',dict(metadata=meta))
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            with self.assertRaisesRegex(ValueError,'constant'):
                profile_case(path,[],backend='cpu')


if __name__=='__main__':unittest.main()
