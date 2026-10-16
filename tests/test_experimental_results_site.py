"""Pending and completed local result navigation with exact batch metadata."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
from html.parser import HTMLParser
import numpy as np
from experimental_10_15.results_site import build_site


class Scripts(HTMLParser):
    def __init__(self):super().__init__();self.active=False;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=='script' and dict(attrs).get('id')=='site-data':self.active=True
    def handle_data(self,data):
        if self.active:self.parts.append(data)
    def handle_endtag(self,tag):
        if tag=='script':self.active=False


class ResultsSiteTests(unittest.TestCase):
    def fixture(self,base):
        rows=[];jobs=[]
        for size in [64,32]:
            name=f'3d_fluorobeads_spherical_0p15w_{size}x{size}'
            folder=base/name;submission=folder/'submission';submission.mkdir(parents=True)
            np.savez(submission/'input.npz',images=np.zeros((2,size,size)),focal_z=[-.001,0.])
            rows.append(dict(name=name,acquisition='3d_fluorobeads_spherical_0p15w',size=size,
                             z_um=[-2.,-1.,0.,1.,2.]))
            jobs.append(dict(name=name,job_id=str(size),included=2))
        (base/'summary.json').write_text(json.dumps(dict(min_snr=None,runs=rows)))
        (base/'jobs.json').write_text(json.dumps(jobs))
        return rows

    def test_all_pending_is_a_valid_unfiltered_site_with_no_replay_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);self.fixture(base)
            path=build_site(base);text=path.read_text()
            self.assertIn('No SNR filtering',text)
            self.assertIn('All valid acquired Z planes',text)
            self.assertNotIn('SNR ≥ 10',text)
            self.assertNotIn('href="index.html"',text)
            self.assertIn('id="viewer" hidden',text)
            parser=Scripts();parser.feed(text);data=json.loads(''.join(parser.parts))
            self.assertEqual(data['resolutions'],[64,32])
            self.assertTrue(all(not row['ready'] for row in data['cases']))
            self.assertEqual(data['cases'][0]['z_counts'],[0,1,1,0,0])

    def test_one_finished_case_does_not_hide_pending_resolution_or_invent_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);rows=self.fixture(base);folder=base/rows[0]['name']
            results=folder/'solve/results';results.mkdir(parents=True)
            (results/'result.json').write_text(json.dumps(dict(status='complete',optimizer_success=True,elapsed_seconds=7)))
            (results/'optimizer.json').write_text(json.dumps(dict(history=[dict(iteration=3)])))
            figures=folder/'solve/figures';figures.mkdir()
            (figures/'correction_replay.html').write_text('<body>existing shared viewer</body>')
            details=dict(input_path=str((folder/'submission/input.npz').resolve()),source_indices=[1,2],
                         estimation_error_rms_waves=[.15,.04])
            (figures/'correction_replay.json').write_text(json.dumps(details))
            path=build_site(base);data=json.loads((base/'site_data.json').read_text())
            self.assertTrue(data['cases'][0]['ready']);self.assertFalse(data['cases'][1]['ready'])
            self.assertEqual(data['cases'][0]['final_rms'],.04)
            self.assertNotIn('final_rms',data['cases'][1])
            self.assertTrue((base/data['cases'][0]['replay']).is_file())
            self.assertIn('32 × 32',path.read_text())
            details['input_path']=str(base/'another_input.npz')
            (figures/'correction_replay.json').write_text(json.dumps(details))
            with self.assertRaises(AssertionError):build_site(base)

    def test_ready_only_collector_skips_pending_first_job_and_collects_later_completion(self):
        path=Path(__file__).resolve().parents[1]/'experimental_10_15/binning_all_images_20261015_161419/collect_results.py'
        spec=importlib.util.spec_from_file_location('binning_collector',path)
        collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);rows=self.fixture(base)
            jobs=json.loads((base/'jobs.json').read_text())
            for j in jobs:j.update(host='user@cluster',remote_dir='/remote/'+j['job_id'],receipt='/receipt/'+j['job_id'])
            (base/'jobs.json').write_text(json.dumps(jobs))
            states=[dict(name=rows[0]['name'],status='running',complete=False),
                    dict(name=rows[1]['name'],status='complete',complete=True)]
            def fetch(receipt,out):
                results=out/'results';results.mkdir(parents=True)
                (results/'result.json').write_text('{"status":"complete"}')
                (results/'optimizer.json').write_text('{}')
                (results/'solution.npz').write_bytes(b'fixture')
            with patch.object(collector,'HERE',base),patch('sys.argv',[str(path),'--ready-only']), \
                 patch.object(collector,'remote_command',return_value=json.dumps(states)), \
                 patch.object(collector,'watch_job') as watch, \
                 patch.object(collector,'fetch_job',side_effect=fetch) as download, \
                 patch.object(collector,'export_replay') as replay,patch.object(collector,'build_site') as site:
                collector.main()
            watch.assert_not_called()
            download.assert_called_once_with(jobs[1]['receipt'],base/jobs[1]['name']/'solve')
            replay.assert_called_once()
            site.assert_called_with(base)


if __name__=='__main__':unittest.main()
