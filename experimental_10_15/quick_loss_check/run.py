"""Reevaluate truth, zero and saved fit with the original profiled objective."""
from pathlib import Path
from experimental_10_15.profile_loss import profile_case
from experimental_10_15.bead_pd_io import write_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    for label,batch in [('experimental','multiple_beads_64x64_flipx_artifacts_excluded_20261015'),
                        ('noisy_simulation','multiple_beads_64x64_simulated_noisy_matched_20261015')]:
        folder=ROOT/batch/'multiple_beads_64x64'
        print(label,flush=True)
        data=profile_case(folder/'submission/input.npz',
            [dict(label='Optimizer estimate',results=folder/'solve/results')],fractions=[0.,1.],backend='cpu',
            progress=lambda n:print(f'  evaluated {n}',flush=True))
        write_json(Path(__file__).parent/(label+'.json'),data)
        print(label, 'zero/truth:',data['loss_over_truth'][0],
              'estimate/truth:',data['references'][0]['loss_over_truth'],flush=True)

if __name__=='__main__':main()
