# Job B (GPU array task, one task per model). Inference only -- no training, no new
# checkpoints. Reloads an existing try23/try24 checkpoint and scores every target disease
# node with it. For the targets this model did NOT hold out, the indication edge was in its
# training graph (job A verified that edge by edge), so the resulting rank is the UNBLINDED
# number: what the model says when it has already been shown the answer.
#
# Node INPUT vectors are random, frozen, stored on the graph (utils.py initialize_node_
# embedding) and NOT saved in the checkpoint, so a reload only reproduces the trained model
# if torch's RNG is in the same state when they are drawn. The training script re-seeds
# immediately before TxGNN(), so this does exactly the same and then VERIFIES it by
# re-scoring the run's own held-out disease and comparing to that run's saved eval pkl.
# Same guard try19 used; without it a silently different model would be reported.
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from txgnn import TxData, TxGNN, TxEval

from config26 import DATA_DIR, EDGE_AUDIT, MODELS, ORDER, OUT, SEEDS, TARGETS

TOL = 1e-3


def set_seed(seed):
    # TxGNN.py hardcodes torch.manual_seed(0) at import -- override per seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scoreable_nodes(audit, model):
    # Skip nodes PrimeKG has no indication edge for at all (TTR 4803/5528): nothing to rank,
    # and eval_disease_centric has no label to build for them.
    a = audit[audit.model == model]
    ok = a.groupby('disease_idx').edge_in.apply(lambda s: (s != 'absent').any())
    return sorted(float(i) for i in ok[ok].index)


def main():
    model_name = ORDER[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    root, holdout = MODELS[model_name]
    audit = pd.read_csv(EDGE_AUDIT)
    nodes = scoreable_nodes(audit, model_name)
    # the model's own held-out node, used for the reproduction check
    own = sorted(i for i in nodes if TARGETS[i][0] == model_name and i in [float(h) for h in holdout])
    if not own:
        raise SystemExit(f'STOP: no own held-out node for {model_name} -- cannot verify the reload')
    check_idx = own[0]
    print(f'[{model_name}] holdout={holdout}  scoring {len(nodes)} target nodes  '
          f'reproduction check on {check_idx}', flush=True)

    for seed in SEEDS:
        sdir = os.path.join(root, model_name, f'seed_{seed}')
        out_path = os.path.join(OUT, model_name, f'seed_{seed}', 'targets.pkl')
        if os.path.exists(out_path):
            print(f'[{model_name} s{seed}] already done, skipping', flush=True)
            continue
        ck = os.path.join(sdir, 'ckpt')
        if not os.path.exists(os.path.join(ck, 'model.pt')):
            raise SystemExit(f'STOP: no checkpoint at {ck}')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        set_seed(seed)
        d = TxData(data_folder_path=DATA_DIR)
        d.prepare_split(disease_eval_idx=holdout, seed=seed)   # cached split, same as training
        set_seed(seed)   # same re-seed point as the training script, right before TxGNN()
        model = TxGNN(data=d, weight_bias_track=False, proj_name='TxGNN',
                      exp_name=f'{model_name}_unblind_s{seed}', device='cuda:0')
        model.load_pretrained(ck)
        out = TxEval(model=model).eval_disease_centric(disease_idxs=nodes, relation='indication',
                                                       return_raw=True, simulate_random=False)

        # ---- reproduction check against this run's own saved scores ----
        saved = pickle.load(open(os.path.join(sdir, f'eval_{int(check_idx)}.pkl'), 'rb'))
        (key, orig), = saved['prediction'].items()
        new = out['prediction'].get(key)
        if new is None:
            raise RuntimeError(f'{model_name} s{seed}: key {key!r} missing from the new output')
        diff = max(abs(new[x] - orig[x]) for x in orig)
        print(f'[{model_name} s{seed}] reproduction check on {check_idx}: '
              f'max |score diff| = {diff:.2e}', flush=True)
        if diff > TOL:
            raise RuntimeError(f'{model_name} s{seed}: reloaded model does NOT reproduce its '
                               'saved scores -- every unblinded number from it would be invalid')

        # Atomic write. An A100 copy and a preemptible A40 copy of this array may both be
        # queued (see README): the skip-if-done check above means they normally divide the
        # work, but if both reach the same seed at once a plain open() could leave a
        # half-written pickle. Write to a per-process temp file and rename -- a rename is
        # atomic, so a reader never sees a partial file and last-writer-wins is harmless
        # (the output is deterministic and the reproduction check above proves it).
        tmp = f'{out_path}.tmp.{os.getpid()}'
        with open(tmp, 'wb') as f:
            pickle.dump({'model': model_name, 'seed': seed, 'holdout': holdout,
                         'nodes': nodes, 'prediction': out['prediction']}, f)
        os.replace(tmp, out_path)
        print(f'[{model_name} s{seed}] saved {out_path}', flush=True)
    print(f'{model_name}: all seeds done')


if __name__ == '__main__':
    main()
