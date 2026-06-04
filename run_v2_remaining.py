"""
run_v2_remaining.py — Run the 3 missing experiments from run_v2_all.py:
  1. NMI + classify-on-Z (3 datasets, seed=42)
  2. Ablation for credit (1 seed)
  3. t-SNE data extraction (3 datasets, seed=42)
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from aens_framework import (
    AENSModel, load_adult_dataset, load_compas_dataset, load_credit_dataset,
    create_data_splits, train_model, evaluate_model,
    extract_latent_Z, classify_on_Z_experiment, _ListShim,
)

V2_ARCH = dict(d_model=128, nhead=8, num_encoder_layers=4,
               dim_feedforward=256, classifier_hidden_dim=64,
               decoder_hidden_dim=64, decoder_depth=1, dropout=0.1)
V2_TRAIN = dict(num_epochs=150, learning_rate=1e-3, weight_decay=1e-5,
                v_learning_rate=0.01, patience=15,
                warmup_epochs=5, gradual_warmup=True, use_cosine_annealing=True)
TOLERANCE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = 'AE_NS_Results_v2'
POS_WEIGHTS = {'adult': 2.0, 'compas': 2.0, 'credit': 3.0}
LOADERS = {'adult': load_adult_dataset, 'compas': load_compas_dataset, 'credit': load_credit_dataset}


def make_loader(X, y, a, shuffle, batch_size=64):
    n = X.shape[0]
    def iterator():
        perm = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            yield {'features': X[idx], 'labels': y[idx], 'protected_attributes': a[idx]}
    return _ListShim(iterator)


def train_v2(ds_name, features, labels, protected, seed, alpha=1.0, beta=0.05,
             label_smoothing=0.03, extra_kwargs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds, val_ds, test_ds = create_data_splits(features, labels, protected, random_state=seed)
    Xt = train_ds.features.to(DEVICE); yt = train_ds.labels.to(DEVICE); at = train_ds.protected_attributes.to(DEVICE)
    Xv = val_ds.features.to(DEVICE);  yv = val_ds.labels.to(DEVICE);  av = val_ds.protected_attributes.to(DEVICE)
    model_kwargs = dict(V2_ARCH, input_dim=features.shape[1], alpha=alpha, beta=beta,
                        fairness_tolerance=TOLERANCE, pos_weight=POS_WEIGHTS[ds_name],
                        label_smoothing=label_smoothing)
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)
    model = AENSModel(**model_kwargs).to(DEVICE)
    model, history, train_time = train_model(
        model, make_loader(Xt, yt, at, True), make_loader(Xv, yv, av, False),
        **V2_TRAIN, device=DEVICE, verbose=False)
    test_loader = make_loader(test_ds.features.to(DEVICE), test_ds.labels.to(DEVICE),
                              test_ds.protected_attributes.to(DEVICE), False)
    metrics = evaluate_model(model, test_loader, device=DEVICE)
    return model, metrics, train_time, train_ds, test_ds


# 1. NMI + classify-on-Z
print('='*60)
print('  EXPERIMENT 3: NMI + classify-on-Z')
print('='*60)
for ds_name in ['adult', 'compas', 'credit']:
    print(f'\n  [{ds_name}]')
    features, labels, protected, info = LOADERS[ds_name]()
    model, m, tt, train_ds, test_ds = train_v2(ds_name, features, labels, protected, seed=42)
    cz = classify_on_Z_experiment(
        model, train_ds.features, train_ds.labels, train_ds.protected_attributes,
        test_ds.features, test_ds.labels, test_ds.protected_attributes,
        device=DEVICE, random_state=42)
    nmi = m.get('nmi', 0)
    result = {
        'dataset': ds_name, 'seed': 42, 'epochs': 150, 'warmup': 5,
        'tolerance': TOLERANCE, 'train_time_s': tt,
        'n_z': int(cz['n_z']), 'n_x': int(cz['n_x']), 'nmi': nmi,
        'classify_on_Z': cz,
        'eval_summary': {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
    }
    with open(os.path.join(OUT, f'{ds_name}_revision.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'    NMI={nmi:.4f} acc_on_Z={cz["acc_on_Z"]:.4f} acc_on_X={cz["acc_on_X"]:.4f} dpd_on_Z={cz["dpd_on_Z"]:.4f}')
    print(f'    => saved {ds_name}_revision.json')

# 2. Credit ablation (if missing)
if not os.path.exists(os.path.join(OUT, 'credit_ablation.json')):
    print('\n' + '='*60)
    print('  EXPERIMENT 6b: Credit ablation')
    print('='*60)
    features, labels, protected, info = LOADERS['credit']()
    input_dim = features.shape[1]
    pw = POS_WEIGHTS['credit']
    variants = {
        'Full AE-NS': {},
        'w/o Reconstruction': {'beta': 0.0, 'use_reconstruction': False},
        'w/o Orthogonality': {'use_orthogonality': False},
        'w/o Lagrangian (Fixed Penalty)': {'use_lagrangian': False, 'initial_v': 1.0},
        'MLP Encoder': {'encoder_type': 'mlp'},
        'Decoder Depth=2': {'decoder_depth': 2},
        'Decoder Depth=0 (Linear)': {'decoder_depth': 0},
    }
    ds_results = {}
    for name, extra in variants.items():
        torch.manual_seed(42); np.random.seed(42)
        model, m, tt, _, _ = train_v2('credit', features, labels, protected, seed=42,
                                       extra_kwargs={'pos_weight': pw, **extra})
        ds_results[name] = {
            'accuracy': {'mean': m['accuracy'], 'std': 0.0, 'values': [m['accuracy']]},
            'f1_weighted': {'mean': m.get('f1_weighted', 0), 'std': 0.0, 'values': [m.get('f1_weighted', 0)]},
            'dpd': {'mean': m['dpd'], 'std': 0.0, 'values': [m['dpd']]},
            'eod': {'mean': m['eod'], 'std': 0.0, 'values': [m['eod']]},
            'aod': {'mean': m['aod'], 'std': 0.0, 'values': [m['aod']]},
            'nmi': {'mean': m.get('nmi', 0), 'std': 0.0, 'values': [m.get('nmi', 0)]},
            'train_time': {'mean': tt, 'std': 0.0, 'values': [tt]},
        }
        print(f'    {name}: acc={m["accuracy"]:.4f} dpd={m["dpd"]:.4f}')
    with open(os.path.join(OUT, 'credit_ablation.json'), 'w') as f:
        json.dump(ds_results, f, indent=2)
    print(f'    => saved credit_ablation.json')

# 3. t-SNE data extraction
print('\n' + '='*60)
print('  EXPERIMENT 7: t-SNE data extraction')
print('='*60)
for ds_name in ['adult', 'compas', 'credit']:
    print(f'\n  [{ds_name}]')
    features, labels, protected, info = LOADERS[ds_name]()
    model, m, tt, _, test_ds = train_v2(ds_name, features, labels, protected, seed=42)
    Z = extract_latent_Z(model, test_ds.features.to(DEVICE), device=DEVICE)
    X = test_ds.features.numpy()
    A = test_ds.protected_attributes.numpy()
    n = Z.shape[0]
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        Z, X, A = Z[idx], X[idx], A[idx]
    npz_path = os.path.join(OUT, f'tsne_data_{ds_name}.npz')
    np.savez(npz_path, Z=Z, X=X, A=A)
    print(f'    Z={Z.shape} X={X.shape} => {npz_path}')

print('\nAll remaining experiments complete!')
