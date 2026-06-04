"""
run_v2_fix_compas.py — Re-run COMPAS with correct pos_weight.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from aens_framework import (
    AENSModel, load_compas_dataset,
    create_data_splits, train_model, evaluate_model,
    extract_latent_Z, classify_on_Z_experiment, _ListShim,
)
from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import ThresholdOptimizer

V2_ARCH = dict(d_model=128, nhead=8, num_encoder_layers=4,
               dim_feedforward=256, classifier_hidden_dim=64,
               decoder_hidden_dim=64, decoder_depth=1, dropout=0.1)
V2_TRAIN = dict(num_epochs=150, learning_rate=1e-3, weight_decay=1e-5,
                v_learning_rate=0.01, patience=15,
                warmup_epochs=5, gradual_warmup=True, use_cosine_annealing=True)
TOLERANCE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = 'AE_NS_Results_v2'
SEEDS = [42, 123, 456]


def make_loader(X, y, a, shuffle, batch_size=64):
    n = X.shape[0]
    def iterator():
        perm = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            yield {'features': X[idx], 'labels': y[idx], 'protected_attributes': a[idx]}
    return _ListShim(iterator)


def compute_pos_weight(labels):
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


features, labels, protected, info = load_compas_dataset()
pw = compute_pos_weight(labels)
print(f'COMPAS pos_weight = {pw:.2f}')

# === 1. AE-NS v2 standalone (3 seeds) ===
print('\n=== COMPAS AE-NS v2 standalone (3 seeds) ===')
per_seed = []
for seed in SEEDS:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds, val_ds, test_ds = create_data_splits(features, labels, protected, random_state=seed)
    Xt = train_ds.features.to(DEVICE); yt = train_ds.labels.to(DEVICE); at = train_ds.protected_attributes.to(DEVICE)
    Xv = val_ds.features.to(DEVICE);  yv = val_ds.labels.to(DEVICE);  av = val_ds.protected_attributes.to(DEVICE)
    Xs = test_ds.features.to(DEVICE); ys = test_ds.labels.to(DEVICE); aas = test_ds.protected_attributes.to(DEVICE)

    model = AENSModel(input_dim=features.shape[1], alpha=1.0, beta=0.05,
                      pos_weight=pw, fairness_tolerance=TOLERANCE,
                      label_smoothing=0.03, **V2_ARCH).to(DEVICE)
    model, _, tt = train_model(
        model, make_loader(Xt, yt, at, True), make_loader(Xv, yv, av, False),
        **V2_TRAIN, device=DEVICE, verbose=False)
    test_loader = make_loader(Xs, ys, aas, False)
    m = evaluate_model(model, test_loader, device=DEVICE)
    ps = {'seed': seed, 'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'],
          'f1_weighted': m.get('f1_weighted', 0), 'dpd': m['dpd'], 'eod': m['eod'],
          'aod': m['aod'], 'auc': m.get('auc', 0), 'nmi': m.get('nmi', 0),
          'pos_rate_0': m.get('pos_rate_group_0', 0), 'pos_rate_1': m.get('pos_rate_group_1', 0),
          'train_time': tt}
    per_seed.append(ps)
    print(f'  seed={seed}: acc={m["accuracy"]:.4f} dpd={m["dpd"]:.4f} nmi={m.get("nmi",0):.4f} pw_rate_0={m.get("pos_rate_group_0",0):.4f} pw_rate_1={m.get("pos_rate_group_1",0):.4f} ({tt:.1f}s)')
agg = {}
for k in ['accuracy', 'f1_macro', 'f1_weighted', 'dpd', 'eod', 'aod', 'auc', 'nmi', 'train_time']:
    vals = [p[k] for p in per_seed if k in p and not np.isnan(p[k])]
    agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
result_v2 = {'dataset': 'compas', 'version': 'v2', 'pos_weight': pw,
             'arch': V2_ARCH, 'training': {**V2_TRAIN, 'label_smoothing': 0.03},
             'per_seed': per_seed, 'aggregate': agg}
with open(os.path.join(OUT, 'v2_compas_results.json'), 'w') as f:
    json.dump(result_v2, f, indent=2)
print(f'  => saved v2_compas_results.json')

# === 2. AE-NS Z + Fairlearn (3 seeds) ===
print('\n=== COMPAS Fairpost (3 seeds) ===')
fp_seeds = []
for seed in SEEDS:
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds, val_ds, test_ds = create_data_splits(features, labels, protected, random_state=seed)
    Xt = train_ds.features.to(DEVICE); yt = train_ds.labels.to(DEVICE); at = train_ds.protected_attributes.to(DEVICE)
    Xv = val_ds.features.to(DEVICE);  yv = val_ds.labels.to(DEVICE);  av = val_ds.protected_attributes.to(DEVICE)
    Xs = test_ds.features.to(DEVICE); ys = test_ds.labels.to(DEVICE); aas = test_ds.protected_attributes.to(DEVICE)

    model = AENSModel(input_dim=features.shape[1], alpha=1.0, beta=0.05,
                      pos_weight=pw, fairness_tolerance=TOLERANCE,
                      label_smoothing=0.03, **V2_ARCH).to(DEVICE)
    model, _, tt = train_model(
        model, make_loader(Xt, yt, at, True), make_loader(Xv, yv, av, False),
        **V2_TRAIN, device=DEVICE, verbose=False)
    test_loader = make_loader(Xs, ys, aas, False)
    m_aens = evaluate_model(model, test_loader, device=DEVICE)

    # Fairpost
    Z_train = extract_latent_Z(model, train_ds.features, device=DEVICE)
    Z_test = extract_latent_Z(model, test_ds.features, device=DEVICE)
    y_train = train_ds.labels.numpy()
    y_test = test_ds.labels.numpy()
    a_test = test_ds.protected_attributes.numpy()
    a_train = train_ds.protected_attributes.numpy()

    to = ThresholdOptimizer(
        estimator=LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced'),
        constraints='demographic_parity', prefit=False, objective='accuracy_score')
    to.fit(Z_train, y_train, sensitive_features=a_train)
    preds = to.predict(Z_test, sensitive_features=a_test)
    acc = float((preds == y_test).mean())
    m0 = a_test == 0; m1 = a_test == 1
    dpd = abs(preds[m0].mean() - preds[m1].mean())
    tpr0 = preds[m0 & (y_test == 1)].mean() if (m0 & (y_test == 1)).sum() else 0
    tpr1 = preds[m1 & (y_test == 1)].mean() if (m1 & (y_test == 1)).sum() else 0
    eod = abs(tpr0 - tpr1)
    fpr0 = preds[m0 & (y_test == 0)].mean() if (m0 & (y_test == 0)).sum() else 0
    fpr1 = preds[m1 & (y_test == 0)].mean() if (m1 & (y_test == 0)).sum() else 0
    aod = 0.5 * (eod + abs(fpr0 - fpr1))

    elapsed = time.time() - t0
    ps = {'seed': seed, 'accuracy': acc, 'dpd': dpd, 'eod': eod, 'aod': aod,
          'aens_accuracy': m_aens['accuracy'], 'aens_dpd': m_aens['dpd'],
          'total_time': elapsed}
    fp_seeds.append(ps)
    print(f'  seed={seed}: FP acc={acc:.4f} dpd={dpd:.4f} | AE-NS acc={m_aens["accuracy"]:.4f} ({elapsed:.1f}s)')

fp_agg = {}
for k in ['accuracy', 'dpd', 'eod', 'aod', 'total_time']:
    vals = [p[k] for p in fp_seeds]
    fp_agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
result_fp = {'dataset': 'compas', 'method': 'AE-NS Z + Fairlearn', 'constraints': 'demographic_parity',
             'pos_weight': pw, 'per_seed': fp_seeds, 'aggregate': fp_agg}
with open(os.path.join(OUT, 'fairpost_3seeds_compas.json'), 'w') as f:
    json.dump(result_fp, f, indent=2)
print(f'  => saved fairpost_3seeds_compas.json')

# === 3. NMI + classify-on-Z (1 seed) ===
print('\n=== COMPAS NMI + classify-on-Z ===')
torch.manual_seed(42); np.random.seed(42)
train_ds, val_ds, test_ds = create_data_splits(features, labels, protected, random_state=42)
Xt = train_ds.features.to(DEVICE); yt = train_ds.labels.to(DEVICE); at = train_ds.protected_attributes.to(DEVICE)
Xv = val_ds.features.to(DEVICE);  yv = val_ds.labels.to(DEVICE);  av = val_ds.protected_attributes.to(DEVICE)
Xs = test_ds.features.to(DEVICE); ys = test_ds.labels.to(DEVICE); aas = test_ds.protected_attributes.to(DEVICE)

model = AENSModel(input_dim=features.shape[1], alpha=1.0, beta=0.05,
                  pos_weight=pw, fairness_tolerance=TOLERANCE,
                  label_smoothing=0.03, **V2_ARCH).to(DEVICE)
model, _, tt = train_model(
    model, make_loader(Xt, yt, at, True), make_loader(Xv, yv, av, False),
    **V2_TRAIN, device=DEVICE, verbose=False)
test_loader = make_loader(Xs, ys, aas, False)
m = evaluate_model(model, test_loader, device=DEVICE)
cz = classify_on_Z_experiment(
    model, train_ds.features, train_ds.labels, train_ds.protected_attributes,
    test_ds.features, test_ds.labels, test_ds.protected_attributes,
    device=DEVICE, random_state=42)
nmi = m.get('nmi', 0)
result_rev = {
    'dataset': 'compas', 'seed': 42, 'epochs': 150, 'warmup': 5,
    'tolerance': TOLERANCE, 'pos_weight': pw, 'train_time_s': tt,
    'n_z': int(cz['n_z']), 'n_x': int(cz['n_x']), 'nmi': nmi,
    'classify_on_Z': cz,
    'eval_summary': {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
}
with open(os.path.join(OUT, 'compas_revision.json'), 'w') as f:
    json.dump(result_rev, f, indent=2)
print(f'  NMI={nmi:.4f} acc_on_Z={cz["acc_on_Z"]:.4f} acc_on_X={cz["acc_on_X"]:.4f}')
print(f'  => saved compas_revision.json')

# skip if already done
if os.path.exists(os.path.join(OUT, 'compas_ablation.json')):
    with open(os.path.join(OUT, 'compas_ablation.json')) as f:
        existing = json.load(f)
    if 'Full AE-NS' in existing and 'w/o Reconstruction' in existing:
        print('\n=== COMPAS ablation already done, skipping ===')
    else:
        pass  # fall through

# === 4. Ablation (1 seed) ===
print('\n=== COMPAS ablation (1 seed) ===')
variants = {
    'Full AE-NS': {},
    'w/o Reconstruction': {'beta': 0.0, 'use_reconstruction': False},
    'w/o Orthogonality': {'use_orthogonality': False},
    'w/o Lagrangian (Fixed Penalty)': {'use_lagrangian': False, 'initial_v': 1.0},
    'MLP Encoder': {'encoder_type': 'mlp'},
    'Decoder Depth=2': {'decoder_depth': 2},
    'Decoder Depth=0 (Linear)': {'decoder_depth': 0},
}
abl_results = {}
for name, extra in variants.items():
    torch.manual_seed(42); np.random.seed(42)
    train_ds2, val_ds2, test_ds2 = create_data_splits(features, labels, protected, random_state=42)
    Xt2 = train_ds2.features.to(DEVICE); yt2 = train_ds2.labels.to(DEVICE); at2 = train_ds2.protected_attributes.to(DEVICE)
    Xv2 = val_ds2.features.to(DEVICE);  yv2 = val_ds2.labels.to(DEVICE);  av2 = val_ds2.protected_attributes.to(DEVICE)
    Xs2 = test_ds2.features.to(DEVICE); ys2 = test_ds2.labels.to(DEVICE); aas2 = test_ds2.protected_attributes.to(DEVICE)

    model_kwargs = dict(input_dim=features.shape[1], alpha=1.0, beta=0.05,
                        pos_weight=pw, fairness_tolerance=TOLERANCE,
                        label_smoothing=0.03, **V2_ARCH)
    model_kwargs.update(extra)
    model = AENSModel(**model_kwargs).to(DEVICE)
    model, _, tt2 = train_model(
        model, make_loader(Xt2, yt2, at2, True), make_loader(Xv2, yv2, av2, False),
        **V2_TRAIN, device=DEVICE, verbose=False)
    test_loader2 = make_loader(Xs2, ys2, aas2, False)
    m2 = evaluate_model(model, test_loader2, device=DEVICE)
    abl_results[name] = {
        'accuracy': {'mean': m2['accuracy'], 'std': 0.0, 'values': [m2['accuracy']]},
        'f1_weighted': {'mean': m2.get('f1_weighted', 0), 'std': 0.0, 'values': [m2.get('f1_weighted', 0)]},
        'dpd': {'mean': m2['dpd'], 'std': 0.0, 'values': [m2['dpd']]},
        'eod': {'mean': m2['eod'], 'std': 0.0, 'values': [m2['eod']]},
        'aod': {'mean': m2['aod'], 'std': 0.0, 'values': [m2['aod']]},
        'nmi': {'mean': m2.get('nmi', 0), 'std': 0.0, 'values': [m2.get('nmi', 0)]},
        'train_time': {'mean': tt2, 'std': 0.0, 'values': [tt2]},
    }
    print(f'  {name}: acc={m2["accuracy"]:.4f} dpd={m2["dpd"]:.4f}')
with open(os.path.join(OUT, 'compas_ablation.json'), 'w') as f:
    json.dump(abl_results, f, indent=2)
print(f'  => saved compas_ablation.json')

# === 5. t-SNE data ===
print('\n=== COMPAS t-SNE data ===')
Z = extract_latent_Z(model, test_ds.features.to(DEVICE), device=DEVICE)
X = test_ds.features.numpy()
A = test_ds.protected_attributes.numpy()
np.savez(os.path.join(OUT, 'tsne_data_compas.npz'), Z=Z, X=X, A=A)
print(f'  Z={Z.shape} X={X.shape}')

print('\nAll COMPAS fixes complete!')
