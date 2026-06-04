"""
run_v2_all.py — Run ALL v2 experiments on GPU for the AE-NS paper revision.

Experiments:
  1. AE-NS v2 baseline (3 seeds) — standalone AE-NS with v2 training
  2. AE-NS Z + Fairlearn post-processing (3 seeds) — extract Z, apply ThresholdOptimizer
  3. NMI + classify-on-Z (1 seed) — disentanglement metrics
  4. Sensitivity alpha sweep (1 seed) — alpha in {0.1, 0.3, 0.5, 0.7, 1.0} at beta=0.05
  5. Sensitivity beta sweep (1 seed) — beta in {0.01, 0.03, 0.05, 0.1, 0.2} at alpha=1.0
  6. Ablation study (1 seed, 7 variants) — remove components
  7. t-SNE data extraction (1 seed) — Z vs X embeddings for visualization

All use v2 config:
  nhead=8, num_encoder_layers=4, label_smoothing=0.03,
  gradual_warmup=True, use_cosine_annealing=True, warmup=5, epochs=150

Saves results to AE_NS_Results_v2/
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from aens_framework import (
    AENSModel, AENSHybridModel,
    load_adult_dataset, load_compas_dataset, load_credit_dataset,
    create_data_splits, train_model, evaluate_model,
    extract_latent_Z, classify_on_Z_experiment,
    _ListShim,
)

SEEDS = [42, 123, 456]
V2_ARCH = dict(d_model=128, nhead=8, num_encoder_layers=4,
               dim_feedforward=256, classifier_hidden_dim=64,
               decoder_hidden_dim=64, decoder_depth=1, dropout=0.1)
V2_TRAIN = dict(num_epochs=150, learning_rate=1e-3, weight_decay=1e-5,
                v_learning_rate=0.01, patience=15,
                warmup_epochs=5, gradual_warmup=True, use_cosine_annealing=True)
TOLERANCE = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = 'AE_NS_Results_v2'
os.makedirs(OUT, exist_ok=True)

LOADERS = {
    'adult':  load_adult_dataset,
    'compas': load_compas_dataset,
    'credit': load_credit_dataset,
}
POS_WEIGHTS = {'adult': 2.0, 'compas': 2.0, 'credit': 3.0}


def make_loader(X, y, a, shuffle, batch_size=64):
    n = X.shape[0]
    def iterator():
        perm = torch.randperm(n, device=DEVICE) if shuffle else torch.arange(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            yield {'features': X[idx], 'labels': y[idx], 'protected_attributes': a[idx]}
    return _ListShim(iterator)


def train_and_eval(ds_name, features, labels, protected, seed, alpha=1.0, beta=0.05,
                   label_smoothing=0.03, extra_kwargs=None):
    """Train AE-NS v2 and return (model, metrics, train_time)."""
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds, val_ds, test_ds = create_data_splits(features, labels, protected, random_state=seed)

    Xt = train_ds.features.to(DEVICE); yt = train_ds.labels.to(DEVICE); at = train_ds.protected_attributes.to(DEVICE)
    Xv = val_ds.features.to(DEVICE);  yv = val_ds.labels.to(DEVICE);  av = val_ds.protected_attributes.to(DEVICE)
    Xs = test_ds.features.to(DEVICE); ys = test_ds.labels.to(DEVICE); aas = test_ds.protected_attributes.to(DEVICE)

    model_kwargs = dict(V2_ARCH, input_dim=features.shape[1], alpha=alpha, beta=beta,
                        fairness_tolerance=TOLERANCE, pos_weight=POS_WEIGHTS[ds_name],
                        label_smoothing=label_smoothing)
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)

    model = AENSModel(**model_kwargs).to(DEVICE)
    model, history, train_time = train_model(
        model, make_loader(Xt, yt, at, True), make_loader(Xv, yv, av, False),
        **V2_TRAIN, device=DEVICE, verbose=False)

    test_loader = make_loader(Xs, ys, aas, False)
    metrics = evaluate_model(model, test_loader, device=DEVICE)
    return model, metrics, train_time, (Xs, ys, aas)


def fairpost_evaluate(model, features, labels, protected, seed):
    """Extract Z from trained model, apply Fairlearn ThresholdOptimizer on Z."""
    from sklearn.linear_model import LogisticRegression
    from fairlearn.postprocessing import ThresholdOptimizer

    torch.manual_seed(seed); np.random.seed(seed)
    train_ds, val_ds, test_ds = create_data_splits(features, labels, protected, random_state=seed)

    Z_train = extract_latent_Z(model, train_ds.features, device=DEVICE)
    Z_test  = extract_latent_Z(model, test_ds.features, device=DEVICE)
    y_train = train_ds.labels.numpy()
    y_test  = test_ds.labels.numpy()
    a_test  = test_ds.protected_attributes.numpy()

    to = ThresholdOptimizer(
        estimator=LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced'),
        constraints='demographic_parity', prefit=False, objective='accuracy_score')
    to.fit(Z_train, y_train, sensitive_features=train_ds.protected_attributes.numpy())

    preds = to.predict(Z_test, sensitive_features=a_test)
    acc = float((preds == y_test).mean())
    m0 = a_test == 0; m1 = a_test == 1
    pr0 = preds[m0].mean(); pr1 = preds[m1].mean()
    dpd = abs(pr0 - pr1)
    tpr0 = preds[m0 & (y_test == 1)].mean() if (m0 & (y_test == 1)).sum() else 0
    tpr1 = preds[m1 & (y_test == 1)].mean() if (m1 & (y_test == 1)).sum() else 0
    eod = abs(tpr0 - tpr1)
    fpr0 = preds[m0 & (y_test == 0)].mean() if (m0 & (y_test == 0)).sum() else 0
    fpr1 = preds[m1 & (y_test == 0)].mean() if (m1 & (y_test == 0)).sum() else 0
    aod = 0.5 * (eod + abs(fpr0 - fpr1))

    try:
        from sklearn.metrics import f1_score, roc_auc_score
        f1w = float(f1_score(y_test, preds, average='weighted'))
        proba = to._pmf_predict(Z_test, sensitive_features=a_test)
        auc = float(roc_auc_score(y_test, proba[:, 1])) if len(np.unique(y_test)) > 1 else float('nan')
    except Exception:
        f1w, auc = float('nan'), float('nan')

    return {'accuracy': acc, 'dpd': dpd, 'eod': eod, 'aod': aod,
            'f1_weighted': f1w, 'auc': auc}


# ======================================================================
# EXPERIMENT 1: AE-NS v2 baseline (3 seeds)
# ======================================================================
def run_v2_baseline():
    print('\n' + '='*70)
    print('  EXPERIMENT 1: AE-NS v2 baseline (3 seeds)')
    print('='*70)
    results = {}
    for ds_name in ['adult', 'compas', 'credit']:
        print(f'\n  [{ds_name}]')
        features, labels, protected, info = LOADERS[ds_name]()
        print(f'    n={features.shape[0]}, dim={features.shape[1]}, pw={POS_WEIGHTS[ds_name]}')
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            model, m, tt, _ = train_and_eval(ds_name, features, labels, protected, seed)
            elapsed = time.time() - t0
            ps = {'seed': seed, 'accuracy': m['accuracy'], 'f1_macro': m['f1_macro'],
                  'f1_weighted': m.get('f1_weighted', 0), 'dpd': m['dpd'], 'eod': m['eod'],
                  'aod': m['aod'], 'auc': m.get('auc', 0), 'nmi': m.get('nmi', 0),
                  'pos_rate_0': m.get('pos_rate_group_0', 0), 'pos_rate_1': m.get('pos_rate_group_1', 0),
                  'train_time': tt}
            per_seed.append(ps)
            print(f'    seed={seed}: acc={m["accuracy"]:.4f} dpd={m["dpd"]:.4f} nmi={m.get("nmi",0):.4f} ({tt:.1f}s)')
        agg = {}
        for k in ['accuracy', 'f1_macro', 'f1_weighted', 'dpd', 'eod', 'aod', 'auc', 'nmi', 'train_time']:
            vals = [p[k] for p in per_seed if k in p and not np.isnan(p[k])]
            agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
        results[ds_name] = {'dataset': ds_name, 'version': 'v2', 'arch': V2_ARCH,
                            'training': {**V2_TRAIN, 'label_smoothing': 0.03},
                            'per_seed': per_seed, 'aggregate': agg}
        with open(os.path.join(OUT, f'v2_{ds_name}_results.json'), 'w') as f:
            json.dump(results[ds_name], f, indent=2)
        print(f'    => saved v2_{ds_name}_results.json')
    return results


# ======================================================================
# EXPERIMENT 2: AE-NS Z + Fairlearn post-processing (3 seeds)
# ======================================================================
def run_fairpost():
    print('\n' + '='*70)
    print('  EXPERIMENT 2: AE-NS Z + Fairlearn post-processing (3 seeds)')
    print('='*70)
    results = {}
    for ds_name in ['adult', 'compas', 'credit']:
        print(f'\n  [{ds_name}]')
        features, labels, protected, info = LOADERS[ds_name]()
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            model, m_aens, tt, _ = train_and_eval(ds_name, features, labels, protected, seed,
                                                   alpha=1.0, label_smoothing=0.03)
            fp = fairpost_evaluate(model, features, labels, protected, seed)
            elapsed = time.time() - t0
            ps = {'seed': seed, **fp, 'train_time_aens': tt, 'total_time': elapsed,
                  'aens_accuracy': m_aens['accuracy'], 'aens_dpd': m_aens['dpd']}
            per_seed.append(ps)
            print(f'    seed={seed}: FP acc={fp["accuracy"]:.4f} dpd={fp["dpd"]:.4f} '
                  f'| AE-NS acc={m_aens["accuracy"]:.4f} dpd={m_aens["dpd"]:.4f} ({elapsed:.1f}s)')
        agg = {}
        for k in ['accuracy', 'dpd', 'eod', 'aod', 'f1_weighted', 'auc', 'total_time']:
            vals = [p[k] for p in per_seed if k in p and not np.isnan(p[k])]
            agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
        results[ds_name] = {'dataset': ds_name, 'method': 'AE-NS Z + Fairlearn',
                            'constraints': 'demographic_parity',
                            'per_seed': per_seed, 'aggregate': agg}
        with open(os.path.join(OUT, f'fairpost_3seeds_{ds_name}.json'), 'w') as f:
            json.dump(results[ds_name], f, indent=2)
        print(f'    => saved fairpost_3seeds_{ds_name}.json')
    return results


# ======================================================================
# EXPERIMENT 3: NMI + classify-on-Z (1 seed)
# ======================================================================
def run_nmi_classify():
    print('\n' + '='*70)
    print('  EXPERIMENT 3: NMI + classify-on-Z (seed=42)')
    print('='*70)
    results = {}
    for ds_name in ['adult', 'compas', 'credit']:
        print(f'\n  [{ds_name}]')
        features, labels, protected, info = LOADERS[ds_name]()
        torch.manual_seed(42); np.random.seed(42)
        train_ds, val_ds, test_ds = create_data_splits(features, labels, protected, random_state=42)
        model, m, tt, _ = train_and_eval(ds_name, features, labels, protected, seed=42)
        cz = classify_on_Z_experiment(
            model, train_ds.features, train_ds.labels, train_ds.protected_attributes,
            test_ds.features, test_ds.labels, test_ds.protected_attributes,
            device=DEVICE, random_state=42)
        nmi = m.get('nmi', 0)
        result = {
            'dataset': ds_name, 'seed': 42, 'nmi': nmi,
            'n_z': int(extract_latent_Z(model, test_ds.features[:1].to(DEVICE), device=DEVICE).shape[1]),
            'n_x': features.shape[1],
            'classify_on_Z': cz,
            'eval_summary': {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        }
        with open(os.path.join(OUT, f'{ds_name}_revision.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f'    NMI={nmi:.4f} acc_on_Z={cz["acc_on_Z"]:.4f} acc_on_X={cz["acc_on_X"]:.4f}')
        print(f'    => saved {ds_name}_revision.json')
        results[ds_name] = result
    return results


# ======================================================================
# EXPERIMENT 4: Sensitivity alpha sweep (1 seed, beta=0.05)
# ======================================================================
def run_sensitivity_alpha():
    print('\n' + '='*70)
    print('  EXPERIMENT 4: Sensitivity alpha sweep (beta=0.05 fixed)')
    print('='*70)
    alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    results = {}
    for ds_name in ['adult', 'compas', 'credit']:
        print(f'\n  [{ds_name}]')
        features, labels, protected, info = LOADERS[ds_name]()
        ds_results = {}
        for alpha in alpha_values:
            torch.manual_seed(42); np.random.seed(42)
            model, m, tt, _ = train_and_eval(ds_name, features, labels, protected, seed=42,
                                              alpha=alpha, beta=0.05)
            ds_results[str(alpha)] = {
                'accuracy': {'mean': m['accuracy'], 'std': 0.0, 'values': [m['accuracy']]},
                'dpd': {'mean': m['dpd'], 'std': 0.0, 'values': [m['dpd']]},
                'eod': {'mean': m['eod'], 'std': 0.0, 'values': [m['eod']]},
                'aod': {'mean': m['aod'], 'std': 0.0, 'values': [m['aod']]},
                'auc': {'mean': m.get('auc', 0), 'std': 0.0, 'values': [m.get('auc', 0)]},
                'nmi': {'mean': m.get('nmi', 0), 'std': 0.0, 'values': [m.get('nmi', 0)]},
            }
            print(f'    alpha={alpha}: acc={m["accuracy"]:.4f} dpd={m["dpd"]:.4f} nmi={m.get("nmi",0):.4f}')
        results[ds_name] = ds_results
        with open(os.path.join(OUT, f'{ds_name}_sensitivity_alpha.json'), 'w') as f:
            json.dump(ds_results, f, indent=2)
        print(f'    => saved {ds_name}_sensitivity_alpha.json')
    return results


# ======================================================================
# EXPERIMENT 5: Sensitivity beta sweep (1 seed, alpha=1.0)
# ======================================================================
def run_sensitivity_beta():
    print('\n' + '='*70)
    print('  EXPERIMENT 5: Sensitivity beta sweep (alpha=1.0 fixed)')
    print('='*70)
    beta_values = [0.01, 0.03, 0.05, 0.1, 0.2]
    results = {}
    for ds_name in ['adult', 'compas', 'credit']:
        print(f'\n  [{ds_name}]')
        features, labels, protected, info = LOADERS[ds_name]()
        ds_results = {}
        for beta in beta_values:
            torch.manual_seed(42); np.random.seed(42)
            model, m, tt, _ = train_and_eval(ds_name, features, labels, protected, seed=42,
                                              alpha=1.0, beta=beta)
            ds_results[str(beta)] = {
                'accuracy': {'mean': m['accuracy'], 'std': 0.0, 'values': [m['accuracy']]},
                'dpd': {'mean': m['dpd'], 'std': 0.0, 'values': [m['dpd']]},
                'eod': {'mean': m['eod'], 'std': 0.0, 'values': [m['eod']]},
                'aod': {'mean': m['aod'], 'std': 0.0, 'values': [m['aod']]},
                'auc': {'mean': m.get('auc', 0), 'std': 0.0, 'values': [m.get('auc', 0)]},
                'nmi': {'mean': m.get('nmi', 0), 'std': 0.0, 'values': [m.get('nmi', 0)]},
            }
            print(f'    beta={beta}: acc={m["accuracy"]:.4f} dpd={m["dpd"]:.4f} nmi={m.get("nmi",0):.4f}')
        results[ds_name] = ds_results
        with open(os.path.join(OUT, f'{ds_name}_sensitivity_beta.json'), 'w') as f:
            json.dump(ds_results, f, indent=2)
        print(f'    => saved {ds_name}_sensitivity_beta.json')
    return results


# ======================================================================
# EXPERIMENT 6: Ablation study (1 seed, 7 variants)
# ======================================================================
def run_ablation():
    print('\n' + '='*70)
    print('  EXPERIMENT 6: Ablation study (seed=42)')
    print('='*70)
    results = {}
    for ds_name in ['adult', 'compas', 'credit']:
        print(f'\n  [{ds_name}]')
        features, labels, protected, info = LOADERS[ds_name]()
        input_dim = features.shape[1]
        pw = POS_WEIGHTS[ds_name]

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
            model, m, tt, _ = train_and_eval(ds_name, features, labels, protected, seed=42,
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
        results[ds_name] = ds_results
        with open(os.path.join(OUT, f'{ds_name}_ablation.json'), 'w') as f:
            json.dump(ds_results, f, indent=2)
        print(f'    => saved {ds_name}_ablation.json')
    return results


# ======================================================================
# EXPERIMENT 7: t-SNE data extraction (1 seed)
# ======================================================================
def run_tsne_data():
    print('\n' + '='*70)
    print('  EXPERIMENT 7: t-SNE data extraction (seed=42)')
    print('='*70)
    results = {}
    for ds_name in ['adult', 'compas', 'credit']:
        print(f'\n  [{ds_name}]')
        features, labels, protected, info = LOADERS[ds_name]()
        torch.manual_seed(42); np.random.seed(42)
        model, m, tt, (Xs, ys, aas) = train_and_eval(ds_name, features, labels, protected, seed=42)
        Z = extract_latent_Z(model, Xs, device=DEVICE)
        X = Xs.cpu().numpy()
        A = aas.cpu().numpy()
        n = Z.shape[0]
        if n > 5000:
            idx = np.random.RandomState(42).choice(n, 5000, replace=False)
            Z, X, A = Z[idx], X[idx], A[idx]
        npz_path = os.path.join(OUT, f'tsne_data_{ds_name}.npz')
        np.savez(npz_path, Z=Z, X=X, A=A)
        print(f'    Z shape={Z.shape}, X shape={X.shape}, saved to {npz_path}')
        results[ds_name] = {'Z_shape': list(Z.shape), 'X_shape': list(X.shape)}
    return results


# ======================================================================
# Main
# ======================================================================
if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print(f'Seeds: {SEEDS}')
    print(f'Tolerance: {TOLERANCE}')
    total_start = time.time()

    r1 = run_v2_baseline()
    r2 = run_fairpost()
    r3 = run_nmi_classify()
    r4 = run_sensitivity_alpha()
    r5 = run_sensitivity_beta()
    r6 = run_ablation()
    r7 = run_tsne_data()

    total_time = time.time() - total_start
    print(f'\n{"="*70}')
    print(f'  ALL EXPERIMENTS COMPLETE — {total_time/60:.1f} min total')
    print(f'{"="*70}')
