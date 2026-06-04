"""
run_v2_fix_adult_credit.py — Re-run Adult and Credit with correct pos_weight.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from aens_framework import (
    AENSModel, load_adult_dataset, load_credit_dataset,
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


def run_ds(ds_name, loader):
    features, labels, protected, info = loader()
    pw = compute_pos_weight(labels)
    print(f'\n{"="*60}')
    print(f'  {ds_name.upper()}  (pw={pw:.2f})')
    print(f'{"="*60}')

    # === 1. AE-NS v2 standalone (3 seeds) ===
    print(f'\n  --- AE-NS v2 standalone (3 seeds) ---')
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
        m = evaluate_model(model, make_loader(Xs, ys, aas, False), device=DEVICE)
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
    with open(os.path.join(OUT, f'v2_{ds_name}_results.json'), 'w') as f:
        json.dump({'dataset': ds_name, 'version': 'v2', 'pos_weight': pw,
                   'arch': V2_ARCH, 'training': {**V2_TRAIN, 'label_smoothing': 0.03},
                   'per_seed': per_seed, 'aggregate': agg}, f, indent=2)
    print(f'    => saved v2_{ds_name}_results.json')

    # === 2. AE-NS Z + Fairlearn (3 seeds) ===
    print(f'\n  --- Fairpost (3 seeds) ---')
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
        m_aens = evaluate_model(model, make_loader(Xs, ys, aas, False), device=DEVICE)

        Z_train = extract_latent_Z(model, train_ds.features, device=DEVICE)
        Z_test = extract_latent_Z(model, test_ds.features, device=DEVICE)
        to = ThresholdOptimizer(
            estimator=LogisticRegression(max_iter=2000, random_state=seed, class_weight='balanced'),
            constraints='demographic_parity', prefit=False, objective='accuracy_score')
        to.fit(Z_train, train_ds.labels.numpy(), sensitive_features=train_ds.protected_attributes.numpy())
        preds = to.predict(Z_test, sensitive_features=test_ds.protected_attributes.numpy())
        y_test = test_ds.labels.numpy(); a_test = test_ds.protected_attributes.numpy()
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
        fp_seeds.append({'seed': seed, 'accuracy': acc, 'dpd': dpd, 'eod': eod, 'aod': aod,
                         'aens_accuracy': m_aens['accuracy'], 'aens_dpd': m_aens['dpd'],
                         'total_time': elapsed})
        print(f'    seed={seed}: FP acc={acc:.4f} dpd={dpd:.4f} ({elapsed:.1f}s)')
    fp_agg = {}
    for k in ['accuracy', 'dpd', 'eod', 'aod', 'total_time']:
        vals = [p[k] for p in fp_seeds]
        fp_agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
    with open(os.path.join(OUT, f'fairpost_3seeds_{ds_name}.json'), 'w') as f:
        json.dump({'dataset': ds_name, 'method': 'AE-NS Z + Fairlearn',
                   'constraints': 'demographic_parity', 'pos_weight': pw,
                   'per_seed': fp_seeds, 'aggregate': fp_agg}, f, indent=2)
    print(f'    => saved fairpost_3seeds_{ds_name}.json')

    # === 3. NMI + classify-on-Z (1 seed) ===
    print(f'\n  --- NMI + classify-on-Z ---')
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
    m = evaluate_model(model, make_loader(Xs, ys, aas, False), device=DEVICE)
    cz = classify_on_Z_experiment(
        model, train_ds.features, train_ds.labels, train_ds.protected_attributes,
        test_ds.features, test_ds.labels, test_ds.protected_attributes,
        device=DEVICE, random_state=42)
    nmi = m.get('nmi', 0)
    with open(os.path.join(OUT, f'{ds_name}_revision.json'), 'w') as f:
        json.dump({'dataset': ds_name, 'seed': 42, 'epochs': 150, 'warmup': 5,
                   'tolerance': TOLERANCE, 'pos_weight': pw, 'train_time_s': tt,
                   'n_z': int(cz['n_z']), 'n_x': int(cz['n_x']), 'nmi': nmi,
                   'classify_on_Z': cz,
                   'eval_summary': {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}},
                  f, indent=2)
    print(f'    NMI={nmi:.4f} acc_on_Z={cz["acc_on_Z"]:.4f} acc_on_X={cz["acc_on_X"]:.4f}')
    print(f'    => saved {ds_name}_revision.json')

    # === 4. t-SNE data (1 seed) ===
    print(f'\n  --- t-SNE data ---')
    Z = extract_latent_Z(model, test_ds.features.to(DEVICE), device=DEVICE)
    X = test_ds.features.numpy()
    A = test_ds.protected_attributes.numpy()
    n = Z.shape[0]
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        Z, X, A = Z[idx], X[idx], A[idx]
    np.savez(os.path.join(OUT, f'tsne_data_{ds_name}.npz'), Z=Z, X=X, A=A)
    print(f'    Z={Z.shape} X={X.shape}')

    return pw


if __name__ == '__main__':
    for ds_name, loader in [('adult', load_adult_dataset), ('credit', load_credit_dataset)]:
        run_ds(ds_name, loader)
    print('\nAll adult/credit fixes complete!')
