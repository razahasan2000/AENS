"""
AE-NS Framework: Experiment Runner (Fixed - GPU)
=================================================
Changes vs original:
- Clears old cached results to force a clean re-run with fixes applied.
- Computes pos_weight from actual class distribution and passes it to AENSModel.
- Sensitivity uses 1 seed + smaller model (d_model=64) -- fast sweep.
- Baseline & Ablation use 3 seeds + full model (d_model=128).
- All 3 datasets: Adult, COMPAS, Credit Default.

Key fixes in aens_framework.py (applied before this run):
  FIX 1: F.relu(dpd - tolerance) -- model no longer rewarded for all-neg predictions.
  FIX 2: pos_weight in cross_entropy -- handles class imbalance.
  FIX 3: per-epoch loss breakdown logging.
  FIX 4: probability diagnostic stats in evaluate_model().
  FIX 5: sweep_thresholds() -- finds optimal operating threshold by F1.
"""

import sys, os, json, time, traceback
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np
from aens_framework import (
    generate_synthetic_fairness_data,
    load_adult_dataset, load_compas_dataset, load_credit_dataset,
    run_baseline_comparison, ablation_study,
    AENSModel, run_multi_seed_experiment,
)

# sklearn-based baselines (XGBoost, Fairlearn)
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import xgboost as xgb
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

# ============================================================================
# Config
# ============================================================================
DATASETS = {
    'adult':  ('Adult Income',   load_adult_dataset),
    'compas': ('COMPAS',         load_compas_dataset),
    'credit': ('Credit Default', load_credit_dataset),
}

NUM_SEEDS_BASELINE  = 3   # seeds for baseline & ablation (full results)
NUM_SEEDS_SENS      = 1   # seeds for sensitivity sweep (speed)

# Sensitivity grid
ALPHA_VALUES = [0.1, 0.5, 1.0]
BETA_VALUES  = [0.01, 0.05, 0.2]

RESULTS_DIR = 'AE_NS_Results'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device      : {device}")
print(f"Datasets    : {list(DATASETS.keys())}")
print(f"Baseline seeds   : {NUM_SEEDS_BASELINE}")
print(f"Sensitivity seeds: {NUM_SEEDS_SENS}  grid: {len(ALPHA_VALUES)}x{len(BETA_VALUES)}")
print("=" * 70)

# ============================================================================
# Helpers
# ============================================================================
def save_json(data, path):
    def _conv(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        if isinstance(o, dict):           return {k: _conv(v) for k, v in o.items()}
        if isinstance(o, list):           return [_conv(v) for v in o]
        return o
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_conv(data), f, indent=2)
    print(f"    Saved -> {path}")

def compute_pos_weight(labels: np.ndarray) -> float:
    """Compute positive class weight = n_neg / n_pos for class-imbalance handling."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def evaluate_sklearn(labels, preds, probs, protected, name='model'):
    """Return metric dict matching the format from evaluate_model for sklearn models."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['f1_macro'] = f1_score(labels, preds, average='macro')
    metrics['f1_weighted'] = f1_score(labels, preds, average='weighted')
    metrics['recall_macro'] = recall_score(labels, preds, average='macro')
    metrics['recall_weighted'] = recall_score(labels, preds, average='weighted')
    try:
        metrics['auc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auc'] = float('nan')

    mask_0 = (protected == 0)
    mask_1 = (protected == 1)
    if mask_0.sum() > 0 and mask_1.sum() > 0:
        pos_rate_0 = preds[mask_0].mean()
        pos_rate_1 = preds[mask_1].mean()
        metrics['dpd'] = abs(pos_rate_0 - pos_rate_1)
        tpr_0 = preds[mask_0 & (labels == 1)].mean() if (mask_0 & (labels == 1)).sum() > 0 else 0
        tpr_1 = preds[mask_1 & (labels == 1)].mean() if (mask_1 & (labels == 1)).sum() > 0 else 0
        metrics['eod'] = abs(tpr_0 - tpr_1)
        fpr_0 = preds[mask_0 & (labels == 0)].mean() if (mask_0 & (labels == 0)).sum() > 0 else 0
        fpr_1 = preds[mask_1 & (labels == 0)].mean() if (mask_1 & (labels == 0)).sum() > 0 else 0
        metrics['aod'] = 0.5 * (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1))
        for g in [0, 1]:
            g_mask = (protected == g)
            metrics[f'acc_group_{g}'] = accuracy_score(labels[g_mask], preds[g_mask])
            metrics[f'f1_group_{g}'] = f1_score(labels[g_mask], preds[g_mask], zero_division=0)
            metrics[f'recall_group_{g}'] = recall_score(labels[g_mask], preds[g_mask], zero_division=0)
            metrics[f'pos_rate_group_{g}'] = preds[g_mask].mean()
    else:
        metrics['dpd'] = float('nan')
        metrics['eod'] = float('nan')
        metrics['aod'] = float('nan')
    return metrics


def run_sklearn_baseline(features, labels, protected, model_cfg, num_seeds=3):
    """Run an sklearn/XGBoost/Fairlearn baseline, returning same format as run_multi_seed_experiment."""
    from aens_framework import create_data_splits
    seeds = [42, 123, 456, 789, 2024]
    all_metrics = []
    training_times = []

    for i, seed in enumerate(seeds[:num_seeds]):
        np.random.seed(seed)
        train_ds, _, test_ds = create_data_splits(features, labels, protected, random_state=seed)

        X_train = train_ds.features.numpy()
        y_train = train_ds.labels.numpy()
        a_train = train_ds.protected_attributes.numpy()
        X_test  = test_ds.features.numpy()
        y_test  = test_ds.labels.numpy()
        a_test  = test_ds.protected_attributes.numpy()

        t0 = time.time()
        model_type = model_cfg.get('type', 'xgboost')

        if model_type == 'xgboost':
            best = model_cfg.get('best', {})
            clf = xgb.XGBClassifier(
                n_estimators=best.get('n_estimators', 100),
                max_depth=best.get('max_depth', 6),
                learning_rate=best.get('learning_rate', 0.1),
                subsample=best.get('subsample', 0.8),
                colsample_bytree=best.get('colsample_bytree', 0.8),
                objective='binary:logistic', eval_metric='logloss',
                random_state=seed, verbosity=0, use_label_encoder=False,
            )
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)

        elif model_type in ('fairlearn_dp', 'fairlearn_eo'):
            estimator = xgb.XGBClassifier(
                n_estimators=50, max_depth=4,
                objective='binary:logistic', verbosity=0,
                random_state=seed, use_label_encoder=False,
            )
            constraint = DemographicParity() if model_type == 'fairlearn_dp' else EqualizedOdds()
            clf = ExponentiatedGradient(estimator, constraint, max_iter=model_cfg.get('max_iter', 50))
            clf.fit(X_train, y_train, sensitive_features=a_train)
            preds = clf.predict(X_test)
            # ExponentiatedGradient doesn't expose predict_proba;
            # use the first underlying predictor for probabilities
            if hasattr(clf, 'predictors_') and len(clf.predictors_) > 0:
                probs = clf.predictors_[0].predict_proba(X_test)[:, 1]
            else:
                probs = preds.astype(float)

        else:
            raise ValueError(f"Unknown sklearn model type: {model_type}")

        train_time = time.time() - t0
        metrics = evaluate_sklearn(y_test, preds, probs, a_test, name=model_cfg.get('name', model_type))
        all_metrics.append(metrics)
        training_times.append(train_time)

    result = {}
    metric_keys = all_metrics[0].keys()
    for key in metric_keys:
        first_val = all_metrics[0].get(key)
        if isinstance(first_val, (int, float, np.integer, np.floating)) and not isinstance(first_val, bool):
            values = [float(m[key]) for m in all_metrics if m.get(key) is not None
                      and isinstance(m[key], (int, float, np.integer, np.floating))
                      and not np.isnan(m[key])]
            result[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'values': values} if values \
                else {'mean': float('nan'), 'std': float('nan'), 'values': []}
        else:
            result[key] = {'values': [m[key] for m in all_metrics if key in m]}
    result['training_time'] = {
        'mean': float(np.mean(training_times)),
        'std': float(np.std(training_times)),
        'values': training_times,
    }
    return result


def run_sensitivity_fast(features, labels, protected, device, num_seeds, pw: float):
    """
    Compact sensitivity sweep: 3 alpha x 3 beta = 9 combos.
    Uses smaller model (d_model=64) for speed.
    Passes pos_weight so AE-NS uses class-imbalance weighting.
    """
    results = {'alpha': {}, 'beta': {}}
    input_dim = features.shape[1]

    print("\n  Alpha sensitivity (beta=0.05 fixed) ...")
    for alpha in ALPHA_VALUES:
        print(f"    alpha={alpha} ...", end='', flush=True)
        t0 = time.time()
        r = run_multi_seed_experiment(
            features, labels, protected,
            model_class=AENSModel,
            model_kwargs=dict(input_dim=input_dim, alpha=alpha, beta=0.05,
                              d_model=64, nhead=4, num_encoder_layers=2,
                              pos_weight=pw, fairness_tolerance=0.01),
            num_seeds=num_seeds, device=device, verbose=False,
            warmup_epochs=3,
        )
        results['alpha'][str(alpha)] = r
        print(f" done ({time.time()-t0:.0f}s)")

    print("\n  Beta sensitivity (alpha=0.5 fixed) ...")
    for beta in BETA_VALUES:
        print(f"    beta={beta} ...", end='', flush=True)
        t0 = time.time()
        r = run_multi_seed_experiment(
            features, labels, protected,
            model_class=AENSModel,
            model_kwargs=dict(input_dim=input_dim, alpha=0.5, beta=beta,
                              d_model=64, nhead=4, num_encoder_layers=2,
                              pos_weight=pw, fairness_tolerance=0.01),
            num_seeds=num_seeds, device=device, verbose=False,
            warmup_epochs=3,
        )
        results['beta'][str(beta)] = r
        print(f" done ({time.time()-t0:.0f}s)")

    return results

# ============================================================================
# Per-dataset runner
# ============================================================================
def run_dataset(key, name, loader):
    print(f"\n{'='*70}")
    print(f"DATASET: {name.upper()}")
    print(f"{'='*70}")

    out_dir = os.path.join(RESULTS_DIR, key)
    os.makedirs(out_dir, exist_ok=True)

    baseline_path    = os.path.join(out_dir, 'baseline_results.json')
    sensitivity_path = os.path.join(out_dir, 'sensitivity_results.json')
    ablation_path    = os.path.join(out_dir, 'ablation_results.json')

    # -- Load data --
    print(f"  Loading {name} ...")
    t0 = time.time()
    features, labels, protected, info = loader()
    print(f"  Loaded in {time.time()-t0:.1f}s | "
          f"{info['n_samples']} samples, {info['n_features']} features")
    print(f"  Balance: {info['class_balance']}  |  {info['protected_attr']}")

    # Compute pos_weight from actual class distribution
    pw = compute_pos_weight(labels)
    print(f"  pos_weight (neg/pos ratio): {pw:.2f}")

    all_results = {'dataset_info': info}

    # -- Experiment 1: Baseline (always re-run with fixes) --
    print(f"\n  [Baseline] Running ({NUM_SEEDS_BASELINE} seeds) with FIX applied ...")
    try:
        t0 = time.time()
        # Pass pos_weight into AE-NS via the baselines dict override
        # run_baseline_comparison uses its own internal model_kwargs dict,
        # so we call run_multi_seed_experiment directly for AE-NS and
        # then call run_baseline_comparison for the other models.
        from aens_framework import (
            AdversarialDebiasingModel, PrejudiceRemoverModel, LAFANModel
        )
        input_dim = features.shape[1]

        baselines_custom = {
            'AE-NS (Ours)': {
                'model_class': AENSModel,
                'model_kwargs': {
                    'input_dim': input_dim, 'alpha': 0.5, 'beta': 0.05,
                    'd_model': 128, 'nhead': 8, 'num_encoder_layers': 4,
                    'pos_weight': pw,
                    'fairness_tolerance': 0.01,
                }
            },
            'Adversarial Debiasing': {
                'model_class': AdversarialDebiasingModel,
                'model_kwargs': {'input_dim': input_dim, 'd_model': 128, 'adv_weight': 1.0}
            },
            'Prejudice Remover': {
                'model_class': PrejudiceRemoverModel,
                'model_kwargs': {'input_dim': input_dim, 'd_model': 128, 'eta': 5.0}
            },
            'LAFAN (Lagrangian w/o AE)': {
                'model_class': LAFANModel,
                'model_kwargs': {'input_dim': input_dim, 'd_model': 128,
                                 'fairness_tolerance': 0.01, 'initial_v': 1.0}
            },
            'Standard NN (No Fairness)': {
                'model_class': PrejudiceRemoverModel,
                'model_kwargs': {'input_dim': input_dim, 'd_model': 128, 'eta': 0.0}
            },
        }

        # sklearn-based baselines configs
        sklearn_baselines = {
            'XGBoost (No Fairness)': {
                'type': 'xgboost',
                'best': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                         'subsample': 0.8, 'colsample_bytree': 0.8},
            },
            'Fairlearn-DP + XGBoost': {
                'type': 'fairlearn_dp',
                'max_iter': 50,
            },
            'Fairlearn-EO + XGBoost': {
                'type': 'fairlearn_eo',
                'max_iter': 50,
            },
        }

        r = {}
        for bname, bcfg in baselines_custom.items():
            print(f"\n  {'='*60}")
            print(f"  Baseline: {bname}")
            print(f"  {'='*60}")
            r[bname] = run_multi_seed_experiment(
                features, labels, protected,
                model_class=bcfg['model_class'],
                model_kwargs=bcfg['model_kwargs'],
                num_seeds=NUM_SEEDS_BASELINE, device=device, verbose=True
            )

        for bname, bcfg in sklearn_baselines.items():
            print(f"\n  {'='*60}")
            print(f"  Baseline: {bname}")
            print(f"  {'='*60}")
            r[bname] = run_sklearn_baseline(
                features, labels, protected,
                model_cfg=bcfg, num_seeds=NUM_SEEDS_BASELINE
            )

        all_results['baseline_comparison'] = r
        save_json(r, baseline_path)
        print(f"  [Baseline] Done in {(time.time()-t0)/60:.1f} min")
    except Exception as e:
        print(f"  [Baseline ERROR] {e}"); traceback.print_exc()

    # -- Experiment 2: Sensitivity --
    print(f"\n  [Sensitivity] Running ({NUM_SEEDS_SENS} seed, {len(ALPHA_VALUES)}x{len(BETA_VALUES)} grid) ...")
    try:
        t0 = time.time()
        r = run_sensitivity_fast(features, labels, protected, device, NUM_SEEDS_SENS, pw)
        all_results['sensitivity_analysis'] = r
        save_json(r, sensitivity_path)
        print(f"  [Sensitivity] Done in {(time.time()-t0)/60:.1f} min")
    except Exception as e:
        print(f"  [Sensitivity ERROR] {e}"); traceback.print_exc()

    # -- Experiment 3: Ablation --
    print(f"\n  [Ablation] Running ({NUM_SEEDS_BASELINE} seeds) ...")
    try:
        t0 = time.time()
        r = ablation_study(
            features, labels, protected,
            num_seeds=NUM_SEEDS_BASELINE, device=device, verbose=True,
            pos_weight=pw
        )
        all_results['ablation_study'] = r
        save_json(r, ablation_path)
        print(f"  [Ablation] Done in {(time.time()-t0)/60:.1f} min")
    except Exception as e:
        print(f"  [Ablation ERROR] {e}"); traceback.print_exc()

    save_json(all_results, os.path.join(out_dir, 'all_results.json'))
    print(f"\n  [{name}] All experiments complete.")
    return all_results

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()
    master = {}

    for key, (name, loader) in DATASETS.items():
        t_ds = time.time()
        try:
            master[key] = run_dataset(key, name, loader)
        except Exception as e:
            print(f"\n[FATAL] {name}: {e}"); traceback.print_exc()
        print(f"\n  [{name}] Wall time: {(time.time()-t_ds)/60:.1f} min")

    save_json(master, os.path.join(RESULTS_DIR, 'master_results.json'))
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE  |  Total: {elapsed:.1f} min")
    print(f"Results: {os.path.abspath(RESULTS_DIR)}")
    print(f"{'='*70}")
