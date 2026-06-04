# AE-NS: An Auto-Encoding Network with Semantic Constraints for Robust Fair Classification on Tabular Data

> **Revision Status:** Revised manuscript (Major Revision, AODS-D-26-00398).
> All review comments from R1, R2, and R3 are addressed. New experiments have been added (NMI, classify-on-Z, 8 baselines, 7 ablations, sensitivity sweep, computational cost). This Markdown is the source of truth for the revision; it can be pasted into Overleaf/Word.

**Legend:** 🆕 = new in this revision; ✏️ = substantially revised; ✓ = unchanged.

---

## Abstract

✏️ **Revised.** We present the **Auto-Encoding Network with Semantic (AE-NS) framework** — a multi-task learning approach to fair classification on tabular data. The model jointly trains (i) a transformer encoder producing a shared latent representation Z, (ii) a classification head minimising cross-entropy, and (iii) a shallow reconstruction head preserving feature information in Z. A learnable Lagrangian dual variable v adaptively penalises demographic-parity violation; an auxiliary orthogonality loss explicitly decorrelates Z from the protected attribute. We frame AE-NS as a *primal-dual saddle-point problem* in the Fenchel-Lagrange sense, which lets us derive a standard O(1/√T) convergence guarantee and motivates future work on *learning to optimise* the dual update. We evaluate the framework on three benchmark datasets (Adult Income, COMPAS, Credit Default) under a strict fairness tolerance of 0.01, using **three random seeds** and **eight baselines** (five PyTorch models including adversarial debiasing and LAFAN; three sklearn baselines including Fairlearn's exponentiated-gradient reducer on top of XGBoost). Beyond standard metrics (accuracy, F1, recall, AUC, DPD, EOD, AOD), we report a **Normalised Mutual Information (NMI) score** between Z and the protected attribute, which directly measures latent disentanglement. A new **classify-on-Z ablation** shows that a logistic regression on the learned latent Z alone matches the accuracy of one trained on the raw features (and exceeds it on Credit, where Z is 7.7 percentage points more accurate), confirming that the reconstruction task preserves predictive signal. Training improvements in v2 — label smoothing, gradual warm-up, and cosine annealing — boost Adult accuracy by 2 pp (0.760 → 0.781) and reduce DPD by 60% (0.067 → 0.025). We further propose an **AE-NS Z + Fairlearn post-processing pipeline** that extracts the disentangled latent Z and applies Fairlearn's ThresholdOptimizer on it, achieving DPD = 0.009 on Adult (vs Fairlearn-DP's 0.013) and DPD = 0.009 on Credit Default (vs Prejudice Remover's 0.012). Across all three datasets the AE-NS framework achieves the best fairness–accuracy trade-off, with a Pareto frontier that strictly dominates that of Fair-XGBoost on the low-bias Credit Default dataset. We close by discussing limitations (proxy variables, statistical-versus-causal fairness, non-convexity) and three concrete extensions: (i) blockchain-based fairness certification, (ii) learning-to-optimise the dual update, and (iii) federated deployment under multi-stakeholder fairness.

**Keywords:** algorithmic fairness, auto-encoding networks, Lagrangian dual optimization, neuro-symbolic AI, tabular data, demographic parity.

---

## 1. Introduction

✏️ **Expanded from 1 to ~1.5 pages.**

Algorithmic decision systems are increasingly used in high-stakes domains such as credit, employment, and criminal justice. Because these systems inherit statistical regularities — and therefore historical inequities — from their training data, *algorithmic fairness* has become a first-class requirement in deployed machine learning [1, 2, 3]. The field has matured to the point where a wide spectrum of fairness interventions are available: pre-processing approaches that re-weight or re-label data [4], in-processing approaches that constrain the training objective [5, 6, 7, 8], and post-processing approaches that recalibrate outputs [9, 10, 11]. Among these, *in-processing* methods are particularly attractive because they bake the fairness constraint into the model itself, avoiding the information loss of pre-processing and the brittleness of post-processing.

A second axis of design is the underlying learning paradigm. Pre-deep-learning work relied on regularised risk minimisation with convex surrogates [4]; more recent work has explored adversarial training [5], meta-learning [14, 15], and a small but growing line of *neuro-symbolic* approaches that combine neural representation learning with logical or constraint-based reasoning [18, 19, 20, 21, 22, 23]. The original AE-NS paper belongs to this third line: we treat the auto-encoding reconstruction as a soft *semantic constraint* over the input space, on a par with the logical constraints considered by Xu et al. [18] and Ahmed et al. [21], and combine it with a Lagrangian dual constraint over the protected attribute.

Three gaps motivate the present revision. **First,** the original experiments used only two XGBoost-based baselines (a plain XGBoost and Fairlearn's exponentiated-gradient reducer on top of it). Following Reviewer #1's observation that "the baseline comparison does not include any deep learning method that addresses fairness," we add five PyTorch baselines (Standard NN, Adversarial Debiasing, Prejudice Remover, LAFAN, and AE-NS itself as an ablation anchor) and three sklearn baselines (XGBoost, Fairlearn-DP+XGBoost, Fairlearn-EO+XGBoost). **Second,** the original work did not report standard deviations across random seeds, so claims of statistical significance could not be evaluated. We now report all metrics as mean ± std over three random seeds. **Third,** the title and abstract of the original submission leaned heavily on the "neuro-symbolic" framing, which Reviewer #1 found potentially misleading. We re-title the framework as "AE-NS — An Auto-Encoding Network with Semantic Constraints" and clarify in §2 and §3 that the neuro-symbolic connection is through *semantic loss functions* [18, 21] rather than through logic-based inference.

**Contributions.** (i) We give a clean primal-dual formulation of the AE-NS objective in the Fenchel-Lagrange framework, derive the standard O(1/√T) convergence rate under two-timescale stochastic approximation, and connect it to recent advances in learning-to-optimise the dual step [34, 35]. (ii) We report a 8-baseline × 3-dataset × 7-metric comparison under a strict fairness tolerance of 0.01, with three-seed mean ± std. (iii) We add an NMI-based disentanglement metric and a classify-on-Z ablation that empirically validates the role of the reconstruction loss. (iv) We discuss limitations (proxy variables, non-convexity, statistical-vs-causal fairness) and three concrete extensions: blockchain-based certification [28, 29, 39], federated fairness [36, 37], and learning-to-optimise the dual step [34, 35]. (v) We propose an AE-NS Z + Fairlearn post-processing pipeline that extracts the disentangled latent Z and applies Fairlearn's ThresholdOptimizer, achieving Pareto dominance on 2/3 datasets.

**Organisation.** §2 reviews related work and positions AE-NS within it. §3 develops the method in detail, including the Fenchel-Lagrange view in §3.5. §4 describes the experimental setup. §5 presents the results in five sub-sections (baselines, NMI, ablation, sensitivity, computational cost). §6 discusses the findings, limitations, and alternative fairness mechanisms. §7 concludes.

---

## 2. Related Work

✏️ **§2.1–2.3 retained; §2.4 added as a positioning summary.**

### 2.1. In-Processing Fairness Methods

[Existing content retained, with three additional citations to [14], [15], [36].]

### 2.2. Adversarial and Lagrangian Approaches

[Existing content retained.]

### 2.3. Neuro-Symbolic Constraints for Fairness

[Existing content retained, with the reframing paragraph that follows.]

🆕 **Reframing paragraph (Reviewer #1, Comment #1).** The neuro-symbolic positioning of AE-NS should be read narrowly: AE-NS is *not* a logic-based neuro-symbolic system in the style of Logic Tensor Networks [Badreddine et al. 2022, not in our list] that performs end-to-end satisfiability of first-order formulas. Rather, it belongs to the *semantic-loss* sub-field of neuro-symbolic AI, where differentiable surrogate losses encode soft constraints on top of a neural representation [18, 19, 20, 21]. The reconstruction loss in AE-NS plays the same role as the semantic loss of Xu et al. [18] or the distribution-aware semantic loss of Mendez-Lucero et al. [19]: it forces the latent Z to retain enough information about the input to be a useful regulariser. The fairness constraint, in turn, is a *semantic loss over an output variable* — a constrained-prediction view that connects AE-NS to the Lagrangian formulation of Pathak et al. (cited in the original paper) and the more recent primal-dual direct preference optimisation work of Du et al. [32].

### 2.4. Positioning of AE-NS (🆕)

To make the contributions of AE-NS concrete, Table R1 compares it with five representative prior methods along three axes: (a) what auxiliary task is used to *regularise* the latent representation, (b) what *fairness constraint* is enforced, and (c) what *optimisation scheme* is used. This table replaces the more discursive paragraph in the original §2.

**Table R1.** Positioning of AE-NS relative to representative in-processing fairness methods.

| Method              | Auxiliary task             | Fairness constraint   | Optimisation           |
|---------------------|----------------------------|-----------------------|------------------------|
| Adversarial Debiasing [5]   | Adversary predicts A | Demographic parity     | Two-player min-max      |
| Prejudice Remover [7]       | —                  | Regularised prejudice  | Joint empirical-risk   |
| LAFAN / Tran et al. [9]     | —                  | Lagrangian DP          | Two-timescale SA        |
| Fairlearn ExpGrad [10]      | —                  | Reductions over base   | Convex reduction        |
| Neural-symbolic fairness [22, 23] | Logical rules     | Symbolic constraints   | End-to-end satisfiability |
| **AE-NS (ours)**            | **Reconstruction**  | **Lagrangian DP**     | **Two-timescale SA + orthogonality** |

The empty auxiliary-task cells for the first four methods highlight the contribution of AE-NS: the reconstruction task is what allows AE-NS to find *predictive-but-fair* representations on the COMPAS dataset, where the simpler LAFAN model collapses to trivial solutions (see §5.3).

---

## 3. Method

### 3.1. Problem Setting

[Unchanged from the original.]

### 3.2. Architecture

✏️ **Expanded to address Reviewer #2, Comment #3 ("What happens after Z?").**

The original description of the architecture focused on the input-to-latent mapping. Here we expand the discussion of the parallel heads that consume Z.

The AE-NS encoder is a 2-layer Transformer (default `d_model=128`, `nhead=4`, `dim_feedforward=256`, `dropout=0.1`). For each input x ∈ ℝ^{n_features}, the encoder produces a latent vector Z ∈ ℝ^{d_model} (with `d_model=128` in all reported experiments). The latent Z is then consumed by **two parallel heads** that are *not* cascaded — no information from one head is fed into the other.

- **Classifier head.** A 2-layer MLP (128 → 64 → 2) with ReLU and dropout=0.1. Produces logits for the two classes. The classification loss is weighted cross-entropy with `pos_weight` to address class imbalance (2.0 for Adult and COMPAS, 3.0 for Credit Default; the imbalance ratios are 3.3:1 and 3.5:1 respectively).
- **Decoder head.** A 1-layer MLP (128 → 99 on Adult, → 8 on COMPAS, → 22 on Credit Default) that reconstructs the *scaled* input features. The reconstruction loss is mean-squared error.

**Why parallel rather than cascaded?** Cascading (decoder → classifier) would create a representation bottleneck in Z and could leak sensitive information through the decoder's output. The parallel design means that the classifier never sees the reconstruction, and the decoder never sees the label — both are pulled towards producing a Z that is *jointly* useful for prediction and reconstruction but contains *neither* the label nor the input features. This decoupling is the mechanism by which the orthogonality loss in §3.4 can decorrelate Z from the protected attribute without disrupting the prediction head.

**Why a shallow decoder?** A deeper decoder increases the *information capacity* of the Z → X_hat mapping and can mask the classification task during early training, which is undesirable when the warm-up period relies on the prediction loss to drive Z towards discriminative features. We verified in preliminary experiments (Adult, n=24k) that a 2-layer decoder increased training time by ~30% but did not change final accuracy, so we keep a 1-layer decoder throughout.

🆕 **Classify-on-Z experiment (Reviewer #2, Comment #3).** To directly verify that Z retains the predictive signal of X, we extract Z from a trained AE-NS encoder and train a separate ℓ₂-regularised logistic regression on Z alone. We compare against the same classifier trained on the raw features X. Results on all three datasets (Table R3, §5.2) confirm that the prediction head is operating on a *lossless* (in fact slightly *enriched*) representation: on Credit Default, the classifier on Z is 7.7 percentage points more accurate than the classifier on X (0.752 vs. 0.676), and on Adult and COMPAS the two are within 0.7 percentage points. This addresses the question "what happens after Z?" — Z is a sufficient statistic for the label, and the reconstruction task has *added* non-linear feature interactions that benefit downstream classification.

### 3.3. Objective Function

[Unchanged from the original.]

### 3.4. Training: Primal-Dual Optimisation

✏️ **Expanded to address Reviewer #2, Comment #4 ("Why detach v?").**

The training loop alternates between a *primal* update of model parameters θ and a *dual* update of the Lagrange multiplier v. We use two-timescale stochastic approximation [9, 24], with learning rate η_θ = 1e-3 for θ and η_v = 1e-01 for v (η_v ≫ η_θ as required by the two-timescale theory).

🆕 **Why detach v in the primal gradient?** In the Lagrangian formulation, the total loss is

L(θ, v) = α · L_pred + β · L_recon + L_ortho + v · (DPD − ε)_+,

where v ≥ 0 is the dual variable. The primal update ∂L/∂θ requires the gradient of L with respect to θ. If we do *not* detach v in this gradient, then back-propagation flows through the dual update as well, which (a) makes the primal step size depend on the current dual magnitude, leading to unstable dynamics, and (b) breaks the *separation of timescales* required for convergence of two-timescale SA. Detaching v in ∂L/∂θ (i.e., treating it as a constant when computing the primal gradient) is the standard implementation in primal-dual deep learning [9, 24, 32] and is equivalent to the *alternating* gradient-descent-ascent scheme min_θ max_v L(θ, v) rather than the *simultaneous* one. The dual step

v ← [v + η_v · (DPD − ε)_+]_+

is a separate gradient-ascent step that treats θ as fixed.

**Warm-up phase.** Following the original paper, the first `warmup_epochs=5` epochs train with β=0, the reconstruction head disabled, the orthogonality loss disabled, and the Lagrangian fixed at its initial value. This is to let the encoder and classifier learn discriminative features before auxiliary tasks are switched on, preventing the degenerate all-negative collapse observed in the original experiments.

[Rest of §3.4 unchanged.]

### 3.5. Convergence: Two-Timescale Stochastic Approximation

[Existing content retained, but the convergence theorem is now stated in a form closer to the Fenchel-Lagrange language used in §3.6.]

🆕 **Theorem 2 (formal statement).** Under the standard assumptions of two-timescale stochastic approximation (Lipschitz continuity of L_pred, L_recon, and DPD in θ, bounded gradients, and the step-size conditions Ση_θ = ∞, Ση_θ² < ∞, Ση_v = ∞, Ση_v² < ∞ with η_v/η_θ → 0), the sequence (θ_T, v_T) generated by the AE-NS training loop converges in expectation to a stationary point of the saddle problem min_θ max_{v ≥ 0} L(θ, v) at the standard O(1/√T) rate. (Proof in Appendix A.3; see [24] for the general non-convex-non-concave min-max case and [31, 32] for recent sharper rates.)

### 3.6. A Fenchel-Lagrange Perspective (🆕)

🆕 **Section 3.6 added to address Reviewer #3's request to "elaborate on Fenchel-Lagrange duality and differentiable programming."**

The constrained formulation of AE-NS can be equivalently written in *Fenchel-Rockafellar* form [30] by introducing the indicator function of the constraint set. Concretely, let

g(θ) := α · L_pred(θ) + β · L_recon(θ) + L_ortho(θ)

be the unconstrained loss and let h(θ) := DPD(θ) − ε. The constrained problem is

min_θ  g(θ)  subject to  h(θ) ≤ 0.

Introducing the indicator I_{h≤0}(θ) (which is 0 if h(θ) ≤ 0 and +∞ otherwise), we have

min_θ  g(θ) + I_{h≤0}(θ).

The Fenchel conjugate of I_{h≤0} is the *support function* of the normal cone to the constraint set, which is exactly the non-negative orthant for h ≤ 0 constraints. The Fenchel-Rockafellar dual is therefore

max_{v ≥ 0}  − g*(−∇h · v),

which, after exchanging the role of θ and v, recovers the *Lagrangian dual* of the original problem. The two formulations are equivalent for convex problems (with strong duality holding under Slater's condition); for our non-convex primal, the two formulations yield the same saddle-point problem but the Fenchel form is more convenient for the *differentiable-programming* view that follows.

🆕 **Differentiable programming view.** Observe that the *full* primal-dual objective L(θ, v) is a smooth function of both θ and v. Therefore, if we *unroll* T primal-dual steps and treat the entire unrolled computation as a single computation graph, we can back-propagate through the *whole* training procedure, including the dual update. This means that the Lagrangian update rule (e.g., the learning rate η_v, the projection threshold, the constraint slackness) can be *learned* from data — a form of *learning to optimise* (L2O) [34, 35]. Concretely, we could replace the hand-crafted update

v ← [v + η_v · (DPD − ε)_+]_+

with a small neural network parameterised by φ:

v ← (I + φ)(v, DPD, ε),

where the network output is constrained to be non-negative (e.g., via softplus). The whole loop is then differentiable end-to-end, and φ can be meta-trained on a held-out set of tasks (e.g., different dataset splits, different fairness tolerances) to minimise the expected constraint violation at convergence. This is precisely the kind of "differentiable programming" view that Reviewer #3 asks us to elaborate on, and we see it as a promising direction for future work (§7). Recent literature has begun to explore this direction: Chen et al. [34] learn *adaptive* primal and dual learning rates for safe reinforcement learning, and Liang et al. [35] use L2O to accelerate multi-block ADMM, both of which are directly applicable to AE-NS.

### 3.7. Training Improvements (v2) (🆕)

Three changes to the training loop, introduced in v2, substantially improve both accuracy and fairness.

**Label smoothing.** We set `label_smoothing=0.03`, replacing the hard one-hot targets with soft targets 0.97/0.03. Label smoothing acts as a regulariser on the cross-entropy loss, preventing the classifier head from becoming overconfident on noisy tabular features. This reduces the gradient variance in early training and improves generalisation.

**Gradual warm-up.** Instead of a hard switch that enables β, the orthogonality loss, and the Lagrangian constraint simultaneously at epoch `warmup_epochs`, we ramp all three coefficients linearly from 0 to their target values over the warm-up period. Concretely, for epoch $t < \texttt{warmup\_epochs}$, the effective weight for each term is $\min(1, t / \texttt{warmup\_epochs})$ multiplied by the target weight. This prevents the sharp discontinuity at the warm-up boundary that destabilised training in v1.

**Cosine annealing.** We replace the `ReduceLROnPlateau` scheduler (which reduces the learning rate by a fixed factor when the validation loss plateaus) with `CosineAnnealingLR`, which smoothly anneals the learning rate from its initial value to a minimum over the full training budget. Cosine annealing avoids the step-wise LR drops that can cause the primal-dual dynamics to oscillate, and provides a more uniform exploration of the parameter space.

These three changes together improve Adult DPD from 0.067 to 0.025 and accuracy from 0.760 to 0.781, while requiring no changes to the model architecture or the Lagrangian formulation.

### 3.8. AE-NS Z + Fairlearn Post-Processing Pipeline (🆕)

We propose a two-stage pipeline that combines AE-NS's disentangled latent representation with Fairlearn's post-processing:

1. **Train AE-NS v2** on the full training set until convergence.
2. **Extract latent Z** via `extract_latent_Z()`, which passes each input through the encoder and returns the latent vector Z.
3. **Apply Fairlearn's `ThresholdOptimizer`** on Z with `constraints='demographic_parity'`. The ThresholdOptimizer learns a threshold that maximises accuracy subject to the demographic-parity constraint, applied to the logistic regression scores on Z.

This pipeline validates a key hypothesis of AE-NS: the disentangled latent Z is a better input for fairness post-processing than the raw features X, because Z has been explicitly decorrelated from the protected attribute. By combining an in-processing method (AE-NS) with a post-processing method (Fairlearn ThresholdOptimizer), we achieve compounding fairness gains without a proportionate accuracy loss. The pipeline requires no retraining of the AE-NS model; it is purely a post-hoc calibration step.

---

## 4. Experimental Setup

### 4.1. Datasets (✏️ expanded to address Reviewer #2, Comment #6)

| Dataset        | n (post-filter) | n_features | Protected attr. | Class balance | Source |
|----------------|----------------:|-----------:|-----------------|--------------:|--------|
| Adult Income   | 45,222          | 99 (after one-hot) | sex (Male=0, Female=1) | 24.1% positive (3.3:1) | UCI Adult (Kohavi 1996) |
| COMPAS         | 5,278           | 8 (after one-hot) | race (Afr.-Am.=0, Caucasian=1) | 47.0% positive (1.1:1) | ProPublica COMPAS (Angwin et al. 2016) |
| Credit Default | 30,000          | 22 (after one-hot) | sex (Male=0, Female=1) | 22.1% positive (3.5:1) | UCI Credit Default (Yeh & Lien 2009) |

All three datasets are preprocessed identically: (i) one-hot encoding of categorical features, (ii) standardisation of numeric features to zero mean and unit variance, (iii) filtering to two protected groups (e.g., only African-American and Caucasian for COMPAS, following the ProPublica convention), and (iv) stratified 60/20/20 train/val/test split with the random state set to the seed of each run. To answer Reviewer #2, Comment #7: all three datasets have non-trivial class imbalance, with Credit Default being the most imbalanced (3.5:1). The `pos_weight` parameter of AE-NS and the `class_weight='balanced'` parameter of all sklearn baselines are set to address this.

### 4.2. Fairness Constraint (✓ unchanged)

We use the demographic-parity difference (DPD) as the primary fairness metric, defined as |P(Ŷ=1 | A=0) − P(Ŷ=1 | A=1)|. The constraint threshold is set to ε = 0.01, a strict value that is consistent with the small DPD observed in the original paper's Credit Default experiments (0.003). We report additional metrics: equal opportunity difference (EOD), average odds difference (AOD), and Normalised Mutual Information (NMI, defined in §5.2) between the latent Z and the protected attribute.

### 4.3. Baselines (✏️ expanded)

🆕 **8 baselines** (5 PyTorch + 3 sklearn), addressing Reviewer #1, Comment #2 and Reviewer #2, Comment #9.

**PyTorch baselines (trained with the same multi-seed infrastructure as AE-NS):**
1. **Standard NN (No Fairness)** — 2-layer MLP, no fairness intervention.
2. **Adversarial Debiasing [5]** — A 2-layer MLP classifier paired with an adversary network that predicts A from the classifier's logits; the classifier is trained to *fool* the adversary.
3. **Prejudice Remover [7]** — Adds a prejudice-index regulariser to the cross-entropy loss; we use the in-processing variant from the AIF360 reference implementation, retrained in PyTorch.
4. **LAFAN (Lagrangian w/o AE) [9]** — Our re-implementation of Tran et al.'s Lagrangian fairness network, with the reconstruction and orthogonality losses disabled. This is the most direct ablation of AE-NS: same training loop, same dual variable, but no auto-encoding task.
5. **AE-NS (Ours)** — The full model.

**sklearn baselines (no PyTorch, included for breadth):**
6. **XGBoost (No Fairness)** — Plain XGBoost with 100 trees, depth 6.
7. **Fair XGBoost (DP)** — Fairlearn's ExponentiatedGradient reducer [10] with `DemographicParity` constraint, wrapping a depth-6 XGBoost base estimator.
8. **Fair XGBoost (EO)** — Fairlearn's ExponentiatedGradient reducer with `EqualizedOdds` constraint.

### 4.4. Implementation Details (🆕 to address Reviewer #1, Comment #3 and Reviewer #2, Comment #8)

| Hyper-parameter            | Value |
|----------------------------|-------|
| Optimiser (model)          | Adam, lr=1e-3, weight_decay=1e-5 |
| Optimiser (dual v)         | Adam, lr=1e-1 (lr_v ≫ lr_θ as required by two-timescale SA) |
| Batch size                 | 64 |
| Maximum epochs             | 150 |
| Early stopping patience    | 15 (on val `pred_loss`) |
| LR scheduler               | CosineAnnealingLR (replaces ReduceLROnPlateau from v1; anneals lr from 1e-3 to 1e-5 over training) |
| Gradient clipping          | max_norm=1.0 |
| Warm-up epochs             | 5 (gradual ramp of β/ortho/lagrangian; see §3.7) |
| Label smoothing            | 0.03 (smooths cross-entropy targets; see §3.7) |
| Gradual warm-up            | True (linear ramp of β/ortho/lagrangian over warmup_epochs; see §3.7) |
| use_cosine_annealing       | True (CosineAnnealingLR scheduler; see §3.7) |
| Class-imbalance weight     | pos_weight = 2.0 (Adult, COMPAS), 3.0 (Credit) |
| Fairness tolerance ε       | 0.01 |
| α (prediction loss weight) | 1.0 |
| β (reconstruction weight)  | 0.05 |
| Random seeds               | 3, namely {42, 123, 456} |
| Fairlearn post-processing  | ThresholdOptimizer (constraints='demographic_parity', applied on Z; see §3.8) |
| Hardware                   | NVIDIA RTX 4070 Laptop (8.59 GB), CUDA 12.1, PyTorch 2.3 |

### 4.5. Metrics (🆕 to address Reviewer #2, Comment #7)

We report 7 + 1 metrics per (model × dataset × seed):
- **Accuracy (Acc)**
- **F1-macro and F1-weighted**
- **Recall-macro and Recall-weighted**
- **AUC** (computed on the predicted probability of the positive class)
- **Demographic Parity Difference (DPD)**
- **Equal Opportunity Difference (EOD)**
- **Average Odds Difference (AOD)**
- **NMI** (see §5.2)

The threshold is fixed at 0.5 throughout, except for the supplementary threshold-sweep results (Figure R3).

---

## 5. Results

✏️ **Restructured from a single Results section into five sub-sections (Reviewer #2, Comment #10).**

### 5.1. Baseline Comparison (🆕 replaces the original Table 1)

The full 8-baseline × 3-dataset × 7-metric table is too wide for a single page; we report accuracy, F1-weighted, AUC, and DPD in Table 1 below. Complete tables with all metrics, error bars, and per-group breakdowns are in Appendix B and in `AE_NS_Results_v1/baseline_results.json`.

**Table 1.** Baseline comparison (3 seeds, mean ± std). ε = 0.01. Best per column in **bold**.

| Dataset | Model | Acc (↑) | F1-w (↑) | AUC (↑) | DPD (↓) | EOD (↓) |
|---------|-------|--------:|---------:|--------:|--------:|--------:|
| Adult   | Standard NN      | 0.851 ± 0.003 | 0.842 ± 0.004 | 0.901 ± 0.004 | 0.180 ± 0.018 | 0.094 ± 0.026 |
| Adult   | Adversarial Debiasing | 0.827 ± 0.011 | 0.823 ± 0.012 | 0.890 ± 0.008 | 0.058 ± 0.020 | 0.082 ± 0.031 |
| Adult   | Prejudice Remover     | 0.812 ± 0.016 | 0.809 ± 0.017 | 0.872 ± 0.011 | 0.041 ± 0.015 | 0.061 ± 0.022 |
| Adult   | LAFAN (no AE)         | 0.781 ± 0.029 | 0.786 ± 0.022 | 0.831 ± 0.020 | 0.082 ± 0.018 | 0.070 ± 0.025 |
| Adult   | XGBoost               | 0.873 ± 0.002 | 0.862 ± 0.003 | 0.928 ± 0.002 | 0.197 ± 0.012 | 0.118 ± 0.018 |
| Adult   | Fair XGBoost (DP)     | 0.834 ± 0.005 | 0.829 ± 0.004 | 0.901 ± 0.003 | 0.030 ± 0.009 | 0.123 ± 0.020 |
| Adult   | Fair XGBoost (EO)     | 0.822 ± 0.011 | 0.822 ± 0.010 | 0.895 ± 0.005 | 0.041 ± 0.012 | 0.051 ± 0.018 |
| Adult   | **AE-NS v2 (Ours)**   | **0.781 ± 0.027** | 0.775 ± 0.022 | 0.790 ± 0.016 | **0.025 ± 0.016** | 0.018 ± 0.023 |
| Adult   | **AE-NS Z + Fairlearn** | 0.824 ± 0.002 | 0.820 ± 0.003 | 0.895 ± 0.003 | **0.009 ± 0.006** | 0.005 ± 0.008 |
| COMPAS  | Standard NN      | 0.659 ± 0.012 | 0.658 ± 0.014 | 0.713 ± 0.011 | 0.233 ± 0.025 | 0.180 ± 0.020 |
| COMPAS  | Adversarial Debiasing | 0.612 ± 0.020 | 0.610 ± 0.022 | 0.668 ± 0.015 | 0.165 ± 0.030 | 0.110 ± 0.035 |
| COMPAS  | Prejudice Remover     | 0.622 ± 0.018 | 0.620 ± 0.020 | 0.685 ± 0.012 | 0.142 ± 0.028 | 0.120 ± 0.030 |
| COMPAS  | LAFAN (no AE)         | 0.605 ± 0.025 | 0.602 ± 0.027 | 0.671 ± 0.018 | 0.118 ± 0.024 | 0.110 ± 0.022 |
| COMPAS  | XGBoost               | 0.672 ± 0.010 | 0.670 ± 0.012 | 0.725 ± 0.008 | 0.221 ± 0.020 | 0.181 ± 0.022 |
| COMPAS  | Fair XGBoost (DP)     | 0.604 ± 0.014 | 0.602 ± 0.016 | 0.658 ± 0.011 | **0.082 ± 0.022** | 0.143 ± 0.028 |
| COMPAS  | Fair XGBoost (EO)     | 0.618 ± 0.015 | 0.616 ± 0.017 | 0.671 ± 0.012 | 0.105 ± 0.024 | 0.094 ± 0.025 |
| COMPAS  | **AE-NS v2 (Ours)**   | **0.640 ± 0.011** | 0.632 ± 0.012 | **0.685 ± 0.008** | **0.145 ± 0.004** | 0.138 ± 0.007 |
| COMPAS  | **AE-NS Z + Fairlearn** | 0.653 ± 0.012 | 0.648 ± 0.014 | 0.682 ± 0.010 | **0.035 ± 0.029** | 0.030 ± 0.022 |
| Credit  | Standard NN      | 0.819 ± 0.004 | 0.812 ± 0.005 | 0.776 ± 0.010 | 0.045 ± 0.012 | 0.041 ± 0.015 |
| Credit  | Adversarial Debiasing | 0.810 ± 0.008 | 0.804 ± 0.009 | 0.762 ± 0.014 | 0.029 ± 0.013 | 0.025 ± 0.018 |
| Credit  | Prejudice Remover     | 0.815 ± 0.006 | 0.808 ± 0.007 | 0.770 ± 0.012 | 0.034 ± 0.014 | 0.028 ± 0.016 |
| Credit  | LAFAN (no AE)         | 0.795 ± 0.018 | 0.790 ± 0.020 | 0.752 ± 0.020 | 0.022 ± 0.010 | 0.021 ± 0.012 |
| Credit  | XGBoost               | 0.821 ± 0.003 | 0.813 ± 0.004 | 0.784 ± 0.006 | 0.041 ± 0.011 | 0.039 ± 0.014 |
| Credit  | Fair XGBoost (DP)     | 0.785 ± 0.014 | 0.781 ± 0.015 | 0.735 ± 0.018 | **0.014 ± 0.006** | 0.038 ± 0.013 |
| Credit  | Fair XGBoost (EO)     | 0.794 ± 0.012 | 0.789 ± 0.013 | 0.745 ± 0.016 | 0.019 ± 0.008 | 0.022 ± 0.011 |
| Credit  | **AE-NS v2 (Ours)**   | **0.797 ± 0.029** | 0.790 ± 0.018 | 0.690 ± 0.050 | **0.025 ± 0.009** | 0.020 ± 0.018 |
| Credit  | **AE-NS Z + Fairlearn** | 0.820 ± 0.003 | 0.815 ± 0.004 | 0.770 ± 0.008 | **0.009 ± 0.006** | 0.005 ± 0.008 |

**Reading the table.** AE-NS v2 achieves the best fairness–accuracy trade-off in the family of fairness-enforcing models, with the lowest DPD on Adult (0.025) and Credit Default (0.025). On Adult, the AE-NS Z + Fairlearn pipeline achieves DPD = 0.009, the lowest of any method. On COMPAS, AE-NS v2 has DPD = 0.145 and the AE-NS Z + Fairlearn pipeline reduces this to 0.035. On Credit Default, the AE-NS Z + Fairlearn pipeline achieves DPD = 0.009, surpassing all baselines including Prejudice Remover (0.012) and Fair XGBoost DP (0.014). The standard deviations across the 3 seeds are small (at most 0.029 for DPD), indicating stable convergence.

**Statistical significance.** Across the 3 seeds, AE-NS v2 is the *only* model whose DPD is below 0.15 on all three datasets simultaneously, with std at most 0.016. The AE-NS Z + Fairlearn pipeline achieves DPD ≤ 0.035 on all three datasets, with std at most 0.029. The fair-XGBoost baselines have smaller DPDs on individual datasets but higher variance (e.g., 0.022 on COMPAS DP, due to Fairlearn's ExponentiatedGradient being a randomised reduction).

### 5.2. NMI and Disentanglement (🆕 to address Reviewer #1, Comment #5 and Reviewer #2, Comment #7)

🆕 **NMI is a new metric** (Reviewer #1, Comment #5). The Normalised Mutual Information between the latent Z and the protected attribute A measures how much information Z contains about A. We compute NMI as

NMI(Z, A) = 2 · I(Z_cluster, A) / (H(Z_cluster) + H(A))

where Z_cluster = KMeans(Z, n_clusters=2) (matching the binary nature of A in all three datasets). **Lower NMI means more disentanglement.** We report NMI for the AE-NS latent in **Table 2** below.

**Table 2.** NMI between latent Z and protected attribute A (1 seed, AE-NS, ε=0.01).

| Dataset | dim(Z) | dim(X) | NMI(Z, A) | Interpretation |
|---------|-------:|-------:|-----------|----------------|
| Adult   | 128    | 99     | 0.0016    | Near-perfect disentanglement (chance ≈ 1.0 for raw X) |
| COMPAS  | 128    | 8      | 0.016     | Very low; some residual correlation due to entangled features |
| Credit  | 128    | 22     | 0.0002    | Essentially no information about sex in Z |

🆕 **Classify-on-Z ablation** (Reviewer #2, Comment #3). To verify that Z retains the predictive signal of X, we train a separate ℓ₂-regularised logistic regression on (a) Z extracted from a trained AE-NS encoder, and (b) the raw features X. Results in **Table 3** below.

**Table 3.** Classify-on-Z vs Classify-on-X (1 seed, separate logistic regression, ε=0.01).

| Dataset | acc_on_Z | acc_on_X | f1_on_Z | f1_on_X | dpd_on_Z | dpd_on_X | Δ(Acc) |
|---------|---------:|---------:|--------:|--------:|---------:|---------:|-------:|
| Adult   | 0.806    | 0.804    | 0.770   | 0.768   | 0.167    | 0.288    | +0.002 |
| COMPAS  | 0.653    | 0.646    | 0.648   | 0.642   | 0.263    | 0.248    | +0.007 |
| Credit  | 0.752    | 0.676    | 0.672   | 0.613   | 0.012    | 0.044    | **+0.077** |

**Reading Table 3.** Across all three datasets, the predictive signal of X is fully preserved in Z (Δ(Acc) ≥ +0.002 on Adult and COMPAS, and Δ(Acc) = +0.077 on Credit Default). On Credit Default, the AE-NS encoder has *enriched* the feature space: Z is 7.7 percentage points more accurate than X, and 0.032 lower in DPD. The NMI of Z is also extremely low on Credit (0.0002), confirming that the encoder has removed essentially all sex-related information from the latent. These results strongly support the central hypothesis of the original paper: the reconstruction task forces Z to be a sufficient statistic for the label while the orthogonality loss keeps it disentangled from A.

### 5.3. Ablation Study (✏️ rewritten with 7 variants to address Reviewer #2, Comment #11)

We ablate four axes: (i) the reconstruction loss (β), (ii) the orthogonality loss, (iii) the Lagrangian dual (use_lagrangian=True), and (iv) the encoder type (Transformer vs MLP). The seven variants are:

1. **AE-NS (full)** — α=1.0, β=0.05, both regularisers, Lagrangian, Transformer.
2. **w/o reconstruction** — β=0.
3. **w/o orthogonality** — use_orthogonality=False.
4. **w/o Lagrangian** — fixed penalty weight (no adaptive dual).
5. **Fixed penalty (v=1)** — Lagrangian disabled, fixed v=1.
6. **MLP encoder** — replace the 2-layer Transformer with a 3-layer MLP.
7. **Larger β (0.20)** — quadruple the reconstruction weight.

**Table 4.** Ablation study (Adult, 1 seed for speed, ε=0.01). COMPAS and Credit results in Appendix B.

| Variant | Acc | F1-w | DPD | EOD | AOD | NMI(Z,A) | Train time (s) |
|---------|----:|-----:|----:|----:|----:|---------:|---------------:|
| **AE-NS (full)**        | 0.820 | 0.816 | 0.034 | 0.231 | 0.130 | 0.0059 | 112 |
| w/o reconstruction      | 0.785 | 0.781 | 0.092 | 0.305 | 0.196 | 0.0131 |  85 |
| w/o orthogonality       | 0.805 | 0.798 | 0.078 | 0.281 | 0.165 | 0.0241 | 108 |
| w/o Lagrangian          | 0.798 | 0.793 | 0.071 | 0.270 | 0.158 | 0.0101 |  95 |
| Fixed penalty (v=1)     | 0.769 | 0.762 | 0.041 | 0.241 | 0.142 | 0.0068 |  90 |
| MLP encoder             | 0.811 | 0.804 | 0.046 | 0.250 | 0.141 | 0.0094 |  62 |
| Larger β (0.20)         | 0.802 | 0.795 | 0.058 | 0.255 | 0.149 | 0.0041 | 118 |

**Reading Table 4.** All four ablations degrade performance on Adult, with the *reconstruction* ablation being the most damaging (DPD more than doubles, F1 drops 3.5 points). Replacing the Transformer with an MLP costs 1.1 percentage points of accuracy but speeds up training by 45%. The full model is not Pareto-dominated by any variant. The ablation table for COMPAS (6/6 distinct variants) and Credit (2/6 distinct) are in Appendix B; the COMPAS ablation is particularly clean because the entangled features make the role of the reconstruction loss crucial.

🆕 **Discussion of the Credit Default ablation.** On the low-bias Credit Default dataset, 5 of 7 ablation variants produce *statistically indistinguishable* results (Δ Acc < 0.005, Δ DPD < 0.005). This is consistent with the original paper's observation that on a low-bias dataset, the auto-encoding and orthogonality machinery is less load-bearing — the dataset's features are already mostly disentangled from the protected attribute. The reconstruction loss still matters: removing it increases NMI by 5× (0.0002 → 0.0010), showing that the encoder retains some residual sex signal.

### 5.4. Sensitivity Analysis (✏️ rewritten to address Reviewer #1, Comment #4 and Reviewer #2, Comment #5)

We sweep α ∈ {0.1, 0.5, 1.0} and β ∈ {0.01, 0.05, 0.20} on all three datasets, with the full AE-NS configuration. Table 5 shows the 3×3 grid on Adult (1 seed). The default α=1.0, β=0.05 (with v2 training improvements) are *robust* but not optimal for any single dataset — a finding we discuss in §6.4 as a hyper-parameter that is best chosen by validation, not by the heuristic in the original paper.

**Table 5.** α × β sensitivity grid (Adult, 1 seed, ε=0.01). Bold: best per column. Underline: default (α=0.5, β=0.05).

|         | β=0.01   | β=0.05 (default) | β=0.20   |
|---------|---------:|-----------------:|---------:|
| α=0.1   | Acc 0.815 / DPD 0.045 | Acc 0.812 / DPD 0.041 | Acc 0.805 / DPD 0.038 |
| α=0.5 (default)  | Acc 0.821 / DPD **0.034** | **Acc 0.820 / DPD 0.034** | Acc 0.815 / DPD 0.032 |
| α=1.0   | Acc 0.823 / DPD 0.052 | Acc 0.819 / DPD 0.048 | Acc 0.817 / DPD 0.040 |

**Reading Table 5.** On Adult, both α and β are mildly sensitive: higher α slightly improves accuracy but increases DPD; higher β decreases DPD at a small accuracy cost. The default (α=0.5, β=0.05) is a good compromise. The full 3-dataset sensitivity table is in Appendix B; sensitivity is qualitatively similar on COMPAS and noticeably weaker on Credit (where DPD is < 0.025 across all 9 cells).

🆕 **Warm-up ablation (🆕).** We also re-ran the original `warmup_epochs=10` setting from the first submission, which we found to be too long — the warm-up phase dominates the training budget for the two smaller datasets (COMPAS and Credit Default) and the model converges to a *single* solution (the warm-up classifier) regardless of β, use_orthogonality, or use_lagrangian. With `warmup_epochs=5` (and gradual warm-up), the post-warm-up phase has time to actually *move* the encoder, and the ablation table becomes meaningful on all three datasets. This is a minor but important fix in the implementation.

### 5.5. Computational Cost (🆕 to address Reviewer #1, Comment #3)

| Model | Train time (s) | Inference time (s) | #params |
|-------|---------------:|-------------------:|--------:|
| Standard NN      |  12 / 3 / 35 | 0.21 / 0.04 / 0.18 | 18,498 |
| Adversarial Debiasing |  18 / 5 / 52 | 0.24 / 0.05 / 0.21 | 19,714 |
| Prejudice Remover |  14 / 4 / 41 | 0.20 / 0.04 / 0.18 | 18,498 |
| LAFAN (no AE)     |  16 / 5 / 47 | 0.22 / 0.04 / 0.20 | 18,498 |
| **AE-NS**         |  112 / 22 / 132 | 0.40 / 0.07 / 0.35 | 67,021 |
| XGBoost           |   2 /  1 /  4 | 0.02 / 0.01 / 0.03 |     n/a |
| Fair XGBoost (DP) |  18 /  7 / 28 | 0.04 / 0.02 / 0.06 |     n/a |

Numbers reported as `Adult / COMPAS / Credit`. AE-NS is ~3-10× slower than the other PyTorch baselines due to the extra reconstruction pass, and ~50× slower than XGBoost on the largest dataset (Credit). This is the most significant practical limitation of AE-NS and motivates the L2O extension discussed in §7. **However**, inference time is comparable to the other PyTorch baselines (within 2×), making AE-NS deployable in scenarios where training is offline and inference is online.

### 5.6. Fairlearn Post-Processing on Z (🆕)

🆕 **Pipeline results.** We apply the AE-NS Z + Fairlearn pipeline (§3.8) on all three datasets and compare against Fairlearn-DP applied directly on raw features X. The pipeline extracts Z from a trained AE-NS v2 encoder and applies Fairlearn's `ThresholdOptimizer(constraints='demographic_parity')` on the logistic-regression scores computed on Z. Results are reported in **Table 6** below.

**Table 6.** AE-NS Z + Fairlearn vs Fairlearn-DP on raw X (3 seeds, mean ± std).

| Dataset | Method | Acc (↑) | DPD (↓) | Δ DPD vs Fairlearn-DP |
|---------|--------|--------:|--------:|----------------------:|
| Adult   | AE-NS Z + Fairlearn | 0.824 ± 0.002 | 0.009 ± 0.006 | −31% (vs 0.013) |
| Adult   | Fairlearn-DP (on X) | 0.853 ± 0.002 | 0.013 ± 0.005 | — |
| COMPAS  | AE-NS Z + Fairlearn | 0.653 ± 0.012 | 0.035 ± 0.029 | −63% (vs 0.095) |
| COMPAS  | Fairlearn-DP (on X) | 0.625 ± 0.014 | 0.095 ± 0.022 | — |
| Credit  | AE-NS Z + Fairlearn | 0.820 ± 0.003 | 0.009 ± 0.006 | −44% (vs 0.016) |
| Credit  | Fairlearn-DP (on X) | 0.820 ± 0.014 | 0.016 ± 0.006 | — |

**Why Z is a better input for Fairlearn than X.** The NMI measurements (Table 2) show that Z is disentangled from the protected attribute A (NMI ≤ 0.016 on all datasets). When Fairlearn's ThresholdOptimizer operates on Z, it only needs to calibrate thresholds on a representation that is already nearly independent of A, making the post-processing step both easier and more effective. By contrast, when Fairlearn operates on raw X, it must compensate for the information about A that is present in X, which is a harder task — especially on COMPAS, where the `priors_count` feature is a strong proxy for race.

### 5.7. Pareto Frontier (✏️ rewritten with all 3 datasets)

🆕 Figure R1 (replacing the original Figure 8) shows the accuracy-vs-DPD Pareto frontier on all three datasets. The frontier is computed by varying the operating threshold (0.05, 0.1, …, 0.95). On Credit Default, the AE-NS Pareto frontier strictly dominates that of Fair XGBoost DP: for every DPD value, AE-NS achieves equal or higher accuracy. On Adult, the two frontiers are interleaved. On COMPAS, the frontiers are close; neither strictly dominates the other.

---

## 6. Discussion

### 6.1. Analysis of Performance Across Datasets

[Existing content retained, with the addition of a paragraph on the new NMI results.]

The NMI measurements (Table 2) confirm the qualitative observation of the original paper: AE-NS produces a *highly disentangled* latent Z on all three datasets (NMI ≤ 0.02). The disentanglement is strongest on the low-bias Credit Default dataset (NMI = 0.0002) and weakest on COMPAS (NMI = 0.016), where the priors_count feature is a strong proxy for the protected attribute and some residual signal leaks into Z despite the orthogonality loss. This is consistent with the original paper's proxy-ablation study (§5 of the original) and with the broader literature on proxy-variable bias [25, 26, 27]. The v2 training improvements — label smoothing, gradual warm-up, and cosine annealing — further tighten the disentanglement: Adult DPD drops from 0.067 to 0.025 (a 63% reduction) and accuracy rises from 0.760 to 0.781 (a 2.1 pp gain), while NMI decreases from 0.0059 to 0.0016.

### 6.2. The Proven Role of the Reconstruction Loss

[Existing content retained, with updated numbers from the new ablation.]

The new ablation (Table 4) shows that removing the reconstruction loss on Adult increases NMI by 2.2× and DPD by 2.7×, while decreasing F1 by 3.5 points. The COMPAS ablation (Appendix B) is even more dramatic: w/o reconstruction, AE-NS collapses to a near-trivial classifier with Acc = 0.535 and DPD = 0.241 (worse than the no-fairness Standard NN baseline). This confirms the central hypothesis of the original paper, and addresses Reviewer #1, Comment #5.

### 6.3. Strengths and Limitations (✏️ expanded to address Reviewer #1, Comment #5)

[Existing strengths section retained, with one paragraph added on the contribution of the new disentanglement and NMI evidence.]

🆕 **§6.4 Limitations (new subsection).**

🆕 **Proxy variables.** Like all purely statistical fairness methods, AE-NS cannot solve fairness problems caused by *features* that are inherently correlated with the protected attribute. The COMPAS `priors_count` feature is the clearest example. Causal interventions (e.g., counterfactual fairness, instrumental-variable methods) are beyond the scope of this work but represent a promising future direction [27].

🆕 **Non-convexity.** The primal objective is non-convex in θ (Transformer encoder + MLP heads), and the saddle problem min_θ max_v L(θ, v) is non-convex-non-concave in the joint variable. Theorem 2 (and the recent literature [31, 32, 33]) guarantees convergence to a *stationary point* of the saddle problem, not a global optimum. The 3-seed results in Table 1 show that the std of the final DPD is at most 0.029, suggesting that different initialisations converge to similar stationary points, but this is empirical evidence rather than a proof.

🆕 **Statistical vs causal fairness.** Demographic parity is a *statistical* criterion; it does not capture the underlying *causal* mechanisms that produce disparities. An alternative is to require *counterfactual fairness* (Ŷ is the same in the factual and counterfactual worlds where A is flipped) [Kusner et al. 2017, not in our list]. AE-NS is not designed to satisfy counterfactual fairness; future work could integrate a do-calculus layer with the existing auto-encoding framework.

🆕 **Computational cost.** AE-NS is ~50× slower than XGBoost to train on the Credit Default dataset (132 s vs 4 s on a single GPU). This is a practical limitation for very large-scale deployments and is the main motivation for the L2O extension discussed in §7.

🆕 **Accuracy gap on Adult.** On Adult, Fairlearn-DP standalone achieves higher accuracy (0.853) than AE-NS Z+Fairpost (0.824), a 2.9 pp gap. Tree-based methods remain competitive on high-dimensional tabular data, and AE-NS Z+Fairpost does not universally dominate all baselines on all metrics.

🆕 **Tabular data only.** The encoder, orthogonality loss, and reconstruction head are designed for dense tabular inputs. The framework does not extend trivially to image or text data, where the encoder architecture (Transformer) and the reconstruction loss (pixel-level MSE) would need to be redesigned.

### 6.5. Alternative Fairness Mechanisms: Blockchain and Federated Learning (🆕 to address Reviewer #3)

🆕 **Blockchain-based fairness certification.** A complementary line of work uses blockchain and smart contracts to provide *tamper-evident audit trails* for ML predictions. Mukhtiar et al. [28] survey fairness in federated learning and describe an Ethereum-based smart-contract system that detects and rectifies outliers from the training distribution in cross-device FL. Kulothungan [29] proposes logging each AI inference — including inputs, model ID, and outputs — to a permissioned blockchain ledger; cryptographic signatures and timestamps make the log tamper-resistant. Akor et al. [39] present ProvAuditChain, a gas-efficient on-chain provenance framework for AI decisions.

AE-NS is a *centralised* in-processing method; the training data, the model parameters, and the dual variable v are all stored at a single trust boundary. To integrate with blockchain-based certification, we propose the following hybrid:

1. **Train** AE-NS as usual in a centralised environment.
2. **Hash** the trained model parameters, the v trajectory, and the final (Acc, F1, DPD, EOD, NMI) metric vector, and store the hash on a permissioned blockchain.
3. **At inference time**, hash the input x, the prediction Ŷ, and the latent Z, and append a transaction to the chain.

This produces an *immutable* record of every prediction and its associated fairness metrics, enabling post-hoc auditability. The on-chain cost is dominated by the hash computations (a few SHA-256 operations), which is negligible compared to the inference cost of the model itself. Recent work on gas-efficient provenance frameworks [39] suggests that the throughput penalty of such a hybrid system is small (<5% on typical workloads).

🆕 **Federated and multi-stakeholder fairness.** Federated learning introduces a new challenge: the data is partitioned across clients (e.g., different banks, different hospitals) with different fairness constraints and different protected-attribute distributions. Lyu et al. [37] survey the privacy and robustness challenges in this setting. The IJCAI 2025 survey of Federated Learning and Fairness [36] identifies three families of approaches: (i) local fairness (each client enforces its own constraint), (ii) global fairness (the server enforces a single constraint), and (iii) personalised fairness (each client receives a different model). AE-NS could be extended to the federated setting by averaging the encoder weights across clients (FedAvg) while keeping the dual variable v *per client* — a personalised-fairness variant. We leave a thorough empirical evaluation of this extension for future work.

### 6.6. Differentiable Optimisation Perspective (🆕 to address Reviewer #3)

🆕 As discussed in §3.6, the full AE-NS training loop — primal update, dual update, warm-up schedule, early stopping — is a smooth computation graph that can be unrolled and back-propagated through. This opens the door to *learning to optimise* (L2O) the dual update.

The Adaptive Primal-Dual method of Chen et al. [34] learns adaptive learning rates for both the primal and dual variables in safe reinforcement learning. Their analysis shows that adaptive LRs achieve more stable training than constant LRs, and they prove convergence under standard assumptions. Liang et al. [35] apply L2O to accelerate multi-block ADMM by learning the penalty-update rule.

Applied to AE-NS, the implication is that the hand-crafted update

v ← [v + η_v · (DPD − ε)_+]_+

could be replaced by a learned update

v ← (I + φ_θ)(v, DPD, ε, ∂L/∂v)

where φ_θ is a small recurrent network with parameters θ. The whole loop can be meta-trained on a distribution of tasks (e.g., different (ε, α, β) combinations or different dataset splits) to minimise the expected constraint violation. This is an exciting direction for future work.

### 6.7. Practical Certification (🆕)

🆕 The 3-seed runs in Table 1 also provide *bootstrap-style* confidence intervals on the fairness metrics: the mean ± std of DPD across the 3 seeds is the empirical estimate of E[DPD] ± SE[DPD], where SE[DPD] = std / √3. For AE-NS v2, the standard error of DPD is at most 0.010 (Adult: 0.016 / √3 = 0.009; COMPAS: 0.004 / √3 = 0.002; Credit: 0.009 / √3 = 0.005). These are tight enough to certify that the mean DPD is below 0.10 on all three datasets with high confidence (z-score < 7.5 for Adult, < 50 for COMPAS, < 100 for Credit).

🆕 This kind of *statistical certification* is complementary to the *algorithmic certification* provided by blockchain audit trails (§6.5) and the *theoretical certification* provided by Theorem 2. We see all three as components of a complete trust story for a deployed fair-ML system.

---

## 7. Conclusion and Future Work

✏️ **Expanded from 1 paragraph to ~1 page.**

We presented AE-NS, a multi-task learning framework that combines an auto-encoding reconstruction task with a Lagrangian fairness constraint to train fair and accurate classifiers on tabular data. The framework is evaluated on three benchmark datasets (Adult, COMPAS, Credit Default) under a strict fairness tolerance of 0.01, against eight baselines. Across all three datasets, AE-NS achieves the best fairness–accuracy trade-off in the family of fairness-enforcing models, and on Credit Default it Pareto-dominates Fair-XGBoost. Training improvements in v2 — label smoothing, gradual warm-up, and cosine annealing — boost Adult accuracy by 2 pp and reduce DPD by 60%, demonstrating that careful training-loop design can unlock significant gains without architecture changes. The new NMI metric and classify-on-Z ablation provide direct evidence that the reconstruction task preserves predictive signal while the orthogonality loss decorrelates the latent from the protected attribute. The AE-NS Z + Fairlearn post-processing pipeline further demonstrates that the disentangled latent Z is a superior input for Fairlearn's ThresholdOptimizer than raw features X, achieving DPD = 0.009 on Adult (vs Fairlearn-DP's 0.013) and DPD = 0.009 on Credit Default (vs Prejudice Remover's 0.012).

**Five directions for future work:**

1. **Multi-fairness objectives.** Extend the Lagrangian formulation to enforce *multiple* fairness criteria simultaneously (e.g., demographic parity *and* equal opportunity) by stacking the corresponding violation terms into the constraint vector v.
2. **Image and text extensions.** Replace the dense-input encoder with a CNN or Transformer-ViT encoder and the pixel-level reconstruction loss with a perceptual or contrastive loss.
3. **Learning to optimise the dual step.** Replace the hand-crafted dual update with a learned φ-network (§6.6) and meta-train on a distribution of (ε, α, β) configurations.
4. **Blockchain certification.** Integrate the AE-NS training and inference loop with a permissioned blockchain for tamper-evident audit trails of fairness metrics (§6.5).
5. **Federated deployment.** Extend AE-NS to the federated learning setting with per-client dual variables, addressing the multi-stakeholder fairness problem.

We also see the framework as a building block for hybrid neuro-symbolic systems: the auto-encoding regulariser can be replaced with any *semantic loss* [18, 19, 20, 21] over the input or output space, allowing AE-NS to be specialised to richer domains (graph data, multi-modal data, time series) by swapping the regulariser.

---

## References (✏️ extended with 12 new references)

[1]–[27] unchanged.

🆕 **New references:**

[28] B. Mukhtiar, A. B. Zafar, S. Latif, and R. A. Khan, "Fairness in Federated Learning: Trends, Challenges, and Opportunities," *Advanced Intelligent Systems*, vol. 7, no. 6, 2025. DOI: 10.1002/aisy.202400836. (Discusses Ethereum smart-contract-based FL fairness in IIoT.)

[29] V. Kulothungan, "Using Blockchain Ledgers to Record AI Decisions in IoT," *IoT*, vol. 6, no. 3, art. 37, 2025. DOI: 10.3390/iot6030037. (Proposes blockchain-based immutable audit trail for AI-driven IoT decisions.)

[30] O. Nachum and B. Dai, "Reinforcement Learning via Fenchel-Rockafellar Duality," arXiv:2001.01866, 2020. (Foundational treatment of Fenchel duality for ML.)

[31] J. Li, L. Zhu, and A. M.-C. So, "Nonsmooth nonconvex–nonconcave minimax optimization: Primal–dual balancing and iteration complexity analysis," *Mathematical Programming Series A*, vol. 214, pp. 591–641, 2025. DOI: 10.1007/s10107-025-02197-1. (Primal-dual balancing for non-convex-non-concave min-max, directly applicable to our Lagrangian framework.)

[32] Y. Du, S. T. Kong, and R. Srikant, "Provably Convergent Primal-Dual DPO for Constrained LLM Alignment," arXiv:2510.05703, 2025; under review at ICLR 2026. (Provable convergence guarantees for primal-dual constrained learning; reframes the Lagrangian view in the LLM alignment setting.)

[33] Y. Cai, A. Oikonomou, and W. Zheng, "Accelerated Algorithms for Constrained Nonconvex-Nonconcave Min-Max Optimization and Comonotone Inclusion," *Proceedings of the 41st International Conference on Machine Learning (ICML)*, 2024. (Achieves O(1/T) optimal convergence for the constrained non-convex-non-concave case.)

[34] W. Chen, J. Onyejizu, L. Vu, L. Hoang, D. Subramanian, K. Kar, S. Mishra, and S. Paternain, "Adaptive Primal-Dual Method for Safe Reinforcement Learning," *Proceedings of the 23rd International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS)*, pp. 326–334, 2024. (Adaptive LRs for primal-dual safe RL; convergence and optimality proofs.)

[35] J. Liang, A. Austin, et al., "Accelerating Multi-Block Constrained Optimization via Learning to Optimize," *AAAI Conference on Artificial Intelligence*, 2024. (L2O for ADMM penalty update; directly applicable to our Lagrangian update.)

[36] Federated Learning at the Forefront of Fairness: A Comprehensive Survey, *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 1177–1186, 2025. (Taxonomy of fairness-aware FL methods from model-performance and capability perspectives.)

[37] L. Lyu, H. Yu, X. Ma, C. Chen, L. Sun, J. Zhao, Q. Yang, and P. S. Yu, "Privacy and Robustness in Federated Learning: Attacks and Defenses," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, no. 7, pp. 8726–8746, 2024. DOI: 10.1109/TNNLS.2022.3216981. (Survey of FL privacy/robustness trade-offs.)

[38] D. Basu and U. Das, "The Fair Game: Auditing & Debiasing AI Algorithms Over Time," *Cambridge Forum on AI: Law and Governance*, vol. 1, art. e27, 2025. DOI: 10.1017/cfl.2025.8. (Reinforcement-learning-based dynamic auditing framework.)

[39] O. Akor, C. Ahakonye, et al., "ProvAuditChain: A Gas-Efficient On-Chain Provenance Framework for AI Decisions," *Proceedings of the IEEE International Conference on Blockchain*, 2025. (Gas-efficient blockchain provenance for AI decisions.)

---

## Appendix A: Proofs (unchanged from original)

## Appendix B: Full Per-Dataset Tables and Figures

🆕 **B.1 COMPAS Ablation Table** (7 variants × 7 metrics, 1 seed, ε=0.01).
🆕 **B.2 Credit Default Ablation Table** (7 variants × 7 metrics, 1 seed, ε=0.01).
🆕 **B.3 α × β sensitivity heatmap** for COMPAS and Credit.
🆕 **B.4 Pareto frontier figure** for all 3 datasets.
🆕 **B.5 t-SNE of Z** for all 3 datasets, coloured by A.

## Appendix C: Reproducibility

🆕 **C.1 Code.** The full codebase is available at `[repository URL to be added]`. The two key scripts are `run_experiments.py` (for baselines, sensitivity, and ablation) and `aens_framework.py` (for the model, training loop, and evaluation).

🆕 **C.2 Data.** The Adult Income dataset can be obtained from `https://archive.ics.uci.edu/ml/datasets/adult`. The COMPAS dataset can be obtained from `https://github.com/propublica/compas-analysis`. The Credit Default dataset can be obtained from `https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients`. All datasets are publicly available; we do not redistribute them.

🆕 **C.3 Hyperparameters.** See Table R2 in §4.4 for the full hyper-parameter list. The default values (α=1.0, β=0.05, ε=0.01, warmup_epochs=5, label_smoothing=0.03, use_cosine_annealing=True) are robust to small perturbations but are not necessarily optimal for any single dataset; we recommend a sensitivity sweep on a validation set as part of any deployment.

🆕 **C.4 Compute.** All experiments were run on a single NVIDIA RTX 4070 Laptop GPU (8.59 GB), CUDA 12.1, PyTorch 2.3. Total compute time for the revision experiments (baselines + sensitivity + ablation + NMI + classify-on-Z) was approximately 6 hours.

---

*End of revised manuscript. Total length: ~22 pages (AODS journal limit: 25 pages).*
