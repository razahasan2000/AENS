# Response Letter to Reviewers

**Manuscript:** AE-NS: An Auto-Encoding Network with Semantic Constraints for Robust Fair Classification on Tabular Data
**Manuscript ID:** AODS-D-26-00398
**Authors:** [Author list]
**Date:** 2 June 2026

---

## Dear Editor and Reviewers,

We thank you for the careful and constructive review of our manuscript. The three reviewers raised 20 distinct concerns spanning methodology, experimental design, reproducibility, and broader impact. We have addressed every concern, and in many cases have added new experiments, new metrics, and new sections to the manuscript. Below we provide a point-by-point response. For the reviewers' convenience, the line numbers below refer to the revised `paper_revision.md` document.

A summary of the most important changes:

1. **Replaced the original 2-baseline comparison with an 8-baseline comparison** (5 PyTorch + 3 sklearn), reporting mean ± std over 3 random seeds.
2. **Added a Normalised Mutual Information (NMI) metric** between the latent Z and the protected attribute A.
3. **Added a "classify-on-Z" experiment** that directly verifies that Z retains the predictive signal of X.
4. **Added a sensitivity sweep** for α and β on all three datasets.
5. **Added a 7-variant ablation study** on all three datasets.
6. **Added a computational cost table** (training time, inference time, #parameters).
7. **Restructured the Results section** into 5 sub-sections (Reviewer #2, Comment #10).
8. **Added a new §6.4 Limitations subsection** and a new §6.5 on alternative fairness mechanisms (blockchain, federated, learning-to-optimise).
9. **Added 12 new references**, including recent (2023-2026) work on non-convex primal-dual optimisation, learning-to-optimise, blockchain-based AI auditing, and federated fairness.
10. **Re-titled the framework** as "AE-NS — An Auto-Encoding Network with Semantic Constraints" to address Reviewer #1's concern that "neuro-symbolic" was misleading; the body text now clarifies the connection to semantic-loss neuro-symbolic AI.
11. **Introduced v2 training improvements** (label smoothing, gradual warm-up, cosine annealing) that boost Adult accuracy by 2 pp (0.760 → 0.781) and reduce DPD by 60% (0.067 → 0.025), with similar gains on COMPAS and Credit Default.
12. **Proposed an AE-NS Z + Fairlearn post-processing pipeline** that extracts the disentangled latent Z and applies Fairlearn's ThresholdOptimizer, achieving DPD = 0.009 on Adult (vs. Fairlearn-DP's 0.013) and DPD = 0.009 on Credit Default (vs. Prejudice Remover's 0.012), with 3-seed mean ± std.

We believe the revised manuscript is significantly stronger than the original submission and addresses all 20 reviewer comments in full. The detailed responses follow.

---

## Reviewer 1

> **Comment 1:** *"The introduction section is too weak. The 'neuro-symbolic' positioning is misleading — the work is closer to multi-task learning with a reconstruction regulariser than to logic-based neuro-symbolic systems. Additionally, the claim of robustness in the abstract is not supported by any OOD (out-of-distribution) experiments."*

**Response:** We have re-titled the manuscript as "AE-NS — An Auto-Encoding Network with Semantic Constraints for Robust Fair Classification on Tabular Data" (new subtitle, unchanged first part). The introduction (§1, `paper_revision.md` lines 31-69) is now expanded to ~1.5 pages and explicitly frames AE-NS as a *multi-task learning* approach with adaptive constraints, while clarifying in §2.3 (lines 91-99) that the neuro-symbolic connection is through the *semantic loss* sub-field of neuro-symbolic AI (Xu et al. [18], Ahmed et al. [21]) rather than through logic-based inference. The word "neuro-symbolic" now appears only in the acronym AE-NS and in the explicit reframing paragraph of §2.3.

Regarding OOD experiments: the "robustness" claim in the original abstract was about *training stability* (the model does not collapse to a trivial solution) rather than *OOD generalisation*. We have re-phrased the abstract to make this explicit. We have *not* added OOD experiments because the scope of the paper is fairness on in-distribution data, but we acknowledge in §6.4 (Limitations) that OOD robustness is an interesting future direction.

> **Comment 2:** *"The comparison with the existing literature is insufficient. The two baselines (XGBoost and Fairlearn-XGBoost) are too few — the work should be compared to at least one deep learning baseline that addresses fairness, such as adversarial debiasing."*

**Response:** We have added 6 new baselines, for a total of **8 baselines** (5 PyTorch + 3 sklearn). The new baselines are: Standard NN (no fairness), Adversarial Debiasing [5], Prejudice Remover [7], LAFAN (Lagrangian w/o AE, our re-implementation of Tran et al. [9]), and Fair XGBoost with EqualizedOdds constraint. The full list is in §4.3 (lines 168-189). The new Table 1 in §5.1 (lines 200-225) shows AE-NS v2 is competitive with or superior to all 8 baselines on the fairness-accuracy trade-off. We further introduce an **AE-NS Z + Fairlearn post-processing pipeline** (§3.8, §5.6) that achieves DPD = 0.009 on Adult (vs. Fairlearn-DP's 0.013) and DPD = 0.009 on Credit Default (vs. Prejudice Remover's 0.012), demonstrating that AE-NS's disentangled representations improve downstream fairness. The per-seed CSV with all metrics is in `AE_NS_Results_v1/baseline_comparison_table.csv` and `AE_NS_Results_v2/`.

> **Comment 3:** *"The experimental setup is missing many critical details: the data split ratio, model architecture, hyperparameters, and the number of random seeds used. The reviewers cannot reproduce the results."*

**Response:** We have added a dedicated **§4.4 Implementation Details table** (lines 195-211) that lists every hyperparameter, the optimizer, the learning rate, the batch size, the early-stopping patience, the warm-up schedule, the random seeds, the fairness tolerance, and the hardware. The data split ratio (60/20/20 stratified) is mentioned in §4.1 (line 132). The full reproducibility information is summarised in **Appendix C** (lines 446-466), including the URLs for the three datasets, the names of the two key Python scripts, and the total compute time.

> **Comment 4:** *"The choice of α=0.5 and β=0.05 is heuristic. A sensitivity analysis is required to justify the choice."*

**Response:** We have added a **3×3 sensitivity grid** (α ∈ {0.1, 0.5, 1.0} × β ∈ {0.01, 0.05, 0.20}) on all three datasets, reported in §5.4 (lines 290-310) and Table 5. The full grid is in `AE_NS_Results_v1/sensitivity_table.csv`. We further sweep α ∈ {0.1, 0.3, 0.5, 0.7, 1.0} with v2 training improvements (label smoothing, cosine annealing, gradual warm-up). The optimal configuration is **α=1.0** (up from 0.5), which puts more weight on prediction loss and paradoxically improves fairness by keeping the encoder focused on discriminative representations. We now recommend α=1.0 as the default (§4.4, Table R2). The full v2 sensitivity results are in `AE_NS_Results_v2/*_sensitivity_*.json`.

> **Comment 5:** *"The limitations section is incomplete. The paper should discuss (a) what fairness criteria beyond demographic parity and equal opportunity could be considered, (b) the limitations of the Lagrangian dual approach in non-convex settings, and (c) the use of a Normalised Mutual Information (NMI) metric between the latent Z and the protected attribute A to verify disentanglement."*

**Response:** We have added a dedicated **§6.4 Limitations** subsection (lines 359-380) that discusses five limitations: proxy variables, non-convexity, statistical vs causal fairness, computational cost, and tabular-data-only. We have added **NMI** as a new metric in §4.5 (line 219) and §5.2 (lines 245-265); Table 2 reports NMI on all three datasets. We have also added the **classify-on-Z** experiment (Table 3, lines 267-275) that empirically verifies the disentanglement claim by showing that the downstream classifier on Z matches or exceeds the classifier on X. NMI is now part of the `evaluate_model` function in `aens_framework.py` (line 1111 in the source file) and is reported alongside the standard 7 metrics in all experiments.

---

## Reviewer 2

> **Comment 1:** *"The introduction section is too concise. The paper would benefit from a more detailed introduction, with a clear motivation, a clear statement of the contributions, and a brief description of the organisation of the paper."*

**Response:** The introduction has been expanded to ~1.5 pages (lines 31-69) and now includes: (i) the broader context of algorithmic fairness and the three families of interventions (pre-, in-, post-processing), (ii) the position of AE-NS within the in-processing family, (iii) a clear statement of the **three gaps** that motivate the revision, (iv) a **four-item list of contributions**, and (v) a **paragraph on the paper organisation**. We hope this addresses your concern.

> **Comment 2:** *"A summary of the related literature is required. The current Related Work section jumps between topics without giving the reader a clear sense of how the proposed work fits in."*

**Response:** We have added **§2.4 Positioning of AE-NS** (lines 105-120), a summary table that compares AE-NS with five representative prior methods (Adversarial Debiasing, Prejudice Remover, LAFAN, Fairlearn ExpGrad, Neural-symbolic fairness) along three axes (auxiliary task, fairness constraint, optimisation scheme). The table makes the contributions of AE-NS concrete: AE-NS is the only method in the comparison that uses both a *reconstruction* auxiliary task *and* a Lagrangian constraint, which is the combination that allows it to find predictive-but-fair representations on the COMPAS dataset.

> **Comment 3:** *"The architecture description is unclear. What happens after the latent representation Z is computed? Is the classifier head cascaded after the decoder? Is the decoder a separate path or part of the same forward pass?"*

**Response:** We have significantly expanded **§3.2 Architecture** (lines 124-148) to clarify that the classifier and decoder are *parallel* heads — Z is shared, not cascaded. The key paragraphs explain: (a) the parallel architecture is necessary to avoid information leakage from the decoder to the classifier, (b) the shallow decoder is necessary to prevent the reconstruction task from dominating the early training, and (c) the new **classify-on-Z experiment** (Table 3) directly verifies that Z retains the predictive signal of X (and exceeds it on Credit Default, where Z is 8.2 percentage points more accurate than X).

> **Comment 4:** *"Why is the dual variable v detached in the primal gradient? This looks like an arbitrary implementation choice that needs justification."*

**Response:** We have added a dedicated paragraph in **§3.4** (lines 158-167) that justifies the `detach(v)` operation. The two-timescale stochastic-approximation literature [9, 24, 32] requires that the primal update treats v as a constant (i.e., detaches it in ∂L/∂θ) and the dual update treats θ as a constant. This is equivalent to an *alternating* gradient-descent-ascent scheme and is the standard implementation. We have cited three relevant references that confirm this practice. The docstring of the `compute_loss` method in `aens_framework.py` (line 380-388 in the source file) also explains this in detail.

> **Comment 5:** *"A sensitivity analysis for the hyper-parameters α and β is required."*

**Response:** Addressed — see Reviewer 1, Comment 4. The 3×3 sensitivity grid on all three datasets is in §5.4 (lines 290-310).

> **Comment 6:** *"The dataset descriptions are too simplistic. The number of samples, the number of features, the class balance, and the protected attribute should be clearly stated for each dataset."*

**Response:** We have expanded **§4.1 Datasets** (lines 130-145) into a structured table that reports, for each of the three datasets, the number of samples (post-filter), the number of features (after one-hot encoding), the protected attribute and its encoding, the class balance, the imbalance ratio, and the source URL. The class-imbalance issue (Reviewer #2, Comment #7) is also addressed in this paragraph.

> **Comment 7:** *"The class imbalance problem is not discussed. Only accuracy is reported; F1, recall, and AUC should also be reported for imbalanced datasets."*

**Response:** We have added F1-macro, F1-weighted, recall-macro, recall-weighted, and AUC to the reported metrics (Table 1, lines 200-225). The NMI is also reported as a new metric (Table 2, lines 252-256). The full per-seed metric vector (8 metrics × 3 seeds × 8 models × 3 datasets) is in `AE_NS_Results_v1/baseline_results.json` and the aggregate CSV in `AE_NS_Results_v1/baseline_comparison_table.csv`.

> **Comment 8:** *"The experimental results do not report standard deviations or error bars. Multiple random seeds should be used."*

**Response:** We have re-run the entire experiment suite with **3 random seeds** (42, 123, 456). Every entry in Table 1 is reported as mean ± std. The per-seed values are in `AE_NS_Results_v1/{adult,compas,credit}/baseline_results.json`. The standard error of the mean for AE-NS's DPD is at most 0.018 (Adult: 0.025/√3, COMPAS: 0.031/√3, Credit: 0.008/√3), giving strong statistical evidence that the mean DPD is below 0.10 on all three datasets.

> **Comment 9:** *"The baseline comparison is inadequate. Only two XGBoost-based baselines are used."*

**Response:** Addressed — see Reviewer 1, Comment 2. We now have **8 baselines** spanning 5 PyTorch models and 3 sklearn models.

> **Comment 10:** *"The results section is not well organized. It jumps between tables and figures without a clear structure."*

**Response:** We have restructured the Results section into **5 sub-sections** (§5.1 Baselines, §5.2 NMI and Disentanglement, §5.3 Ablation, §5.4 Sensitivity, §5.5 Computational Cost, §5.6 Pareto Frontier), each with its own introduction, table, and reading-the-table paragraph.

> **Comment 11:** *"The ablation study is insufficient. Only one variant is tested; several other variants should be considered."*

**Response:** We have expanded the ablation to **7 variants** (§5.3, lines 277-307): the full model, w/o reconstruction, w/o orthogonality, w/o Lagrangian, fixed penalty v=1, MLP encoder, and larger β=0.20. The ablation is run on all three datasets; the COMPAS ablation (6/6 distinct variants) is particularly clean.

> **Comment 12:** *"The paper would benefit from a discussion of future work and the inclusion of more recent references (2023-2026)."*

**Response:** The Conclusion (§7, lines 404-424) now lists **five concrete directions for future work**: multi-fairness objectives, image/text extensions, learning-to-optimise the dual step, blockchain certification, and federated deployment. The reference list has been extended with **12 new references** [28]-[39] covering non-convex primal-dual optimisation (2023-2025), learning-to-optimise (2024-2025), blockchain-based AI auditing (2025), federated fairness (2024-2025), differentiable programming (2020, 2024-2025), and trustworthy AI (2024-2025).

---

## Reviewer 3

> **Topic 1: Blockchain-enabled training.** *"The paper should discuss blockchain-enabled training of the model as an alternative or complementary approach to the Lagrangian framework. This would make the paper more relevant to the current literature on decentralised trust in AI."*

**Response:** We have added a dedicated **§6.5 Alternative Fairness Mechanisms** subsection (lines 383-400) that discusses three alternative mechanisms: (i) blockchain-based fairness certification, (ii) federated learning, and (iii) hybrid systems. We propose a concrete hybrid architecture where the AE-NS model is trained in a centralised environment and the resulting model parameters, dual variable v trajectory, and final fairness metrics are hashed and stored on a permissioned blockchain. The on-chain throughput penalty is estimated to be <5% based on the gas-efficient provenance framework of Akor et al. [39]. We cite Mukhtiar et al. [28], Kulothungan [29], and Akor et al. [39] as primary references for the blockchain-AI auditing literature.

> **Topic 2: Fenchel-Lagrange duality and differentiable programming.** *"The paper would benefit from a more detailed discussion of Fenchel-Lagrange duality and the differentiable programming view of the Lagrangian update. The connection to learning to optimise (L2O) should be made explicit."*

**Response:** We have added a dedicated **§3.6 A Fenchel-Lagrange Perspective** (lines 188-208) that derives the Fenchel-Rockafellar dual of the AE-NS constrained problem and shows that the resulting saddle-point problem is equivalent to the Lagrangian dual. We have also added a **§6.6 Differentiable Optimisation Perspective** (lines 401-410) that explains how the entire AE-NS training loop is a smooth computation graph that can be unrolled and back-propagated through. We explicitly connect this to learning-to-optimise (L2O) and propose replacing the hand-crafted dual update with a learned φ-network. We cite Nachum and Dai [30] for the foundational Fenchel duality treatment, Li et al. [31] for primal-dual balancing in non-convex-non-concave settings, and Chen et al. [34] and Liang et al. [35] for adaptive primal-dual L2O methods.

> **Topic 3: Limitations of the Lagrangian approach and certification.** *"The paper should discuss the limitations of the Lagrangian approach (in particular, the lack of a global optimality guarantee for non-convex problems) and the relationship to statistical and algorithmic certification of the fairness constraint."*

**Response:** We have added a dedicated **§6.7 Practical Certification** (lines 411-419) that discusses three complementary forms of certification: (i) *statistical certification* using the 3-seed bootstrap CIs on the fairness metrics, (ii) *algorithmic certification* using the blockchain audit trail (cross-referenced from §6.5), and (iii) *theoretical certification* from Theorem 2 (the O(1/√T) convergence rate of the primal-dual scheme). The non-convexity limitation is now discussed in detail in **§6.4** (lines 365-368), with references to the recent primal-dual convergence literature [31, 32, 33].

---

## Summary of New Experiments

| Experiment | Purpose | Datasets | Seeds | Location |
|------------|---------|----------|-------|----------|
| 8-baseline comparison | Reviewers 1#2, 2#9 | All 3 | 3 | `AE_NS_Results_v1/baseline_comparison_table.csv` |
| Sensitivity sweep (α × β) | Reviewers 1#4, 2#5 | All 3 | 1 | `AE_NS_Results_v1/sensitivity_table.csv` |
| 7-variant ablation | Reviewer 2#11 | All 3 | 1 | `AE_NS_Results_v1/ablation_table.csv` |
| NMI metric | Reviewer 1#5 | All 3 | 1 | `AE_NS_Results_v2/*_revision.json` |
| Classify-on-Z | Reviewer 2#3 | All 3 | 1 | `AE_NS_Results_v2/*_revision.json` |
| Computational cost | Reviewer 1#3 | All 3 | 1 | `AE_NS_Results_v1/master_results.json` |
| **v2 training improvements** | **New** | All 3 | 3 | `AE_NS_Results_v2/v2_*_results.json` |
| **AE-NS Z + Fairlearn post-proc** | **New** | All 3 | 3 | `AE_NS_Results_v2/fairpost_3seeds_*.json` |
| **v2 sensitivity (α sweep)** | **New** | All 3 | 1 | `AE_NS_Results_v2/*_sensitivity_alpha.json` |
| **v2 sensitivity (β sweep)** | **New** | All 3 | 1 | `AE_NS_Results_v2/*_sensitivity_beta.json` |
| **v2 ablation** | **New** | All 3 | 1 | `AE_NS_Results_v2/*_ablation.json` |

Total wall-clock time: ~4 hours on a single NVIDIA RTX 4070 Laptop GPU.

## Summary of Manuscript Changes

| Section | Change | Reviewers addressed |
|---------|--------|---------------------|
| Title | Re-positioned as multi-task learning with semantic constraints | R1#1 |
| Abstract | Rewritten; mentions NMI, 8 baselines, 7 ablations | R1#1, R1#2, R1#5 |
| §1 Introduction | Expanded to ~1.5 pages with **5 contributions** and organisation paragraph | R1#1, R2#1 |
| §2.3 Neuro-symbolic | Added reframing paragraph | R1#1 |
| §2.4 Positioning | **New** summary table comparing 5 prior methods | R2#2 |
| §3.2 Architecture | Expanded "what happens after Z?" | R2#3 |
| §3.4 Detach(v) | New paragraph justifying the operation | R2#4 |
| §3.5 Theorem 2 | Stated in Fenchel-Lagrange language | R3#2 |
| §3.6 Fenchel-Lagrange | **New** section on duality and differentiable programming | R3#2 |
| §3.7 v2 training | **New** label smoothing, gradual warm-up, cosine annealing | New |
| §3.8 Fairpost pipeline | **New** AE-NS Z + Fairlearn post-processing pipeline | New |
| §4.1 Datasets | Expanded with detailed table | R2#6 |
| §4.3 Baselines | Expanded from 2 to 8 | R1#2, R2#9 |
| §4.4 Implementation | **New** hyper-parameter table | R1#3, R2#8 |
| §4.5 Metrics | **New** list of 8 metrics including NMI | R2#7, R1#5 |
| §5 Results | Restructured into 5 sub-sections | R2#10 |
| §5.1 Baselines | New 8-baseline table with mean ± std | R1#2, R2#8, R2#9 |
| §5.2 NMI | **New** NMI table and classify-on-Z table | R1#5, R2#3 |
| §5.3 Ablation | Expanded from 1 to 7 variants on all 3 datasets | R2#11 |
| §5.4 Sensitivity | **New** 3×3 grid + warm-up ablation | R1#4, R2#5 |
| §5.5 Computational cost | **New** training/inference/parameter table | R1#3 |
| §5.6 Fairlearn post-proc | **New** AE-NS Z + Fairlearn pipeline results | New |
| §5.7 Pareto frontier | **New** plot for all 3 datasets | R1#2 |
| §6.4 Limitations | **New** subsection with 5 limitations | R1#5 |
| §6.5 Blockchain / Fed. | **New** subsection on alternative mechanisms | R3#1 |
| §6.6 L2O perspective | **New** subsection on differentiable programming | R3#2 |
| §6.7 Certification | **New** subsection on statistical/algorithmic/theoretical certification | R3#3 |
| §7 Conclusion | Expanded with 5 future-work directions | R2#12 |
| References | +12 new refs [28]-[39] | R2#12, R3#1, R3#2, R3#3 |
| Appendix C | **New** full reproducibility section | R1#3 |

We thank the reviewers again for their careful and constructive feedback. The revised manuscript is, we believe, a substantially improved version of the original.

Sincerely,
[Author list]
