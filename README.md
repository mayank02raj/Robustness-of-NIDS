# Categorical Robustness Assessment for ML-based Network Intrusion Detection Systems

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-see%20below-lightgrey.svg)](#license-and-release-terms)
[![Paper](https://img.shields.io/badge/paper-under%20review%20%40%20IEEE%20Access-b31b1b.svg)](#paper)
[![Dataset](https://img.shields.io/badge/dataset-ACI--IoT--2023-purple.svg)](#dataset)

A reproducible, six-dimensional adversarial-robustness evaluation framework for ML-based Network Intrusion Detection Systems. Trains three fundamentally different classifier architectures — CNN, LSTM, and Random Forest — on the ACI-IoT-2023 dataset (1.23M samples, 12 attack classes), then systematically stress-tests them with FGSM, PGD, CLEVER certification, transfer attacks, perturbability analysis, distribution-shift evaluation, and diffusion-based reconstruction. The results expose a pattern we formalize as the **False Champion Problem**: the model that wins the clean-data benchmark is the one that collapses first when an adversary pushes back.

Supports the paper:

> **Raj, M.**, Bastian, N. D., Fiondella, L., Kul, G. *Categorical Robustness Assessment for Machine Learning based Network Intrusion Detection Systems.* Under review at IEEE Access.

## The False Champion Problem

Most ML-based NIDS papers report >99% accuracy on clean test data and stop there. This work asks the next question — what happens under adversarial perturbation? — and the answer inverts conventional wisdom about which model to deploy.

**Headline result on ACI-IoT-2023 under FGSM at ε = 0.01 (1% of normalized feature range):**

|Model            |Clean accuracy|Under FGSM ε = 0.01|Accuracy drop|
|-----------------|--------------|-------------------|-------------|
|**Random Forest**|**99.98%**    |**26.8%**          |**−73.2 pp** |
|LSTM             |99.36%        |85.0%              |−14.4 pp     |
|**CNN**          |99.06%        |**95.5%**          |**−3.6 pp**  |

The Random Forest wins the clean-data benchmark by a fraction of a percentage point. It is also the model that loses 88 F1 points to a single-step attack with the smallest perturbation budget we tested. The CNN — which would have lost a pure-accuracy bake-off — is the deployment-robust choice by a **68.66 percentage-point robustness advantage**.

**Under stronger PGD attack at ε = 0.1 (40 iterations):** all three models converge near 28–30% accuracy, but Random Forest’s F1 collapses to 13.9% while CNN retains 28.6% and LSTM retains 25.0% — meaning Random Forest isn’t just wrong more often, it’s making systematically worse predictions.

The practical implication: every SOC running an ML-based NIDS is making an implicit bet that benchmark numbers translate to production. This repository provides evidence that the bet is often losing, a framework for measuring exactly how, and architecture-specific deployment guidance for adversarial environments.

## Why this result happens

Random Forests partition feature space with **axis-aligned, threshold-based splits**. An imperceptible nudge to a single feature can push a sample across a threshold, flipping how every tree in the ensemble routes it. Discontinuous decision boundaries create exploitable cliffs.

CNNs and LSTMs learn **smooth, continuous decision surfaces** through gradient-based optimization. Crossing from one class region to another requires traversing a gradient field. The same perturbation budget that obliterates Random Forest barely moves a CNN’s prediction.

The paper shows this architecturally, not just empirically — the *mechanism* that makes tree ensembles fast, interpretable, and high-accuracy on clean data is the same mechanism that makes them catastrophically brittle under attack.

## What’s in this repository

The numbered scripts build sequentially into the complete evaluation framework.

|Script                    |Purpose                                                                                                                                                                                                                                                                                                                                                                                |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`1_Base.py`               |Baseline training — CNN (6-layer FC, 256 units, BatchNorm + Dropout 0.3), LSTM (2-layer, 256 units, recurrent dropout 0.2, + 3 FC layers), Random Forest (GridSearchCV over 12 configurations). Stratified 80/20 split on 1.23M samples, 40% minority-class oversampling, class-balanced loss weighting. Reports aggregate, class-wise, and category-wise accuracy/precision/recall/F1.|
|`2_Plus_Class_wise.py`    |Per-class evaluation for all 12 attack classes. Exposes class-imbalance effects hidden by aggregate accuracy — ARP Spoofing (only 6 samples) and UDP Flood (142 test samples) are where the models actually fail.                                                                                                                                                                      |
|`3_Plus_category_wise.py` |Per-category evaluation — groups the 12 classes into 5 categories (Benign, Reconnaissance, DoS, Brute Force, Spoofing) and evaluates category-level detection. Aligned with the categorical robustness framework proposed in the paper.                                                                                                                                                |
|`4_plus_FGSM.py`          |Fast Gradient Sign Method attack at ε ∈ {0.01, 0.05, 0.1}. White-box for CNN/LSTM, transfer-based gray-box for Random Forest (adversarial examples crafted against CNN surrogate).                                                                                                                                                                                                     |
|`5_plus_PGD.py`           |Projected Gradient Descent — 40 iterations, step size α = ε/4, L∞ projection at every step. The stronger iterative attack, considered the gold standard for empirical robustness testing.                                                                                                                                                                                              |
|`6_plus_Clever_Perturb.py`|CLEVER score computation (L2 and L∞) plus perturbability analysis — attack-agnostic robustness estimates via extreme value theory, used for cross-architecture comparison independent of specific attacks.                                                                                                                                                                             |
|`Figures/`                |Result plots from the paper — baseline bar charts, class-wise F1 heatmaps, degradation heatmaps, PGD curves, category-wise radar charts.                                                                                                                                                                                                                                               |

## Six-dimensional evaluation framework

The paper goes beyond “accuracy under FGSM” and integrates six complementary robustness dimensions:

```
  ┌──────────────────────────────────────────────────────────┐
  │              ACI-IoT-2023 (1.23M, 12 classes)            │
  │    Benign · Recon · DoS · Brute Force · Spoofing         │
  └─────────────────────────┬────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │    CNN    │     │   LSTM    │     │    RF     │
    │ 6-FC+BN+  │     │ 2-LSTM+   │     │  100–200  │
    │  Dropout  │     │  3-FC     │     │   trees   │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │            Six-Dimensional Robustness Evaluation          │
  ├──────────────────────────────────────────────────────────┤
  │  1. Baseline metrics — aggregate + class + category       │
  │  2. Adversarial attacks — FGSM / PGD × ε ∈ {0.01,.05,.1}  │
  │  3. Transfer attacks — CNN surrogate → RF (gray-box)      │
  │  4. Certified robustness — CLEVER L2 / L∞ scores         │
  │  5. Distribution shift — covariate (Gaussian) + label    │
  │  6. Generalization — diffusion-based reconstruction      │
  └──────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────┐
  │  False Champion Problem · Reconnaissance Universal        │
  │  Vulnerability · Scenario-Specific Deployment Guidance   │
  └──────────────────────────────────────────────────────────┘
```

## Three research questions, three findings

The paper is organized around three research questions. Each has a clean answer backed by the data.

**RQ1 — How does adversarial robustness differ across fundamentally distinct ML architectures?**
Dramatically. Random Forest drops 73 percentage points at ε = 0.01 while CNN retains 96% of baseline. This is not a dataset artifact — it is a direct consequence of axis-aligned vs. smooth decision boundaries. Architecture matters more than hyperparameters, dataset, or training tricks.

**RQ2 — Is high baseline accuracy a reliable indicator of adversarial robustness?**
No — it is actively misleading. The model with the highest baseline (99.98% Random Forest) is the model with the worst adversarial behavior. Benchmarks that report only clean accuracy are structurally incapable of identifying this failure mode.

**RQ3 — How does vulnerability vary across attack categories and individual classes?**
Non-uniformly, and in ways that matter operationally. **Reconnaissance attacks** (Ping Sweep, OS Scan, Vulnerability Scan) are universally vulnerable across all architectures — CLEVER scores consistently below 1.5 versus above 8.0 for Brute Force classes. Minority classes (ARP Spoofing with only 6 samples, UDP Flood with 142 test samples) become completely undetectable under perturbation. Class-balanced training alone cannot fix this when the underlying data is scarce.

## Deployment guidance from the paper (Table 6)

This is the section recruiters and hiring managers care most about — concrete guidance, not just analysis:

|Scenario                          |CNN|LSTM|RF |Recommendation               |
|----------------------------------|:-:|:--:|:-:|-----------------------------|
|Early-stage detection (ε ≤ 0.01)  |✓  |✓   |✗  |**CNN**                      |
|Heavy adversarial attack (ε = 0.1)|✓  |✓   |✗  |CNN or LSTM                  |
|Benign traffic classification     |✓  |✓   |✗  |**CNN**                      |
|Reconnaissance detection          |△  |△   |✗  |Requires adversarial training|
|DoS detection                     |✓  |✓   |✗  |CNN or LSTM                  |
|Distribution-shift tolerance      |✓  |✓   |✗  |**CNN**                      |
|Production NIDS (overall)         |✓  |✓   |✗  |**CNN**                      |

**Bottom line:** deploy CNN-based architectures for production NIDS in any environment where adversaries are assumed. Do not deploy Random Forest for production intrusion detection in adversarial conditions, regardless of its benchmark accuracy.

## Quick start

```bash
git clone https://github.com/mayank02raj/Robustness-of-NIDS.git
cd Robustness-of-NIDS

# Environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Dataset — ACI-IoT-2023 from Canadian Institute for Cybersecurity, UNB
# https://www.unb.ca/cic/datasets/
# Place CSVs under ./data/ACI-IoT-2023/

# Run the full pipeline (sequential)
python 1_Base.py                   # baseline training + eval
python 2_Plus_Class_wise.py        # per-class metrics
python 3_Plus_category_wise.py     # per-category metrics
python 4_plus_FGSM.py              # FGSM attacks at 3 ε values
python 5_plus_PGD.py               # PGD attacks at 3 ε values
python 6_plus_Clever_Perturb.py    # CLEVER + perturbability
```

Each script writes results to `Figures/` and prints summary tables to stdout. The full baseline-plus-attack evaluation takes roughly 4–6 hours on the hardware listed below.

## Reproducibility

|Setting                    |Value                                                                      |
|---------------------------|---------------------------------------------------------------------------|
|Python                     |3.10+                                                                      |
|PyTorch                    |2.0                                                                        |
|scikit-learn               |1.3                                                                        |
|NumPy / Pandas / Matplotlib|1.24 / 2.0 / 3.7                                                           |
|Hardware (paper results)   |MacBook Pro M4 Max, 128 GB unified memory                                  |
|Random seed                |42 (NumPy + PyTorch), deterministic CUDA where possible                    |
|Dataset                    |ACI-IoT-2023, 1,231,411 samples, 12 classes, 5 categories                  |
|Train/test split           |Stratified 80/20 → 985,128 train · 246,283 test                            |
|Class imbalance mitigation |40% minority-class oversampling + class-balanced loss weighting            |
|CNN                        |6 FC layers × 256 units, BatchNorm, Dropout 0.3, Adam lr=5e-4              |
|LSTM                       |2 LSTM layers × 256 units, recurrent dropout 0.2, 3 FC layers, Adam lr=1e-3|
|Random Forest              |GridSearchCV: 100–200 estimators, depth {None, 20, 30}, min_split {2, 5}   |
|Attacks                    |FGSM + PGD at ε ∈ {0.01, 0.05, 0.1}; PGD 40 iter, α = ε/4                  |

All intermediate outputs — confusion matrices, classification reports, adversarial accuracies, CLEVER scores, perturbability metrics — are saved to disk for independent verification.

## Complete results table

Condensed from Table 4 of the paper:

|Attack               |ε   |CNN Acc         |CNN F1         |LSTM Acc|LSTM F1|RF Acc   |RF F1    |
|---------------------|---:|---------------:|--------------:|-------:|------:|--------:|--------:|
|**Baseline**         |—   |0.991           |0.992          |0.994   |0.994  |**1.000**|**1.000**|
|FGSM                 |0.01|**0.955**       |**0.960**      |0.850   |0.872  |0.268    |0.116    |
|FGSM                 |0.05|0.561           |0.593          |0.583   |0.569  |0.266    |0.112    |
|FGSM                 |0.10|0.495           |0.487          |0.513   |0.481  |0.266    |0.112    |
|PGD                  |0.01|**0.854**       |**0.887**      |0.686   |0.731  |0.269    |0.118    |
|PGD                  |0.05|0.453           |0.475          |0.490   |0.483  |0.266    |0.112    |
|PGD                  |0.10|0.287           |0.286          |0.296   |0.250  |0.280    |0.139    |
|Δ baseline → PGD 0.10|    |**−70.4 pp acc**|**−70.6 pp F1**|−69.8   |−74.4  |**−72.0**|**−86.1**|

## Why this work matters

Most adversarial-ML research targets computer vision, where perturbations are continuous pixel grids and every ε-ball maps to a visually imperceptible image. NIDS is fundamentally different in ways that break direct transfer:

1. **Features are tabular and mixed-type** — numeric, categorical, binary, with semantic constraints. You cannot “nudge” a TCP flag by 0.03.
1. **Labels have operational consequences.** A flipped label isn’t a lower accuracy number — it’s a real attacker past a real defense.
1. **Classes are heavily imbalanced in the real world.** Aggregate accuracy hides per-class failure. A model that detects DoS floods while missing reconnaissance can score >99% accuracy and remain useless against APTs.

The framework in this repository was designed for these constraints. That is why robustness is reported per-class and per-category alongside aggregate numbers, not instead of them.

## Skills demonstrated

Adversarial machine learning, robustness certification (CLEVER), PyTorch model training, tabular deep learning, feature engineering for network traffic, ML experimental design, reproducibility practices, scientific Python tooling, categorical robustness frameworks, distribution-shift analysis, transfer-attack methodology, publication-grade result reporting.

## Skills mapped to job postings

- **“Adversarial robustness evaluation”** — FGSM, PGD, and CLEVER implemented end-to-end on a production-scale NIDS dataset
- **“ML for cybersecurity / NIDS”** — 1.23M packet ACI-IoT-2023 with 12 attack classes across 5 categories
- **“Model evaluation and benchmarking”** — categorical robustness framework beyond aggregate accuracy
- **“Deep learning for tabular data”** — 6-layer CNN and 2-layer LSTM both evaluated on network flow features
- **“Research communication”** — IEEE Access submission backing the code
- **“Reproducible ML”** — fixed seeds, pinned dependencies, complete hyperparameter tables, documented hardware
- **“Collaboration with government / defense research”** — DoD Cooperative Agreement, collaboration with U.S. Military Academy at West Point, DHS-funded work via the Homeland Security Community of Best Practices

## Related work in my portfolio

This repository is part of a coherent research program. If this work interests you, see also:

- [`SOC-home-lab`](https://github.com/mayank02raj/SOC-home-lab) — detection infrastructure: 11-service Dockerized SOC with Sigma rules, threat hunting, and ATT&CK-mapped adversary emulation
- [`ATTACK-Coverage-Dashboard`](https://github.com/mayank02raj/ATTACK-Coverage-Dashboard) — MITRE ATT&CK detection-coverage analytics with weighted scoring across 130+ threat actors
- [`Phishing-URL-Detector`](https://github.com/mayank02raj/Phishing-URL-Detector) — production-shaped ML service with SHAP explainability and PSI drift monitoring; the deployment counterpart to this research

Across the four repos, the story is end-to-end: measure where ML detection fails (this repo), know where coverage is honest (`ATTACK-Coverage-Dashboard`), run the whole stack in a realistic environment (`SOC-home-lab`), and ship ML detection to production (`Phishing-URL-Detector`).

## Paper

**Categorical Robustness Assessment for Machine Learning based Network Intrusion Detection Systems.**
Raj, M., Bastian, N. D., Fiondella, L., Kul, G.
*Under review at IEEE Access.*

Preprint available on request — please reach out via the contact details below. If you use this work, please cite:

```bibtex
@article{raj2026categorical,
  title   = {Categorical Robustness Assessment for Machine Learning based
             Network Intrusion Detection Systems},
  author  = {Raj, Mayank and Bastian, Nathaniel D. and Fiondella, Lance
             and Kul, Gokhan},
  journal = {IEEE Access (under review)},
  year    = {2026},
  note    = {Preprint available on request}
}
```

## Funding and disclaimer

This work was supported in part by the U.S. Military Academy (USMA) under Cooperative Agreement No. W911NF-22-2-0160, and in part by the Homeland Security Community of Best Practices (HS CoBP) through the U.S. Department of the Air Force under contract FA8075-18-D-0002/FA8075-21-F-0074.

The views and conclusions expressed in this paper are those of the authors and do not reflect the official policy or position of the U.S. Military Academy, U.S. Army, U.S. Department of Homeland Security, the U.S. Department of the Air Force, or U.S. Government.

## License and release terms

License status is **pending institutional review** prior to open-source release.

This repository contains research software produced under a U.S. Department of Defense Cooperative Agreement (W911NF-22-2-0160) and in part under a U.S. Department of Homeland Security contract through the U.S. Department of the Air Force (FA8075-18-D-0002/FA8075-21-F-0074). Data rights, release terms, and applicable open-source licensing are being confirmed with:

- The Principal Investigator (Dr. Gokhan Kul, UMass Dartmouth)
- UMass Dartmouth’s Office of Research Administration
- Co-investigator institutions (U.S. Military Academy at West Point)

Until a formal license is posted, please contact the corresponding author before using, redistributing, or building on this code. For academic evaluation, reference, and review of the paper’s results, the code is provided as-is. A formal license file (`LICENSE`) will be added to this repository once release terms are finalized.

## Contact

**Mayank Raj** — M.S. Data Science (Thesis Track), UMass Dartmouth · Graduating May 2026

- Portfolio: [mayank02raj.github.io](https://mayank02raj.github.io)
- LinkedIn: [linkedin.com/in/mayank02raj](https://linkedin.com/in/mayank02raj)
- GitHub: [github.com/mayank02raj](https://github.com/mayank02raj)
- Email: mraj1@umassd.edu

Open to full-time cybersecurity and ML-security roles in the US. F-1 STEM OPT eligible — no sponsorship required through August 2029.

## Limitations and honest caveats

The paper is transparent about the following, and so is this README:

1. **Single dataset.** ACI-IoT-2023 is modern and well-constructed, but generalization to enterprise and cloud environments with different traffic distributions is future work. The claim is about *architectural* robustness, not dataset-specific numbers.
1. **White-box attacks on neural networks.** We assume attackers have full model access for FGSM/PGD on CNN and LSTM. Real-world black-box scenarios may behave differently, though the transfer-attack results (CNN surrogate → Random Forest) suggest our examples generalize.
1. **Fixed perturbation budgets.** Adaptive adversaries who query the model iteratively (decision-based, score-based attacks) could potentially exceed our static FGSM/PGD degradation numbers.
1. **No defenses tested.** This work characterizes the *vulnerability*. Adversarial training, input preprocessing, and certified defenses (randomized smoothing, interval bound propagation) are natural follow-ons and are not evaluated here.
1. **Feature-space perturbations may not all map to realizable packet modifications.** Our ε ≤ 0.1 in normalized space corresponds to modest flow-statistic changes, but a complete real-world evasion pipeline would require additional packet-level constraints. This is acknowledged in the paper as a direction for future work.
1. **CLEVER is a statistical estimate**, not a deterministic certificate. Formal certification via randomized smoothing or interval bound propagation would strengthen the robustness claims.

## Extension ideas

- Implement adversarial training (Madry-style) on CNN and measure whether it closes the residual robustness gap under PGD at ε = 0.1
- Port the evaluation to CIC-IDS2017 and UNSW-NB15 for cross-dataset validation of the False Champion Problem
- Add feature-space constraints so perturbations respect protocol semantics (TCP flag validity, checksum consistency, packet length feasibility)
- Compare against certified defenses via ART’s randomized smoothing and interval bound propagation
- Extend to sequence-level attacks: perturb whole MITRE ATT&CK chains rather than individual packets
- Ensemble the CNN and LSTM with complementary class-specific strengths (LSTM for benign, CNN for reconnaissance) and measure whether the ensemble beats either alone
- Integrate with [`ATTACK-Coverage-Dashboard`](https://github.com/mayank02raj/ATTACK-Coverage-Dashboard) so robustness-under-perturbation becomes an explicit dimension in detection-coverage scoring

## Co-authors

- **Dr. Nathaniel D. Bastian** — Deputy Director of Robotics Research Center, U.S. Military Academy at West Point
- **Dr. Lance Fiondella** — Director of Cybersecurity Center, UMass Dartmouth (NSA/DHS-designated CAE-R)
- **Dr. Gokhan Kul** (advisor) — Associate Director of Cybersecurity Center, UMass Dartmouth
