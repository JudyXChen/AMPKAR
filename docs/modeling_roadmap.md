# AMPKAR Modeling Roadmap

This document tracks the full scope of modeling work, from new model variants through inference, prediction, and experimental design.

## 1. Model 5 — MA Beta Nonessential [DONE]

**Status:** Complete.

Created `MA_nonessential_phos_diffrax.py` and `MA_nonessential_phos.json`. Differs from Model 3 in how AMP/ADP modulate enzyme activity:

- **Model 3 (alpha):** AMP/ADP binding modifies dissociation rates — alphaLKB1 < 1 reduces kOff (LKB1 stays bound longer), alphaPP > 1 increases kOff (PP dissociates faster)
- **Model 5 (beta):** AMP/ADP binding modifies catalytic rates — betaLKB1 > 1 increases LKB1 phosphorylation, betaCaMKK > 1 increases CaMKK phosphorylation, betaPP < 1 decreases PP dephosphorylation

25 parameters (vs 24 in Model 3): added betaCaMKK, replaced alphaLKB1→betaLKB1, alphaPP→betaPP. Same 33 states.

**Files created:**
- `src/ampk_models/models/MA_nonessential_phos_diffrax.py`
- `src/ampk_models/models/MA_nonessential_phos.json`
- `src/ampk_models/param_est/inference_run_MA_nonessential_phos.sh`

---

## 2. Model 6 — MM Beta Nonessential [DONE]

**Status:** Complete.

Created the Michaelis-Menten counterpart of Model 5. Based on Model 4 (`MM_nonessential`) but with beta parameterization on Vmax terms instead of alpha on Km terms:

- **Model 4 (alpha MM):** alphaLKB1 < 1 reduces KmLKB1 (higher affinity), alphaPP > 1 increases KmPP (lower affinity)
- **Model 6 (beta MM):** betaLKB1 > 1 increases VmaxLKB1 (faster catalysis), betaCaMKK > 1 increases VmaxCaMKK, betaPP < 1 decreases VmaxPP (slower catalysis)

24 parameters (vs 23 in Model 4): added betaCaMKK, replaced alphaLKB1→betaLKB1, alphaPP→betaPP. Same 14 states.

**Files created:**
- `src/ampk_models/models/MM_nonessential_phos_diffrax.py`
- `src/ampk_models/models/MM_nonessential_phos.json`
- `src/ampk_models/param_est/inference_run_MM_nonessential_phos.sh`

---

## 3. Multi-AMP Binding Stoichiometry [DONE]

**Status:** Complete. All 4 models updated.

AMPK binds up to 3 AMP molecules (3 Bateman domains). Updated all models with:

- **Forward rate:** `kOnAMP * AMP * X` → `kOnAMP * AMP**3 * X` (cubic binding)
- **Reverse rate:** Unchanged (first-order dissociation of the complex)
- **Stoichiometry:** d_AMP contributions from AMP-binding fluxes multiplied by 3 (3 AMP consumed/released per binding event)
- **ADP binding:** Unchanged (single-site, as literature suggests only 1 regulatory ADP site)
- **Exponent:** Fixed at 3 (not configurable)

**Changes per model:**

| Model | Fluxes changed | d_AMP change |
|-------|---------------|-------------|
| MA alpha (`MA_nonessential_diffrax.py`) | J1, J4, J10, J21, J29, J40 | `-J → -3*J` for all 6 |
| MA beta (`MA_nonessential_phos_diffrax.py`) | same 6 fluxes | same |
| MM alpha (`MM_nonessential_diffrax.py`) | J1, J4 | `-J1-J4 → -3*J1-3*J4` |
| MM beta (`MM_nonessential_phos_diffrax.py`) | J1, J4 | same |

**Note:** kOnAMP priors may need adjustment during prior elicitation since the cubic [AMP]³ term changes the effective binding rate magnitude.

---

## 4. Prior Elicitation for Models 5 & 6 [DONE]

**Status:** Complete for both Models 5 and 6.

`prior_elicit_ampk.ipynb` restructured into sections:
1. Introduction and imports
2. Direct prior elicitation (maxent for all models)
3. Alpha models (3.1 MA nonessential, 3.2 MM nonessential)
4. Beta models (4.1 MA nonessential phos, 4.2 MM nonessential phos)

Model 5 (MA beta) priors:
- betaLKB1: LogNormal(mu=3, sigma=2.5), truncated lower=1.0
- betaCaMKK: LogNormal(mu=3, sigma=2.5), truncated lower=1.0
- betaPP: LogNormal(mu=-2, sigma=1), truncated upper=1.0
- Shared kinetic params use same priors as Model 3

Model 6 (MM beta) priors:
- betaLKB1: LogNormal(mu=2, sigma=1), truncated lower=1.0
- betaCaMKK: LogNormal(mu=2, sigma=1), truncated lower=1.0
- betaPP: LogNormal(mu=-1, sigma=1), truncated upper=1.0
- Shared kinetic params use same priors as Model 4

---

## 5. Inference & Model Selection (Models 3–6)

**Status:** Shell scripts ready for all four models (3–6).

Run Pathfinder inference for all four model variants using `inference_lkb1kd.py` with joint WT + LKB1 KD data:

| Model | Kinetics | Modulation mechanism | Shell script | Status |
|-------|----------|---------------------|--------|--------|
| 3 — MA alpha | Mass-action | alpha on Kd (dissociation rates) | `inference_run_MA_nonessential.sh` | Ready |
| 4 — MM alpha | Michaelis-Menten | alpha on Km | `inference_run_MM_nonessential.sh` | Ready |
| 5 — MA beta | Mass-action | beta on kcat (catalytic rates) | `inference_run_MA_nonessential_phos.sh` | Ready |
| 6 — MM beta | Michaelis-Menten | beta on Vmax | `inference_run_MM_nonessential_phos.sh` | Ready |

**Comparison:** Use ELPD (LOO cross-validation) via `model_selection_kinase_KO.py` (adapted for 2-condition case). This answers: *Does AMP/ADP enhance AMPK signaling by modifying binding affinity or catalytic rate? And is mass-action or MM the better kinetic framework?*

---

## 6. Compare Single-AMP vs [AMP]^3 Binding Posteriors

**Status:** Not started.

The cubic AMP binding stoichiometry ([AMP]^3) appears to cause bimodality in posteriors across most models (kOff,AMP and betaAMP especially). This suggests the [AMP]^3 nonlinearity creates compensatory parameter regimes that the data cannot distinguish.

**What's needed:**
1. Re-run all 4 models with single AMP binding and compare posterior identifiability (unimodal vs bimodal) and ELPD
2. Consider a Hill-type binding approach: `kOnAMP * AMP^n * X` with `n` as a fitted parameter (possibly non-integer), which would let the data determine the effective cooperativity
3. Literature review of how other AMPK models handle multi-site AMP binding (see `docs/amp_binding_literature.md`)

---

## 7. Stimulus Strength & Frequency Analysis

**Status:** Partially explored. Current implementation uses `kGly` reduction as a proxy for metabolic stress.

### Current state
- **Stimulus strength** (`prediction/stimulus_strength.ipynb`): Varies stressed `kGly` from 0.4 → 0.005 (100-fold reduction = full ionomycin). Each value produces a different pAMPKAR trajectory. This generates dose-response curves.
- **Frequency** (`prediction/frequency.ipynb`): Has draft code for `pulse_input()` and `square_input()` time-dependent `kGly` profiles, but these functions are **not implemented**. Currently only step inputs work.

### What's needed
- Implement time-dependent `kGly(t)` in the ODE module (pulse, square wave, arbitrary waveform)
- Clarify biological meaning: stimulus strength = ionomycin dose → glycolysis inhibition magnitude; frequency = repeated transient Ca²⁺ pulses
- After calcium-CaMKK2 module (item 8) is added, stimulus strength/frequency should also modulate CaMKK2 activation via the RCamp-driven pathway

---

## 8. Calcium-CaMKK2 Module

**Status:** Design complete. See `docs/calcium_camkk2_model.md`.

Add a calcium-dependent CaMKK2 activation block using RCamp data as measured input:

$$[\text{CaMKK2}^*](t) = [\text{CaMKK2}]_{\text{tot}} \left[ f_{\text{basal}} + (1 - f_{\text{basal}}) \cdot \frac{\text{Ca}(t)^n}{K_d^n + \text{Ca}(t)^n} \right]$$

Key points:
- CaMKK2 has ~60–70% autonomous (Ca²⁺-independent) activity
- Activation is via conformational relief of autoinhibition by Ca₄CaM, NOT phosphorylation
- Adds 2–3 new parameters: $f_{\text{basal}}$, $K_d$, $n$
- Requires `diffrax.LinearInterpolation` for RCamp signal in ODE

**Implementation steps:**
1. Interpolate RCamp as `Ca(t)` via diffrax
2. Modify ODE modules to accept time-dependent `CaMKK2_active(t)`
3. Update `solve_traj()` to pass interpolant (basal phase: `Ca = 0`, stressed phase: RCamp data)
4. Add new priors and include in inference scripts

---

## 9. Kinase KO Inference Strategy

**Status:** Not started. Relevant scripts exist for 3-condition case (Linden data) but need adaptation for HeLa 2-condition case.

For joint inference across WT, LKB1 KD, and (future) CaMKK2 KO conditions, compare three parameter-sharing strategies:

| Strategy | Description | Degrees of freedom | Script |
|----------|-------------|-------------------|--------|
| **Fully shared** | One parameter set for all conditions; KO implemented by zeroing specific params | Low | `inference_lkb1kd.py` |
| **Partially shared** | Most params shared; specific params (e.g., CaMKK2-related) allowed to differ between conditions | Medium | New script needed |
| **Fully independent** | Separate parameter set per condition | High | `inference_kinase_KO_independent_pymc.py` |

Compare via ELPD. The partially shared approach is most interesting for testing whether LKB1 KO also affects CaMKK2 activity (a specific biological hypothesis).

**Note:** Currently we only have WT + LKB1 KD data for HeLa. CaMKK2 KO data exists only in Linden's dataset (`Schmitt_et_al_2022_data/`). If we want to predict CaMKK2 KO behavior in HeLa, we need to either (a) run the experiment, or (b) use the fitted model to simulate it.

---

## 10. Basal Level Offset

**Status:** Conceptual. Requires additional data.

The current model output is `pAMPKAR / AMPKARtot`, which is normalized to start at 0 (the basal steady-state has some pAMPK, but normalization removes it). To capture absolute basal AMPK activity:

$$\text{Output}(t) = a \cdot \frac{[\text{pAMPKAR}](t)}{[\text{AMPKAR}]_{\text{tot}}} + b$$

where $a$ is a scaling factor and $b$ is the basal offset. This affine transform would allow the model to predict absolute activity levels, not just fold-change from baseline.

**Requires:** Experimental data that reports absolute (non-normalized) AMPKAR signal, or independent measurements of basal AMPK phosphorylation levels.

---

## 11. Experimental Design & Scientific Questions

**Status:** Not started.

Create a structured mapping from biological questions to models, experiments, and simulation protocols:

- What questions are we trying to answer with each model variant?
- Which experiments are needed to distinguish between models?
- What predictions can we make and how do we test them?
- How do stimulus strength, frequency, and calcium-CaMKK2 predictions inform experimental design?

This should be a living document that connects the modeling work to testable hypotheses.

---

## Additional Considerations

### 11. Sensitivity analysis for multi-AMP binding

After implementing [AMP]³ binding (item 3), run local and global sensitivity analyses to check:
- Which parameters become more/less influential with cubic binding?
- Does the cubic term create identifiability issues (e.g., kOnAMP and [AMP] becoming confounded)?
- How does the steady-state AMP-AMPK fraction change compared to single-AMP binding?

### 12. Model validation with 2-DG data

All current inference uses ionomycin data. The 2-DG dataset (`HeLaAMPKAR3_RCamp_2DG.npz`, `HeLaAMPKAR3_2DG.npz` if available) provides an independent validation set — 2-DG is a glycolysis inhibitor (purely metabolic, minimal Ca²⁺ effect), so the calcium-CaMKK2 module should predict near-baseline CaMKK2 activation for 2-DG. This is a strong test of the model.

### 13. Convergence diagnostics across models

Different model structures may have different posterior geometry. Track and compare:
- Pathfinder convergence (ELBO traces, path quality)
- Rhat and ESS for posterior samples
- Prior-posterior contraction (are all parameters informed by data?)
