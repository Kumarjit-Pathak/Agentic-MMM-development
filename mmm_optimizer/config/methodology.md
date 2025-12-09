# MMM Methodology Specification

This document defines the **PROTECTED METHODOLOGY** for Marketing Mix Modeling (MMM) that cannot be changed during hyperparameter optimization.

## Overview

The methodology consists of fundamental modeling choices that define the scientific framework of the MMM. These choices are based on peer-reviewed research and industry best practices, and changing them would invalidate model comparability and business insights.

---

## Protected Methodology Components

### 1. Adstock Type: `beta_gamma`

**Description:** Beta-Gamma decay function for modeling advertising carryover effects.

**Rationale:**
- Models realistic decay patterns with flexible shape parameters
- Captures both immediate and delayed advertising effects
- Validated by Montgomery et al. (1989), Broadbent (1997)
- Allows channel-specific decay rates while maintaining functional form

**Mathematical Form:**
```
impact_t = x_t + Σ(beta × gamma^lag × x_(t-lag))
```

**Why Protected:**
- Alternative adstock types (geometric, Weibull, delayed) have different carryover assumptions
- Changing adstock type mid-optimization invalidates historical comparisons
- Beta-Gamma provides sufficient flexibility through decay parameter tuning

**Tunable Parameters:**
- `tv_adstock_decay`: 0.2 - 0.9 (longer carryover)
- `digital_adstock_decay`: 0.2 - 0.7 (shorter carryover)
- `print_adstock_decay`: 0.1 - 0.6 (medium carryover)
- `ooh_adstock_decay`: 0.2 - 0.8 (medium-long carryover)

---

### 2. Saturation Type: `hill`

**Description:** Hill saturation function for modeling diminishing returns to advertising spend.

**Rationale:**
- S-shaped curve that ensures bounded, realistic response
- Models both threshold effects (initial ramp-up) and saturation (plateau)
- Standard in pharmacology and marketing science literature
- Guarantees mathematically valid ROI calculations

**Mathematical Form:**
```
response = (spend^alpha) / (K^alpha + spend^alpha)
```

Where:
- `alpha`: Steepness of saturation curve (1.5 - 3.5)
- `K`: Half-saturation point (derived from data)

**Why Protected:**
- Alternative saturation types (exponential, linear, logistic) have fundamentally different behavior
- Exponential functions don't saturate (unbounded growth - unrealistic)
- Linear functions don't capture diminishing returns
- Hill function is the industry standard for MMM

**Tunable Parameters:**
- `tv_saturation_alpha`: 1.5 - 3.5
- `digital_saturation_alpha`: 1.5 - 3.5
- `print_saturation_alpha`: 1.5 - 3.5
- `ooh_saturation_alpha`: 1.5 - 3.5

---

### 3. Hierarchy Levels: `["brand", "region", "channel"]`

**Description:** Three-level hierarchical structure for aggregating marketing effects.

**Rationale:**
- Mirrors actual organizational structure (brand → regional teams → channel managers)
- Enables granular insights while maintaining statistical power
- Allows different saturation/adstock parameters per hierarchy level
- Supports both top-down planning and bottom-up execution

**Hierarchy Structure:**
```
Brand (National/Portfolio)
└── Region (Geographic Markets)
    └── Channel (TV, Digital, Print, OOH, etc.)
```

**Why Protected:**
- Adding/removing hierarchy levels changes model complexity and data requirements
- Different hierarchies require different prior distributions
- Changing levels mid-optimization breaks continuity of insights
- Business stakeholders expect consistent reporting structure

**Examples:**
- Brand: "Brand_A", "Brand_B"
- Region: "Northeast", "Southeast", "West", "Central"
- Channel: "TV", "Digital_Display", "Digital_Search", "Print", "OOH"

---

### 4. Architecture Type: `hierarchical_neural_additive`

**Description:** Hierarchical Neural Additive Model (H-NAM) architecture combining interpretability with flexibility.

**Rationale:**
- Neural Additive Models (NAMs) provide feature-wise interpretability
- Hierarchical structure respects organizational levels
- Additive form ensures interpretable contribution analysis
- Balances flexibility (neural networks) with explainability (additive structure)

**Architecture Components:**
1. **Input Layer:** Raw features (spend, impressions, control variables)
2. **Transformation Layer:** Adstock and saturation transformations
3. **Hierarchical Layers:** Brand → Region → Channel aggregation
4. **Additive Output:** Sum of channel contributions + base sales

**Why Protected:**
- Alternative architectures (pure neural nets, linear models, GAMs) have different interpretability trade-offs
- Pure neural nets lose channel-level interpretability
- Linear models can't capture non-linear saturation
- Architecture change requires retraining from scratch

**Key Properties:**
- **Interpretability:** Each channel's contribution is traceable
- **Flexibility:** Non-linear transformations via neural sub-networks
- **Additivity:** Total sales = Σ(channel effects) + base
- **Hierarchical:** Respects organizational structure

---

### 5. Loss Function: `mse`

**Description:** Mean Squared Error (MSE) loss for model training.

**Rationale:**
- Standard regression loss for continuous outcomes (sales)
- Penalizes large errors more than small errors
- Mathematically tractable and well-behaved
- Directly related to R² and MAE metrics

**Mathematical Form:**
```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

**Why Protected:**
- Alternative losses (MAE, Huber, quantile) have different robustness properties
- MSE is sensitive to outliers, which is appropriate for sales data (spikes matter)
- Changing loss function mid-optimization disrupts convergence
- Business stakeholders understand squared error interpretation

**Related Metrics:**
- **MAPE:** Mean Absolute Percentage Error (interpretability)
- **R²:** Coefficient of determination (goodness of fit)
- **MAE:** Mean Absolute Error (robustness check)

---

## Summary: What CAN Be Changed

While the methodology is protected, the following **hyperparameters** are fully tunable:

### Adstock Parameters (4 parameters)
- `tv_adstock_decay`: 0.2 - 0.9
- `digital_adstock_decay`: 0.2 - 0.7
- `print_adstock_decay`: 0.1 - 0.6
- `ooh_adstock_decay`: 0.2 - 0.8

### Saturation Parameters (4 parameters)
- `tv_saturation_alpha`: 1.5 - 3.5
- `digital_saturation_alpha`: 1.5 - 3.5
- `print_saturation_alpha`: 1.5 - 3.5
- `ooh_saturation_alpha`: 1.5 - 3.5

### Neural Network Training (4 parameters)
- `learning_rate`: 0.0001 - 0.01
- `regularization`: 0.001 - 0.1
- `batch_size`: 16, 32, 64, 128
- `epochs`: 50 - 1000

**Total Tunable Hyperparameters:** 12

---

## References

1. Montgomery, D. B., & Silk, A. J. (1989). "Clusters of consumer interests and opinion leaders' spheres of influence." *Journal of Marketing Research*
2. Broadbent, S. (1997). "Accountable Advertising." *NTC Publications*
3. Agarwal, R., et al. (2021). "Neural Additive Models: Interpretable Machine Learning with Neural Nets." *NeurIPS*
4. Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects." *Google Research*

---

## Version History

- **v1.0 (Dec 2025):** Initial methodology specification
  - Beta-Gamma adstock
  - Hill saturation
  - 3-level hierarchy (brand/region/channel)
  - Hierarchical Neural Additive architecture
  - MSE loss function

---

## Usage

This methodology specification is used by:
1. **Data Scientist Agent:** LLM system prompt to enforce restrictions
2. **Business Analyst Agent:** Context for validating model outputs
3. **Stakeholder Agent:** Understanding of what can/cannot be changed in approval decisions
4. **Documentation:** Reference for stakeholders and future model iterations

**To modify methodology:** Update this file and regenerate agent prompts. Note that methodology changes require full model retraining and stakeholder approval.
