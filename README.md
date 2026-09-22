# Enhancing Coastal Flood Hazard Modeling with Synthetic Event Design

## Overview

This repository provides an end-to-end workflow for generating physically consistent synthetic compound coastal flood events. The framework combines **Extreme Value Theory (EVT)**, **nonstationary dependence modeling**, **vine copulas**, and **data-driven event selection** to generate trivariate flood scenarios for coastal flood hazard assessment.

The workflow is designed for three flood drivers:

- **Storm Surge (NTR)**
- **Rainfall (RF)**
- **River Discharge (RD)**

and supports two complementary approaches:

- **Design-based approach** – representative events selected from return-period isolines.
- **Response-based approach** – representative events selected using weighted clustering for efficient hydrodynamic simulations.

---

## Repository Structure

```
├── input/                  # Input datasets
├── outputs/                # Generated intermediate and final outputs
├── pipelines/              # Python and R processing functions
├── Python_Workflow.ipynb   # Python workflow
├── R_Workflow.R            # Statistical modeling workflow
├── environment.yml         # Conda environment
└── README.md
```

---

## Workflow

### Step 1 – Peak-Over-Threshold (POT) Event Extraction (Python)

For each center variable (NTR, RF, or RD), a Peak-Over-Threshold (POT) analysis is performed.

For every identified peak:

- the peak value of the center variable is extracted,
- the corresponding maximum values of the remaining two variables are identified within the event window,

producing trivariate extreme-event datasets for each center variable.

Outputs include:

- POT event datasets
- Threshold values

---

### Step 2 – Marginal Distribution Modeling (R)

The extracted POT events are used to fit marginal distributions.

- The center variable is modeled using a **nonstationary Generalized Pareto Distribution (GPD)**.
- The associated variables are fitted using their best-fitting probability distributions.
- Each variable is transformed into the uniform probability space.

---

### Step 3 – Nonstationary Vine Copula Modeling (R)

The transformed variables are used to construct a trivariate **nonstationary vine copula**, preserving the dependence structure among:

- Storm Surge
- Rainfall
- River Discharge

Synthetic multivariate events are then generated in probability space.

---

### Step 4 – Back Transformation (R)

The simulated copula samples are transformed back into physical units using the fitted marginal distributions.

Outputs include synthetic datasets of:

- Storm Surge
- Rainfall
- River Discharge

with realistic joint dependence.

---

### Step 5 – Event Selection (Python)

Two alternative approaches are provided.

#### Design-Based Approach

Large synthetic datasets are used to derive **bivariate return-period isolines**.

Representative design events are selected from these isolines for specified return periods.

---

#### Response-Based Approach

Large synthetic datasets are reduced using **weighted coreset K-means clustering**.

Each representative event is assigned a statistical weight that is later used to estimate flood frequencies from hydrodynamic simulation results.

---

### Step 6 – Time-Series Reconstruction (Python)

Each synthetic event is reconstructed into complete time series by:

- identifying historical analog events,
- scaling the magnitude,
- aligning the temporal evolution,

while preserving realistic event dynamics.

Separate workflows reconstruct:

- Storm Surge time series
- Rainfall time series
- River Discharge time series

---

### Step 7 – Spatial Rainfall Generation (Python)

Synthetic rainfall time series are converted into spatially distributed rainfall fields using historical NLDAS rainfall data.

For each event:

- the nearest NLDAS grid cell to the rainfall gauge is identified,
- the historical spatial rainfall field is scaled according to the synthetic rainfall magnitude,
- spatial rainfall fields are exported as NetCDF files for hydrodynamic modeling.

---

### Step 8 – Water Level Reconstruction (Python)

To generate realistic coastal boundary conditions, the reconstructed synthetic storm surge time series are combined with the corresponding historical tidal signal and mean sea level.

For each synthetic event:

- the reconstructed storm surge time series is aligned with the historical event,
- A random astronomical tide and random lag time are added,
- A random mean sea level is incorporated,
- the final total water level time series is generated.

The resulting water level time series preserve realistic tidal variability while matching the synthetic storm surge magnitude, providing physically consistent coastal boundary conditions for hydrodynamic flood simulations.

---

## Software

The workflow combines Python and R.

### Python

Used for:

- POT event extraction
- Event reconstruction
- Time-series scaling and alignment
- Spatial rainfall generation
- Representative-event selection

### R

Used for:

- Marginal distribution fitting
- Nonstationary GPD modeling
- Vine copula fitting
- Synthetic event simulation

---

## Outputs

The workflow generates:

- POT event datasets
- Threshold estimates
- Synthetic multivariate events
- Design-based events from bivariate isolines
- Representative weighted response-based events
- Reconstructed boundary-condition time series
- Spatial rainfall NetCDF files

These outputs can be directly used as boundary conditions for hydrodynamic flood models such as **SFINCS**.

---

## Citation

If you use this repository in your research, please cite the associated publication (to be added).
