# ORCO

ORCO calculates a ranked posterior probability distribution over the 20 canonical amino-acid residue types from partial sets of atom-specific protein NMR chemical shifts. It then converts that distribution into the entropy-adaptive Goldilocks candidate set defined in the preprint.

This README documents `orco.py`, the single-file Python implementation. Complete model definitions, reference statistics, and validation are provided in the [ORCO preprint](https://doi.org/10.2139/ssrn.7182159). The browser implementation is supplemental.

## Requirements

- Python 3 (verified with Python 3.12.13)
- No third-party packages; only the standard-library `math` module is used
- No external data file; the reference statistics are embedded in `orco.py`

## Quick start

Edit the `SPIN_SYSTEMS` list near the beginning of `orco.py`. Each row represents one spin system:

```python
SPIN_SYSTEMS = [
    ["sample_1", 4.23, 54.45, 33.27, 170.54, 0.00, None],
]
```

The columns are:

```text
[name, HA, CA, CB, CO, N, HN]
```

Chemical shifts are given in ppm. Use `None` or `0.00` for an unobserved atomic site. `name` is used only to label the output and is not used in the calculation. Add any number of rows, then run:

```bash
python orco.py
```

## Output

For each spin system, `orco.py` reports the observed shifts and the Goldilocks candidate set. `SHOW_20_CLASS_POSTERIOR` and `SHOW_63_CLASS_POSTERIOR` control whether the corresponding diagnostic probability tables are also printed. These settings do not change the calculation.

Definitions of the reported distributions and the Goldilocks set are provided in the [ORCO preprint](https://doi.org/10.2139/ssrn.7182159).

The calculation can also be called directly:

```python
from orco import classify

result = classify(["sample_1", 4.23, 54.45, 33.27, 170.54, 0.00, None])
```

The default `SPIN_SYSTEMS` block contains two GB1 examples and two limiting-case examples. They can be replaced with the user's own spin systems.

## Repository contents

- `orco.py`: Python reference implementation.
- `index.html`: supplemental browser interface and GitHub Pages site.

## Citation

Owen Raymond and Bo Chen. *ORCO: An Entropy Based Adaptive Residue-Type Classifier for Partial Protein NMR Chemical Shift Observations* (2026). [SSRN preprint 7182159](https://doi.org/10.2139/ssrn.7182159).
