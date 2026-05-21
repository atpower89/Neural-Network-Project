# Neural Network Limitations in Data-Limited Materials Discovery

Controlled experiments demonstrating the failure modes of feedforward neural networks when applied outside their training distribution. Accompanies a written report examining extrapolation, data coverage, and overfitting in the context of materials property prediction.

## Repository Structure

```
├── main.py               # Run experiments and save results
├── configs.py            # Experiment configurations
├── experiments.py        # Experiment logic
├── models.py             # Neural network definition and training
├── data.py               # Dataset generation
├── split.py              # Train/test splitting strategies
├── results.py            # Results storage and loading
├── visualization.py      # Plotting utilities
├── figures/              # Standalone figure scripts
│   ├── activation_extrapolation.py
│   ├── interp_vs_extrap.py
│   ├── overfitting.py
│   └── polynomial_gap.py
├── results/              # Saved experiment outputs (JSON)
└── NN_Project_Paper.pdf  # Accompanying report
```

## Dependencies

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- NumPy
- Matplotlib

Install with:
```bash
pip install torch numpy matplotlib
```

## Usage

### Running Experiments

Edit the experiment selection at the bottom of `main.py`, then run from the project root:

```bash
python main.py
```

Results are saved to `results/` as timestamped JSON files. Available experiments are defined in `configs.py`.

### Regenerating Figures

Each figure script in `figures/` loads the most recent matching result from `results/`. Run from the project root:

```bash
python figures/activation_extrapolation.py
python figures/interp_vs_extrap.py
python figures/polynomial_gap.py
python figures/overfitting.py
```

Pre-saved results are included in the repository, so figures can be regenerated without re-running experiments.

## Experiments

| Experiment | Description |
|---|---|
| `linear_extrapolation_range` | Three activation functions (ReLU, tanh, sigmoid) trained on a linear function; evaluated beyond the training boundary |
| `sine_extrapolation` | Network trained on $\sin(x)$ over $[0, 2\pi]$; evaluated over $[0, 4\pi]$ |
| `polynomial_gap` | Network trained on $x^2$ with a gap in $[-1, 1]$; evaluated across the full range |
| `sine_overfit_clean` / `sine_overfit_noisy` | Side-by-side comparison of clean and noisy sinusoidal training |
