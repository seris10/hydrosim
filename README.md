# HydroSim

HydroSim models rainfall, infiltration, surface runoff and soil-water storage on a regular grid. It uses NumPy and includes a small experiment that plots rainfall, surface water and the water budget.

## Quick start

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,plots]'
hydrosim --out example-output
```

The example rains on a 24 × 24 synthetic terrain for five minutes, then runs another five minutes without rain. It writes `results.json`, `rainfall-terrain.png` and `water-budget.png`. Choose a new or empty output directory. `--size 64 --duration 3600` runs a larger, longer experiment.

## Python API

```python
import numpy as np
from hydrosim import HydroSim, HydroConfig, RainField
z = np.zeros((8, 8))
sim = HydroSim(z, dx=2.0, rain=RainField(z.shape, 2.0, mode="uniform"),
               config=HydroConfig(open_edges=(), enable_et=False))
sim.step(5.0, rain_rate_mm_h=30.0)
print(sim.balance())
```

Elevation, depth and cell spacing are in meters. Time is in seconds and rainfall is in millimeters per hour. `balance()` reports cubic meters, apart from its dimensionless relative error. `storages()` separates surface, soil and groundwater volumes.

The exported `HydroSim` class checks inputs before calling the model. Soil parameters and example rain are supplied assumptions. This is useful for numerical experiments and educational visualizations; it has not been calibrated as a flood forecast.

## Tests

```sh
python -m pytest tests -q
python validation/validate_hydrosim.py
```

Tests cover water accounting, analytic reference cases, unit conversion, invalid inputs and preservation of the original numerical core. The source comes from [Yellowstone](https://github.com/seris10/yellowstone). `UPSTREAM.json` records file hashes and `ORIGIN.md` describes the extraction. Apache-2.0.
