"""Input checks around the unchanged Yellowstone numerical implementation."""
from dataclasses import asdict
import numbers
import numpy as np
from .core import HydroSim as OriginalHydroSim, HydroConfig, Climate, SOILS


def number(value, name, minimum=0, positive=False):
    """Require a finite real value in the supported range."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f"{name} requires a finite real number")
    if value < minimum or (positive and value <= minimum):
        raise ValueError(f"{name} is outside the supported range")


def configuration(cfg, climate):
    """Validate supported simulation and climate settings."""
    if not isinstance(cfg, HydroConfig) or not isinstance(climate, Climate):
        raise ValueError("config and climate require HydroConfig and Climate")
    if cfg.infiltration not in {"green_ampt", "horton", "constant", "none"}:
        raise ValueError("unknown infiltration method")
    if not isinstance(cfg.open_edges, (tuple, list)) or any(x not in {"N", "E", "S", "W"} for x in cfg.open_edges) or len(set(cfg.open_edges)) != len(cfg.open_edges):
        raise ValueError("open_edges requires unique N/E/S/W values")
    if cfg.outflow_slope != "auto":
        number(cfg.outflow_slope, "outflow_slope")
    for name, value in asdict(cfg).items():
        if name in {"infiltration", "open_edges", "outflow_slope", "channel_acc_threshold", "enable_et"}:
            continue
        number(value, name, positive=name in {"root_depth_m", "manning_overland", "manning_channel", "recovery_time_s", "cfl_alpha", "max_substeps", "gw_recession_time_s", "kinematic_viscosity_m2_s"})
    if not isinstance(cfg.enable_et, bool) or not isinstance(climate.diurnal, bool):
        raise ValueError("ET and diurnal switches require booleans")
    if not isinstance(cfg.max_substeps, int) or cfg.cfl_alpha > 1 or not 0 < cfg.depletion_p < 1:
        raise ValueError("invalid timestep or depletion settings")
    if cfg.channel_acc_threshold is not None:
        number(cfg.channel_acc_threshold, "channel_acc_threshold", positive=True)
        if not isinstance(cfg.channel_acc_threshold, int):
            raise ValueError("channel_acc_threshold requires an integer")
    for name, value in asdict(climate).items():
        if name != "diurnal":
            number(value, name, minimum=-10000)
    if not -100 <= climate.temp_c <= 100 or not 0 <= climate.rel_humidity <= 100 or climate.wind_u2 < 0 or climate.rn_mj_m2_day < 0 or climate.elevation_m > 10000 or not 0 <= climate.start_hour < 24:
        raise ValueError("climate is outside the supported demonstration range")


class HydroSim(OriginalHydroSim):
    """Validate constructor and step inputs before using the original solver."""
    def __init__(self, z, dx, soil_map=None, soil_names=None, config=None,
                 climate=None, rain=None, kc=0.8, init_wetness=0.5,
                 init_depth=None, channel_mask=None):
        terrain = np.asarray(z)
        if terrain.dtype.kind not in "fiu" or terrain.ndim != 2 or min(terrain.shape) < 2 or not np.isfinite(terrain).all():
            raise ValueError("z requires a finite numeric grid of at least 2 by 2")
        number(dx, "dx", positive=True)
        number(init_wetness, "init_wetness")
        if init_wetness > 1:
            raise ValueError("init_wetness requires [0,1]")
        cfg = HydroConfig() if config is None else config
        met = Climate() if climate is None else climate
        configuration(cfg, met)
        names = list(soil_names) if soil_names is not None else ["loam"]
        if not names or any(x not in SOILS for x in names):
            raise ValueError("soil_names contains an unknown soil")
        if soil_map is not None:
            smap = np.asarray(soil_map)
            if smap.shape != terrain.shape or smap.dtype.kind not in "iu" or smap.min() < 0 or smap.max() >= len(names):
                raise ValueError("soil_map requires in-range integer indices matching z")
        for name, value in [("kc", kc), ("init_depth", init_depth)]:
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"{name} requires finite nonnegative numbers")
            try:
                np.broadcast_to(array, terrain.shape)
            except ValueError as exc:
                raise ValueError(f"{name} must broadcast to z") from exc
        if channel_mask is not None and np.asarray(channel_mask).shape != terrain.shape:
            raise ValueError("channel_mask must match z")
        if rain is not None and rain.shape != terrain.shape:
            raise ValueError("rain must match z")
        super().__init__(z, dx, soil_map, names, cfg, met, rain, kc, init_wetness, init_depth, channel_mask)

    def step(self, dt, rain_rate_mm_h=0.0):
        """Advance positive seconds using a finite nonnegative rainfall rate in mm/h."""
        number(dt, "dt", positive=True)
        number(rain_rate_mm_h, "rain_rate_mm_h")
        configuration(self.cfg, self.climate)
        return super().step(dt, rain_rate_mm_h)
