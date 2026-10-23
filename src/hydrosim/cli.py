"""Synthetic rainfall and water-budget demonstration."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from . import HydroSim, HydroConfig, RainField, make_terrain


def demo(out, size=24, duration=600, dt=5):
    """Save synthetic rainfall, runoff, storage, and closure results."""
    if not isinstance(size, int) or not 2 <= size <= 256 or not 1 <= duration <= 86400 or not 0 < dt <= duration:
        raise ValueError("size requires 2..256 and duration requires 1..86400 seconds")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    terrain = make_terrain(size, size, 2.0, relief_m=.25, seed=7)
    sim = HydroSim(terrain, 2, config=HydroConfig(enable_et=False), rain=RainField(terrain.shape, 2, mode="uniform", seed=7))
    rows=[]
    start=time.perf_counter()
    while sim.t < duration:
        rate=45.0 if sim.t < duration/2 else 0.0
        sim.step(min(dt, duration-sim.t),rate)
        rows.append(dict(time_s=sim.t,rainfall_mm_h=rate,**{k+'_m3':v for k,v in sim.storages().items()},**{k+'_m3':v for k,v in sim.balance().items() if k!='rel_error'}))
    elapsed=time.perf_counter()-start
    result=dict(schema="hydrosim.synthetic_demo.v1",fixture_class="synthetic",seed=7,grid=[size,size],dx_m=2,duration_s=duration,step_s=dt,wall_time_s=elapsed,python=platform.python_version(),platform=platform.platform(),boundary="open N/E/S/W",infiltration="green_ampt",et_enabled=False,balance=sim.balance(),samples=rows,limitations=["Authored rainfall and terrain; no field calibration or forecast validation.","Elapsed time measures this fixture on this machine, not game frame rate."])
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    t=np.array([x['time_s'] for x in rows])/60
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot(t,[x['rainfall_mm_h'] for x in rows]); axes[0].set(xlabel='Time (min)',ylabel='Rainfall (mm/h)',title='Synthetic uniform rainfall')
    im=axes[1].imshow(terrain,extent=(0,size*2,0,size*2),origin='lower'); axes[1].set(xlabel='East (m)',ylabel='North (m)',title='Synthetic terrain, seed 7');fig.colorbar(im,ax=axes[1],label='Elevation (m)')
    fig.suptitle('Synthetic fixture | source core SHA256 '+hashlib.sha256(Path(__file__).with_name('core.py').read_bytes()).hexdigest()[:12], fontsize=10)
    fig.savefig(out/'rainfall-terrain.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for key,label in [('rain_m3','Rain input'),('outflow_m3','Runoff export'),('d_storage_m3','Storage change')]:
        axes[0].plot(t,[x[key] for x in rows],label=label)
    axes[0].legend();axes[0].set(xlabel='Time (min)',ylabel='Cumulative volume (m3)',title='Water budget')
    axes[1].plot(t,[x['error_m3'] for x in rows]);axes[1].set(xlabel='Time (min)',ylabel='Closure residual (m3)',title='Rain - export - storage change')
    im=axes[2].imshow(sim.h*1000,extent=(0,size*2,0,size*2),origin='lower');axes[2].set(xlabel='East (m)',ylabel='North (m)',title='Final surface water');fig.colorbar(im,ax=axes[2],label='Depth (mm)')
    fig.suptitle(f'Synthetic rainfall/runoff | {size}x{size}, {duration:g} s | source '+hashlib.sha256(Path(__file__).with_name('core.py').read_bytes()).hexdigest()[:12], fontsize=10)
    fig.savefig(out/'water-budget.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
    return result


def main():
    """Run a small synthetic experiment with explicit units."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--size',type=int,default=24)
    parser.add_argument('--duration',type=float,default=600)
    args=parser.parse_args()
    try: demo(args.out,args.size,args.duration)
    except (ValueError, OSError) as exc: parser.error(str(exc))
