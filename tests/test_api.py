"""Standalone input rejection, parity, and water-budget tests."""
import numpy as np
import pytest
from hydrosim import HydroSim, HydroConfig, RainField, make_terrain
from hydrosim.core import HydroSim as Original


def make(cls=HydroSim, **kw):
    z=make_terrain(6,6,2,seed=4)
    return cls(z,2,rain=RainField(z.shape,2,mode='uniform',seed=4),**kw)


@pytest.mark.parametrize('edges,rate',[((),0),((),40),(('S',),40)])
def test_budget_and_unchanged_trajectory(edges,rate):
    args=dict(config=HydroConfig(open_edges=edges,enable_et=False))
    a,b=make(**args),make(Original,**args)
    for _ in range(50): a.step(2,rate);b.step(2,rate)
    for name in ['h','S','F','qx','qy']:
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert a.balance()==b.balance()
    assert abs(a.balance()['error'])<1e-10


@pytest.mark.parametrize('name,value',[('dx',0),('dx',True),('dx',np.nan),('init_wetness',2),('init_depth',-1),('kc',np.inf),('soil_names',['unlisted']),('soil_map',np.zeros((6,6),bool)),('channel_mask',np.zeros((2,2)))])
def test_bad_construction(name,value):
    kw={name:value}; dx=kw.pop('dx',2)
    with pytest.raises(ValueError): HydroSim(np.zeros((6,6)),dx,**kw)


@pytest.mark.parametrize('dt,rain',[(0,1),(-1,1),(np.nan,1),(1,-1),(1,np.inf),(True,1),(1,True)])
def test_bad_step_preserves_state(dt,rain):
    sim=make();before=sim.h.copy()
    with pytest.raises(ValueError): sim.step(dt,rain)
    np.testing.assert_array_equal(before,sim.h)
    assert sim.t==0


@pytest.mark.parametrize('config',[HydroConfig(infiltration='bad'),HydroConfig(root_depth_m=0),HydroConfig(max_substeps=0),HydroConfig(open_edges=('X',)),HydroConfig(outflow_slope=-1)])
def test_bad_configuration(config):
    with pytest.raises(ValueError): make(config=config)


def test_mm_per_hour_to_cubic_meters():
    sim=HydroSim(np.zeros((3,3)),2,config=HydroConfig(infiltration='none',open_edges=(),enable_et=False,perc_factor=0),rain=RainField((3,3),2,mode='uniform'))
    sim.step(60,6)
    assert sim.balance()['rain']==pytest.approx(36*0.006/60)
    assert sim.h.mean()==pytest.approx(0.006/60)
