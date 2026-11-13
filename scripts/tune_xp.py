import numpy as np, pandas as pd, warnings, sys, itertools; warnings.filterwarnings('ignore')
import importlib.util
spec = importlib.util.spec_from_file_location('ev','scripts/eval_xp.py'); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
PM = ev.PM
def score(p):
    res=[]
    for test in ['2024-25','2025-26','2026-27']:
        te, E, cal = ev.run(test, p)
        y=te.pts.to_numpy(); x=E.xp.to_numpy(); rel=(te.p_play>0.5).to_numpy()
        d = te.assign(xp=x)
        top = d.groupby('gw').apply(lambda g: g.groupby('code').agg(xp=('xp','sum'),pts=('pts','sum')).nlargest(30,'xp').pts.mean()).mean()
        res += [np.sqrt(np.mean((y[rel]-x[rel])**2)), np.corrcoef(y[rel],x[rel])[0,1], top]
    return res
base = PM.PlayerParams()
print('base', np.round(score(base),4))
grid = {'hl_days':[150,220,450,700], 'k_goal':[3,5,12,20], 'k_ast':[3,5,12,20], 'xg_blend':[0.5,0.9,1.0], 'xa_blend':[0.3,0.8,1.0], 'k_dc':[2,8], 'k_bres':[6,30], 'k_yc':[6,30], 'k_sv':[6,40]}
for k, vals in grid.items():
    for v in vals:
        p = PM.PlayerParams(**{k:v}); print(f'{k}={v}', np.round(score(p),4))
