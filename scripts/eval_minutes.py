import time, numpy as np, pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from fpl_smart import history, minutes as MN
t=time.time()
m = history.load_matches()
ms = MN.add_state(m); print('state', round(time.time()-t,1),'s', ms.shape)
fr = MN.training_frame(ms); print('frame', fr.shape, round(time.time()-t,1),'s')
bf = MN.backtest_features(ms)
order = ['2022-23','2023-24','2024-25','2025-26','2026-27']
preds = []
for test in ['2024-25','2025-26','2026-27']:
    start = ms[ms.season==test].kickoff.min()
    tr = fr[fr.t_kick < start]          # targets strictly before the test season starts
    mdl = MN.MinutesModel.train(tr)
    mdl.save(f'pre_{test}')
    idx = ms.index[ms.season==test]
    p = mdl.predict(bf.loc[idx]); p['season']=test
    preds.append(p); print(test, 'train rows', len(tr), round(time.time()-t,1),'s')
P = pd.concat(preds); ev = ms.loc[P.index].join(P[['p_play','p60','emin']], rsuffix='_p')
ev['y_play']=(ev.minutes>0).astype(int); ev['y60']=(ev.minutes>=60).astype(int)
bfe = bf.loc[ev.index]
for s, g in ev.groupby('season'):
    b = bfe.loc[g.index]
    naive_play = b.pl_e5.fillna(0.3).clip(0.02,0.98); naive60 = b.p60_e5.fillna(0.2).clip(0.02,0.98)
    print(s, 'play logloss %.4f (naive ewm %.4f)  p60 logloss %.4f (naive %.4f)  mins MAE %.2f (naive %.2f) RMSE %.2f' % (
        log_loss(g.y_play, g.p_play.clip(1e-4,1-1e-4)), log_loss(g.y_play, naive_play), log_loss(g.y60, g.p60_p.clip(1e-4,1-1e-4)), log_loss(g.y60, naive60),
        (g.minutes-g.emin).abs().mean(), (g.minutes-b.min_e5.fillna(20)).abs().mean(), np.sqrt(((g.minutes-g.emin)**2).mean())))
print('calibration p60:'); print(ev.groupby(pd.cut(ev.p60_p,[0,.02,.1,.25,.5,.75,.9,.97,1], include_lowest=True)).agg(n=('y60','size'), pred=('p60_p','mean'), act=('y60','mean')).round(3))
imp = pd.Series(mdl.models['p60'].feature_importance('gain'), index=MN.FEATURES).sort_values(ascending=False); print((imp/imp.sum()).round(3).head(15).to_dict())
P.to_parquet('data/processed/minutes_wf.parquet')
