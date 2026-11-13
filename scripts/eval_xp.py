"""Walk-forward accuracy of expected points vs FPL's own projection and simple baselines."""
import os, sys, time, numpy as np, pandas as pd
from fpl_smart import history, market, minutes as MN, player_model as PM, team_model as TM
t=time.time()
m = history.load_matches()
if not os.path.exists('data/processed/minutes_wf.parquet'):
    sys.exit('run scripts/eval_minutes.py first: it writes the walk-forward minutes predictions used here')
tm = market.attach_market(history.team_matches(m))
twf = TM.walk_forward(tm, [s for s in sorted(tm.season.unique()) if s >= '2023-24'])
mwf = pd.read_parquet('data/processed/minutes_wf.parquet')
ms0 = MN.add_state(m)
ms0 = ms0.join(mwf[['p_play','p60','emin']].rename(columns={'p60':'p60_hat'}))
ms0 = PM.prepare(ms0, twf)
bf = MN.backtest_features(ms0)

def run(test, p=PM.PlayerParams(), verbose=True, mode='prior'):
    start = ms0[ms0.season==test].kickoff.min()
    train = (ms0.kickoff < start).to_numpy()
    if test=='2025-26':   # first season with defensive-contribution data: calibrate that part in-season
        train_cal = train | (ms0.season==test).to_numpy()
    else: train_cal = train
    cal, ms = PM.fit_calibration(ms0, train_cal if test=='2025-26' else train, p)
    if test=='2025-26':
        cal2, _ = PM.fit_calibration(ms0, train, p); cal2.dc = cal.dc; cal2.priors['dc']=cal.priors['dc']; cal = cal2
        ms = ms0.copy(); ms['bres'] = (ms['bonus'] - PM.bonus_fitted(ms, cal.bonus_coef)) * (ms['minutes']>0) * ms['has_xg']
    S = PM.state_backtest(ms, p.hl_days)
    te = ms[ms.season==test]
    R = PM.rates(S.loc[te.index], te.pos, te.value, cal, p)
    key = te.team_code.astype(str)+'_'+te.fixture.astype(str)
    E = PM.expected_points_team(R, key, te.pos, te.lam, te.lam_a, te.p_play, te.p60_hat, te.emin, cal, defcon=test>='2025-26', mode=mode)
    return te, E, cal

def report(te, E, label):
    y = te.pts.to_numpy(); x = E.xp.to_numpy()
    b = bf.loc[te.index]
    rel = (te.p_play>0.5).to_numpy()
    out = {'n':len(te), 'rmse_all':np.sqrt(np.mean((y-x)**2)), 'mae_all':np.mean(np.abs(y-x)), 'rmse_rel':np.sqrt(np.mean((y[rel]-x[rel])**2)), 'corr_rel':np.corrcoef(y[rel],x[rel])[0,1], 'bias_rel': np.mean(x[rel]-y[rel])}
    f = te.fpl_xp.notna().to_numpy()
    if f.sum()>1000:
        fx = te.fpl_xp.to_numpy()
        out.update({'FPL_rmse_all':np.sqrt(np.mean((y[f]-fx[f])**2)), 'my_rmse_same':np.sqrt(np.mean((y[f]-x[f])**2)), 'FPL_rmse_rel':np.sqrt(np.mean((y[f&rel]-fx[f&rel])**2)), 'my_rmse_rel_same':np.sqrt(np.mean((y[f&rel]-x[f&rel])**2)), 'FPL_corr_rel':np.corrcoef(y[f&rel],fx[f&rel])[0,1], 'my_corr_rel_same':np.corrcoef(y[f&rel],x[f&rel])[0,1]})
        # decision metric: actual points of the 30 highest-projected players each GW
        d = te[f].assign(xp=x[f]); 
        top_my = d.groupby('gw').apply(lambda g: g.groupby('code').agg(xp=('xp','sum'),pts=('pts','sum')).nlargest(30,'xp').pts.mean()).mean()
        top_fpl = d.groupby('gw').apply(lambda g: g.groupby('code').agg(xp=('fpl_xp','sum'),pts=('pts','sum')).nlargest(30,'xp').pts.mean()).mean()
        out.update({'top30_my':top_my, 'top30_fpl':top_fpl})
    print(label, {k:(round(v,4) if isinstance(v,float) else v) for k,v in out.items()})
    return out

if __name__=='__main__':
    for test in ['2024-25','2025-26','2026-27']:
        te, E, cal = run(test)
        report(te, E, test)
        pl = te.minutes>0
        comp = pd.DataFrame({'pred':[E.x_goals.sum(), E.x_assists.sum(), E.p_cs[te.pos<=2].sum(), E.bonus.sum(), E.saves.sum(), E.defcon.sum()/2, E.gc.sum(), E.cards.sum(), E.app.sum(), E.xp.sum()],
          'act':[te.goals.sum(), te.assists.sum(), te.cs60[te.pos<=2].sum(), te.bonus.sum(), (te.saves//3).sum(), te.dc_hit.sum(), -(te.gc//2)[te.pos<=2].sum(), -(te.yc+3*te.rc+2*te.og+2*te.pm).sum(), ((te.minutes>0)*1+(te.minutes>=60)*1).sum(), te.pts.sum()]}, index=['goals','assists','cs(def)','bonus','savepts','dc_hits','gc','cards','app','TOTAL'])
        comp['ratio']=comp.pred/comp.act; print(comp.round(3).T.to_string())
        print('  cal: cs', np.round(cal.cs,3), 'dc', {k:np.round(v,2) for k,v in cal.dc.items()}, 'mult', {k:{a:round(b,2) for a,b in v.items()} for k,v in cal.mult.items()})
    print(round(time.time()-t,1),'s')
