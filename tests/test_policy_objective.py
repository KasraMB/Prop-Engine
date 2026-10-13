from propfirm_engine import BracketHistory,DollarPolicy,Engine
from propfirm_engine import fitting
from test_chronological_backtest import config,session_day,spec,trade


def test_direct_scoring_preserves_selection_and_avoids_candidate_replays(monkeypatch):
    history=BracketHistory(tuple(trade(session_day(i),target=10) for i in range(10)))
    policy=DollarPolicy.constant(200)
    common=dict(policy=policy,risk_bounds={r.name:(100,300) for r in policy.regimes},
                generations=2,population=4,seed=17,
                objective=lambda r:r.policy.regimes[0].risk_dollars)
    original=fitting.backtest;calls=[]
    def counted(*args,**kwargs):
        calls.append(args[1]);return original(*args,**kwargs)
    monkeypatch.setattr(fitting,'backtest',counted)
    legacy=Engine().fit(spec(),history,config(),**common)
    old_count=len(calls);calls.clear();scored=[]
    def direct(p,h):
        assert max(h.sessions)<history.sessions[7]
        scored.append(p);return p.regimes[0].risk_dollars
    fast=Engine().fit(spec(),history,config(),policy_objective=direct,**common)
    assert fast==legacy
    assert len(scored)==fast.evaluations
    assert len(calls)<=3 and old_count>=fast.evaluations+2
