from fpl_smart.scoring import score_gw

POS = {1: 1, 2: 1, **{i: 2 for i in range(3, 8)}, **{i: 3 for i in range(8, 13)}, 13: 4, 14: 4, 15: 4}
LINEUP = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]        # 3-4-3
BENCH = [2, 12, 6, 7]                                  # keeper, then a midfielder and two defenders
PTS = {p: 2 for p in POS}
MINS = {p: 90 for p in POS}


def test_captain_doubles():
    assert score_gw(LINEUP, BENCH, 13, 14, {**PTS, 13: 10}, MINS, POS).points == 20 + 10 + 10


def test_vice_takes_over_when_captain_does_not_play():
    r = score_gw(LINEUP, BENCH, 13, 14, {**PTS, 13: 0, 14: 6}, {**MINS, 13: 0}, POS)
    assert r.captain == 14 and (13, 12) in r.subs
    assert r.points == 9 * 2 + 6 + 2 + 6          # nine others, vice, the substitute, vice again


def test_triple_captain():
    assert score_gw(LINEUP, BENCH, 13, 14, {**PTS, 13: 10}, MINS, POS, chip="3xc").points == 20 + 10 + 20


def test_substitute_must_keep_three_defenders():
    r = score_gw(LINEUP, BENCH, 13, 14, PTS, {**MINS, 3: 0}, POS)
    assert r.subs == [(3, 6)]                     # skips the midfielder: only three defenders started


def test_keeper_only_replaced_by_keeper():
    r = score_gw(LINEUP, BENCH, 13, 14, PTS, {**MINS, 1: 0}, POS)
    assert r.subs == [(1, 2)]
    r = score_gw(LINEUP, BENCH, 13, 14, {**PTS, 1: 0, 2: 0}, {**MINS, 1: 0, 2: 0}, POS)
    assert r.subs == [] and r.points == 10 * 2 + 2    # ten outfielders and the armband, no keeper points


def test_bench_that_did_not_play_cannot_come_on():
    r = score_gw(LINEUP, BENCH, 13, 14, PTS, {**MINS, 8: 0, 12: 0}, POS)
    assert r.subs == [(8, 6)]


def test_bench_boost_counts_everyone():
    assert score_gw(LINEUP, BENCH, 13, 14, PTS, MINS, POS, chip="bboost").points == 15 * 2 + 2


def test_missing_players_score_nothing():
    r = score_gw(LINEUP, BENCH, 13, 14, {8: 5}, {8: 90}, POS)
    assert r.points == 5 and r.captain is None
