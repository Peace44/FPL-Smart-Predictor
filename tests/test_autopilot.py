"""Acting on the account: the executor against a stand-in for FPL, the sign-in token store, and
the timing rules. No network: `FakeFPL` plays the part of the signed-in API."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from fpl_smart import account as acc
from fpl_smart import autopilot, execute, my_team
from fpl_smart.solver import MIN_PLAY, GWPlan

NOW = pd.Timestamp("2026-11-20 08:00", tz="UTC")
DEADLINE = "2026-11-21T11:00:00Z"


class FakeFPL:
    """Behaves like the signed-in FPL API as this project understands it."""

    def __init__(self, players, squad, bank=20, free=1, chips=("wildcard", "freehit", "bboost", "3xc")):
        self.cost_of = players["now_cost"].to_dict()
        self.pos = players["pos"].to_dict()
        gk = [p for p in squad if self.pos[p] == 1]
        out = [p for p in squad if self.pos[p] != 1]
        order = gk[:1] + out[:10] + gk[1:] + out[10:]
        self.picks = [{"element": p, "position": i + 1, "is_captain": i == 1, "is_vice_captain": i == 2,
                       "selling_price": self.cost_of[p], "purchase_price": self.cost_of[p]} for i, p in enumerate(order)]
        self.bank, self.limit, self.made = bank, free, 0
        self.chips = {c: "available" for c in chips}
        self.posts: list[tuple] = []
        self.ignore_chip = False          # FPL takes the transfers but does not register the chip
        self.forget_captain = False       # FPL saves the line-up but not the armband

    def my_team(self):
        active = any(self.chips.get(c) == "active" for c in ("wildcard", "freehit"))
        return {"picks": [dict(p) for p in self.picks],
                "transfers": {"bank": self.bank, "limit": None if active else self.limit, "made": self.made,
                              "cost": 4, "status": "unlimited" if active else "cost", "value": 1000},
                "chips": [{"name": c, "status_for_entry": s, "is_pending": False} for c, s in self.chips.items()]}

    @property
    def cost(self):                       # points spent on transfers, as the project works it out from FPL's fields
        return acc.points_spent(self.my_team())

    def post_transfers(self, gw, transfers, chip=None):
        self.posts.append(("transfers", chip, len(transfers)))
        if chip and not self.ignore_chip:
            self.chips[chip] = "active"
        elif not chip:                    # as FPL did in GW11: no chip named cancels one still pending
            for c in ("wildcard", "freehit"):
                if self.chips.get(c) == "active":
                    self.chips[c] = "available"
        for t in transfers:
            slot = next(p for p in self.picks if p["element"] == t["element_out"])
            assert self.pos[t["element_in"]] == self.pos[t["element_out"]]
            assert t["selling_price"] == slot["selling_price"] and t["purchase_price"] == self.cost_of[t["element_in"]]
            self.bank += t["selling_price"] - t["purchase_price"]
            slot.update(element=t["element_in"], selling_price=t["purchase_price"], purchase_price=t["purchase_price"],
                        is_captain=False, is_vice_captain=False)
        assert self.bank >= 0
        self.made += len(transfers)
        return {}

    def post_picks(self, picks, chip=None):
        self.posts.append(("picks", chip, len(picks)))
        price = {p["element"]: p for p in self.picks}
        assert sorted(p["element"] for p in picks) == sorted(price) and [p["position"] for p in picks] == list(range(1, 16))
        self.picks = [{**price[p["element"]], **p} for p in picks]
        if self.forget_captain:
            for p in self.picks:
                p["is_captain"] = False
        for c in ("bboost", "3xc"):
            if self.chips.get(c) == "active" and c != chip:
                self.chips[c] = "available"
        if chip:
            self.chips[chip] = "active"
        return {}


def bootstrap(players):
    return {"elements": [{"id": int(i), "element_type": int(r.pos), "team": int(r.team_id), "now_cost": int(r.now_cost),
                          "web_name": r["name"], "status": "a", "chance_of_playing_next_round": None}
                         for i, r in players.iterrows()],
            "events": [{"id": 1, "is_next": True, "is_current": False, "deadline_time": DEADLINE}],
            "game_settings": {"transfers_cap": 20}}


def target(players, squad, swaps=0, chip=None, hits=0):
    """A legal plan for gameweek 1: `swaps` cheapest-for-cheapest changes, then a 4-4-2."""
    pos, cost, club = players["pos"], players["now_cost"], players["team_id"]
    new = list(squad)
    for k in range(swaps):
        out = sorted((p for p in new if p in squad), key=lambda p: (pos[p] != 2 + k % 3, -cost[p]))[0]   # dearest original, rotating position
        clubs = pd.Series([club[p] for p in new if p != out]).value_counts()
        cand = [p for p in players.index if p not in new and p not in squad and pos[p] == pos[out]
                and cost[p] <= cost[out] and clubs.get(club[p], 0) < 3]
        new[new.index(out)] = max(cand, key=lambda p: cost[p])
    by = {k: [p for p in new if pos[p] == k] for k in (1, 2, 3, 4)}
    lineup = by[1][:1] + by[2][:4] + by[3][:4] + by[4][:2]
    bench = by[1][1:] + by[2][4:] + by[3][4:] + by[4][2:]
    assert all(sum(1 for p in lineup if pos[p] == k) >= MIN_PLAY[k] for k in MIN_PLAY)
    week = GWPlan(1, chip, [p for p in new if p not in squad], [p for p in squad if p not in new], hits, 1,
                  new, lineup, bench, lineup[5], lineup[6], 0, 50.0)
    return SimpleNamespace(plan=SimpleNamespace(weeks=[week]))


@pytest.fixture()
def world(market, squad_state):
    players, _ = market
    return players, list(squad_state.squad), bootstrap(players)


def run(rec, fpl, bs, **kw):
    return execute.apply(rec, fpl, bs, now=NOW, **kw)


# ----------------------------------------------------------------------------- executor
def test_transfer_then_lineup_and_readback(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=1)
    out = run(rec, fpl, bs)
    w = rec.plan.weeks[0]
    assert out.ok and out.verified and out.changed and out.points_cost == 0
    assert fpl.posts == [("transfers", None, 1), ("picks", None, 15)]
    assert {p["element"] for p in fpl.picks} == set(w.squad)
    assert execute._lineup_state(fpl.my_team()) == (frozenset(w.lineup), tuple(w.bench), w.captain, w.vice)


def test_second_run_changes_nothing(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=1)
    run(rec, fpl, bs)
    fpl.posts.clear()
    again = run(rec, fpl, bs)
    assert again.ok and not again.changed and fpl.posts == [] and "Nothing to change" in again.actions[0]


def test_dry_run_sends_nothing(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=1)
    out = run(rec, fpl, bs, dry_run=True)
    assert out.ok and not out.changed and not out.verified and fpl.posts == []
    assert [r["kind"] for r in out.requests] == ["transfers", "picks"]


def test_refuses_a_hit_the_plan_did_not_expect(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad, free=0), target(players, squad, swaps=1, hits=0)
    out = run(rec, fpl, bs)
    assert "Refused" in out.error and "4 points" in out.error and fpl.posts == []


def test_refuses_a_hit_above_the_unattended_limit(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad, free=1), target(players, squad, swaps=2, hits=1)
    assert "limit for unattended" in run(rec, fpl, bs, max_new_hits=0).error and fpl.posts == []
    assert run(rec, fpl, bs, max_new_hits=1).ok and fpl.cost == 4


def test_refuses_when_prices_moved_out_of_budget(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad, bank=0), target(players, squad, swaps=1)
    bought = rec.plan.weeks[0].transfers_in[0]
    next(e for e in bs["elements"] if e["id"] == bought)["now_cost"] += 30
    out = run(rec, fpl, bs)
    assert "over budget" in out.error and fpl.posts == []


def test_refuses_close_to_the_deadline_and_illegal_plans(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=1)
    late = execute.apply(rec, fpl, bs, now=pd.Timestamp(DEADLINE) - pd.Timedelta(minutes=5))
    assert "too late" in late.error and fpl.posts == []
    w = rec.plan.weeks[0]
    w.captain = w.bench[1]
    assert "captain" in run(rec, fpl, bs).error and fpl.posts == []


def test_wildcard_is_played_with_one_transfer_first(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=5, chip="wildcard")
    out = run(rec, fpl, bs)
    assert out.ok and out.verified and fpl.cost == 0 and fpl.chips["wildcard"] == "active"
    assert fpl.posts == [("transfers", "wildcard", 1), ("transfers", "wildcard", 4), ("picks", None, 15)]
    assert {p["element"] for p in fpl.picks} == set(rec.plan.weeks[0].squad)


def test_wildcard_that_does_not_register_costs_nothing(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=5, chip="wildcard")
    fpl.ignore_chip = True
    out = run(rec, fpl, bs)
    assert "Mismatch" in out.error and "no points spent" in out.error
    assert fpl.posts == [("transfers", "wildcard", 1)] and fpl.made == 1 and fpl.cost == 0


def test_wildcard_dropped_by_fpl_is_reported_with_the_fix(world):
    """GW11: FPL let go of the pending wildcard and charged the rest. Say so, and say how to undo it."""
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=5, chip="wildcard")
    real = fpl.post_transfers

    def drops_chip(gw, transfers, chip=None):
        res = real(gw, transfers, chip)
        if fpl.made > 1:
            fpl.chips["wildcard"] = "available"
        return res
    fpl.post_transfers = drops_chip
    out = run(rec, fpl, bs)
    assert "Mismatch" in out.error and "no longer active" in out.error and "FPL site" in out.error


def test_wildcard_needs_a_free_transfer_to_test_with(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad, free=0), target(players, squad, swaps=5, chip="wildcard")
    assert "by hand" in run(rec, fpl, bs).error and fpl.posts == []


def test_more_transfers_are_free_once_the_wildcard_is_active(world):
    players, squad, bs = world
    fpl = FakeFPL(players, squad)
    run(target(players, squad, swaps=3, chip="wildcard"), fpl, bs)
    fpl.posts.clear()
    out = run(target(players, squad, swaps=5, chip="wildcard"), fpl, bs)
    assert out.ok and fpl.cost == 0 and fpl.posts[0][:2] == ("transfers", "wildcard")   # keeps the chip named
    # ... but a plan that ignores the active chip is refused
    assert "already active" in run(target(players, squad, swaps=1), fpl, bs).error


def test_triple_captain_is_played_with_the_lineup(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, chip="3xc")
    out = run(rec, fpl, bs)
    assert out.ok and fpl.posts == [("picks", "3xc", 15)] and fpl.chips["3xc"] == "active"
    assert any("Triple Captain" in a for a in out.actions)
    fpl.chips["3xc"] = "played"
    assert "available" in run(target(players, squad, chip="bboost", swaps=0), FakeFPL(players, squad, chips=("3xc",)), bs).error


def test_a_lineup_fpl_did_not_keep_is_reported(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad)
    fpl.forget_captain = True
    out = run(rec, fpl, bs)
    assert "Mismatch" in out.error and not out.verified


def test_nothing_is_sent_when_fpl_answers_in_an_unknown_shape(world):
    players, squad, bs = world
    for breakage in (lambda mt: mt["transfers"].pop("cost"), lambda mt: mt["picks"].pop(),
                     lambda mt: mt.update(chips=None), lambda mt: mt["transfers"].update(status="frozen"),
                     lambda mt: mt["picks"][0].pop("selling_price")):
        fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=1)
        real = fpl.my_team

        def broken(real=real, breakage=breakage):
            mt = real()
            breakage(mt)
            return mt
        fpl.my_team = broken
        out = run(rec, fpl, bs)
        assert "Refused" in out.error and "not laid out as expected" in out.error and fpl.posts == []
    assert acc.shape_problem(FakeFPL(players, squad).my_team()) is None


def test_unlimited_transfers_are_enough_evidence_of_the_chip(world):
    players, squad, bs = world
    fpl, rec = FakeFPL(players, squad), target(players, squad, swaps=4, chip="wildcard")
    real = fpl.my_team

    def unlabelled():            # FPL reports unlimited transfers but words the chip status differently
        mt = real()
        for c in mt["chips"]:
            if c["status_for_entry"] == "active":
                c["status_for_entry"] = "pending"
        return mt
    fpl.my_team = unlabelled
    out = run(rec, fpl, bs)
    assert out.ok and fpl.cost == 0 and any("does not label" in w for w in out.warnings)
    assert {p["element"] for p in fpl.picks} == set(rec.plan.weeks[0].squad)


def test_first_swap_is_legal_on_its_own(world):
    players, squad, bs = world
    rec = target(players, squad, swaps=5, chip="wildcard")
    w = rec.plan.weeks[0]
    els = {e["id"]: e for e in bs["elements"]}
    sell = {p: els[p]["now_cost"] for p in squad}
    one = execute.first_swap(w.transfers_out, w.transfers_in, squad, els, sell, bank=0)
    assert els[one["element_in"]]["element_type"] == els[one["element_out"]]["element_type"]
    assert one["selling_price"] - one["purchase_price"] >= 0


# ----------------------------------------------------------------------------- sign-in tokens
class FakeHTTP:
    """A token endpoint that, like the real one, replaces the refresh token on every use."""

    def __init__(self, valid):
        self.valid, self.n, self.seen = set(valid), 0, []

    def post(self, url, data=None, **kw):
        tok = data["refresh_token"]
        self.seen.append(tok)
        if tok not in self.valid:
            return SimpleNamespace(status_code=400, text="{}", json=lambda: {"error": "invalid_grant"})
        self.valid.discard(tok)
        self.n += 1
        new = f"r{self.n}"
        self.valid.add(new)
        body = {"access_token": f"a{self.n}", "refresh_token": new, "expires_in": 600}
        return SimpleNamespace(status_code=200, text=json.dumps(body), json=lambda: body)


def test_refresh_token_is_rotated_saved_and_never_replayed(tmp_path):
    http, store = FakeHTTP({"seed"}), tmp_path / "auth" / "token.json"
    a = acc.Account(1, "seed", store, http)
    assert a.access_token() == "a1" and a.access_token() == "a1"          # cached while it is valid
    assert json.loads(store.read_text())["refresh_token"] == "r1"
    assert oct(store.stat().st_mode)[-3:] == "600"
    assert acc.Account(1, "seed", store, http).access_token(force=True) == "a2"   # a new process continues the chain
    assert http.seen == ["seed", "r1"]                                    # the used seed was not sent again


def test_a_newly_pasted_token_replaces_a_dead_one(tmp_path):
    http, store = FakeHTTP({"seed"}), tmp_path / "token.json"
    acc.Account(1, "seed", store, http).access_token()
    http.valid.clear()                                                    # the sign-in lapsed
    with pytest.raises(acc.AuthError):
        acc.Account(1, "seed", store, http).access_token(force=True)
    http.valid.add("pasted")
    blob = json.dumps({"id_token": "x", "refresh_token": "pasted", "profile": {}})
    assert acc.Account(1, blob, store, http).access_token(force=True).startswith("a")
    assert acc.parse_refresh_token('"abc"') == "abc" and acc.parse_refresh_token(blob) == "pasted"
    pretty = json.dumps({"id_token": "x", "refresh_token": "pasted", "profile": {"a": 1}}, indent=4)
    assert acc.parse_refresh_token("\n" + pretty + "\n") == "pasted"       # as copied from the browser's viewer


def test_fpl_reports_the_price_of_a_hit_not_the_points_spent():
    mt = {"transfers": {"cost": 4, "status": "cost", "limit": 2, "made": 0, "bank": 59, "value": 939},
          "chips": [{"name": "wildcard", "status_for_entry": "available", "is_pending": False}]}   # as FPL sent it
    assert acc.points_spent(mt) == 0 and acc.free_transfers_left(mt) == 2 and acc.active_chip(mt) is None
    mt["transfers"]["made"] = 3
    assert acc.points_spent(mt) == 4 and acc.free_transfers_left(mt) == 0
    mt["transfers"].update(status="unlimited", limit=None)
    assert acc.points_spent(mt) == 0 and acc.unlimited_transfers(mt)
    mt["chips"][0]["is_pending"] = True
    assert acc.active_chip(mt) == "wildcard"


def test_a_dead_pasted_token_is_tried_once(tmp_path):
    http, store = FakeHTTP(set()), tmp_path / "token.json"
    for _ in range(3):
        with pytest.raises(acc.AuthError):
            acc.Account(1, "stale", store, http).access_token()
    assert http.seen == ["stale"]


# ----------------------------------------------------------------------------- timing
def test_when_to_plan_and_when_to_act():
    s = autopilot.Settings(enabled=True)
    now = pd.Timestamp("2026-11-20 08:00", tz="UTC")
    ago = lambda h, **kw: {"time": str(now - pd.Timedelta(hours=h)), "flags": {"1": "a|None"}, **kw}  # noqa: E731
    d = lambda left, last, can=True, in_play=False, flags=None: autopilot.decide(  # noqa: E731
        left, last, flags or {"1": "a|None"}, s, can, in_play, now)[0]
    assert d(60, None) == "skip"                                  # far away
    assert d(40, None) == "plan"                                  # inside two days: plan, do not touch
    assert d(27, None) == "apply"                                 # inside the acting window
    assert d(27, None, can=False) == "plan"                       # no sign-in: plans only
    assert d(27, None, in_play=True) == "plan"                    # previous gameweek still being played
    assert d(5, None, in_play=True) == "apply"                    # ... unless the deadline is close
    assert d(27, ago(1, applied=False)) == "apply"                # a failed attempt is retried
    assert d(20, ago(2, applied=True)) == "skip"                  # settled, nothing new
    assert d(20, ago(2, applied=True), flags={"1": "d|75"}) == "apply"    # injury news
    assert d(14, ago(13, applied=True)) == "apply"                # periodic refresh
    assert d(4, ago(4, applied=True)) == "apply"                  # final check
    assert d(2, ago(2, applied=True)) == "skip"                   # already looked inside the last hours
    assert d(0.1, None) == "plan" and d(-1, None) == "skip"       # too late to act safely
    # checked once a day, this run may be the last before the deadline: do not wait for the old gameweek
    daily = autopilot.Settings(enabled=True, apply_within_h=24, check_every_h=24)
    assert autopilot.decide(23, None, {}, daily, True, True, now)[0] == "apply"
    assert autopilot.decide(23, None, {}, autopilot.Settings(enabled=True, apply_within_h=24), True, True, now)[0] == "plan"
    assert autopilot.decide(30, None, {}, daily, True, False, now)[0] == "plan"       # outside the 24 hours
    # checked every 11 hours with a 24-hour window: two looks, the second being the last before the deadline
    s11 = autopilot.Settings(enabled=True, apply_within_h=24, check_every_h=11)
    d11 = lambda left, last, in_play=False: autopilot.decide(left, last, {"1": "a|None"}, s11, True, in_play, now)[0]  # noqa: E731
    assert d11(39, None) == "plan" and d11(28, ago(11, applied=False)) == "skip"      # not yet inside the 24 hours
    assert d11(17, ago(11, applied=False)) == "apply"                                 # first look inside: act
    assert d11(6, ago(11, applied=True)) == "apply"                                   # no later look: final check
    assert d11(12.5, ago(11, applied=True)) == "skip"                                 # another look will still come
    assert d11(1.5, ago(11, applied=True)) == "apply"
    assert d11(11.2, ago(11, applied=True)) == "apply"            # the look after this one would come 12 minutes before: too late
    assert d11(20, None, in_play=True) == "plan" and d11(12, None, in_play=True) == "apply"


# ----------------------------------------------------------------------------- signed-in team state
def test_team_state_prefers_the_account_over_reconstruction(world, monkeypatch):
    players, squad, bs = world
    bs = {**bs, "events": [{"id": 1, "is_current": True, "is_next": False, "finished": True},
                           {"id": 2, "is_current": False, "is_next": True}],
          "chips": [{"name": "wildcard", "start_event": 2, "stop_event": 19}]}
    for e in bs["elements"]:
        e["cost_change_start"] = 0
    api = my_team.fpl_api
    monkeypatch.setattr(api, "entry", lambda *a, **k: {"started_event": 1, "summary_overall_points": 50})
    monkeypatch.setattr(api, "entry_history", lambda *a, **k: {"current": [{"event": 1, "event_transfers": 0, "event_transfers_cost": 0}], "chips": []})
    monkeypatch.setattr(api, "entry_transfers", lambda *a, **k: [])
    monkeypatch.setattr(api, "entry_picks", lambda *a, **k: {
        "picks": [{"element": p, "position": i + 1, "is_captain": i == 0, "is_vice_captain": i == 1} for i, p in enumerate(squad)],
        "entry_history": {"bank": 5}})
    public = my_team.load(1, bs)
    assert public.free_transfers == 1 and public.bank == 5 and public.active_chip is None

    fpl = FakeFPL(players, squad, bank=33, free=4)
    fpl.picks[3]["selling_price"] -= 2
    st = my_team.load(1, bs, private=fpl.my_team())
    assert st.free_transfers == 4 and st.bank == 33 and sorted(st.squad) == sorted(squad)
    assert st.sell_price[fpl.picks[3]["element"]] == fpl.picks[3]["selling_price"]
    assert not any("reconstructed" in n for n in st.notes)

    fpl.post_transfers(2, [], "wildcard")
    st = my_team.load(1, bs, private=fpl.my_team())
    assert st.active_chip == "wildcard" and st.free_transfers == 1      # unlimited now; the banked count is unchanged


# ----------------------------------------------------------------------------- Discord reports
def test_discord_message_fits_the_limits_and_goes_only_to_discord(monkeypatch):
    from fpl_smart import notify
    body = notify.discord_payload("T" * 400, ["x" * 900] * 9, ["p" * 900] * 5, None,
                                  {"Plan": "a" * 2000, "Line-up": "b" * 2000, "Deadline": "c" * 2000, "Extra": "d"}, "f" * 900)
    e = body["embeds"][0]
    size = len(e["title"]) + len(e["description"]) + len(e["footer"]["text"]) + sum(len(x["name"]) + len(x["value"]) for x in e["fields"])
    assert size <= 6000 and len(e["fields"]) == 4 and e["fields"][0]["name"] == "Needs your attention"
    assert e["color"] == notify.COLOURS["warn"] and body["allowed_mentions"] == {"parse": []}

    sent = []
    monkeypatch.setattr(notify.requests, "post", lambda url, **kw: sent.append((url, kw)) or SimpleNamespace(status_code=204))
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert notify.discord("hello") is False and sent == []                       # not set up: silent
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.com/api/webhooks/1/abc")
    assert notify.discord("hello") is False and sent == []                       # not Discord: never sent
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", ' "https://discord.com/api/webhooks/1/abc" ')
    assert notify.discord("hello", ["one"]) is True and sent[0][0] == "https://discord.com/api/webhooks/1/abc"
    assert sent[0][1]["json"]["embeds"][0]["description"] == "• one"

    def boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(notify.requests, "post", boom)
    assert notify.discord("hello") is False                                      # a failed send never raises


def test_gameweek_result_is_reported_once(monkeypatch, tmp_path):
    monkeypatch.setattr(autopilot, "REPORTS", tmp_path)
    monkeypatch.setattr(autopilot, "team_id", lambda: 1)
    hist = {"current": [{"event": 10, "points": 50, "total_points": 534, "overall_rank": 6_300_000, "event_transfers": 1,
                         "event_transfers_cost": 0, "points_on_bench": 3},
                        {"event": 11, "points": 71, "total_points": 601, "overall_rank": 5_900_000, "event_transfers": 11,
                         "event_transfers_cost": 4, "points_on_bench": 6}],
            "chips": [{"name": "wildcard", "event": 11}]}
    monkeypatch.setattr(autopilot.fpl_api, "entry_history", lambda *a, **k: hist)
    got = []
    ok = {"value": False}
    monkeypatch.setattr(autopilot.notify, "discord", lambda title, lines=(), problems=(), kind=None, **k: got.append((title, list(lines), kind)) or ok["value"])
    bs = {"events": [{"id": 10, "finished": True, "data_checked": True, "average_entry_score": 48},
                     {"id": 11, "finished": True, "data_checked": True, "average_entry_score": 55},
                     {"id": 12, "finished": False, "data_checked": False}]}
    assert autopilot.result_report(bs) is False and len(got) == 1        # Discord did not take it: not marked as done
    ok["value"] = True
    assert autopilot.result_report(bs) is True and autopilot.result_report(bs) is False and len(got) == 2
    title, lines, kind = got[-1]
    text = " ".join(lines)
    assert title == "GW11 result: 67 points" and kind == "ok"
    assert "71 minus 4" in text and "average was 55" in text and "up 400,000 places" in text and "Wildcard" in text


def test_idle_check_reports_at_most_every_few_hours(monkeypatch, tmp_path):
    monkeypatch.setattr(autopilot, "REPORTS", tmp_path)
    got = []
    monkeypatch.setattr(autopilot.notify, "discord", lambda title, lines=(), problems=(), kind=None, **k: got.append((title, list(lines), list(problems), kind)) or True)
    s = autopilot.Settings(enabled=True, apply_within_h=24)
    now = pd.Timestamp("2026-11-15 20:00", tz="UTC")
    good = SimpleNamespace(access_token=lambda force=False: "a")
    assert autopilot.idle_check(good, 11, pd.Timestamp(DEADLINE), 135.0, now, s) is None
    assert autopilot.idle_check(good, 11, pd.Timestamp(DEADLINE), 133.0, now, s) is None
    assert len(got) == 1 and got[0][0] == "Check: nothing to do yet" and "sign-in: working" in " ".join(got[0][1])

    def dead(force=False):
        raise acc.AuthError("expired")
    (tmp_path / "autopilot-state.json").unlink()
    assert autopilot.idle_check(SimpleNamespace(access_token=dead), 11, pd.Timestamp(DEADLINE), 100.0, now, s) == "expired"
    assert got[-1][3] == "error" and "expired" in got[-1][2][0]
