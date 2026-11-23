"""A new season moves last season's per-gameweek files out of reports/. No network."""
import json

from fpl_smart import autopilot, season


def bs(year):
    return {"events": [{"id": 1, "deadline_time": f"{year}-08-14T17:30:00Z"},
                       {"id": 38, "deadline_time": f"{year + 1}-05-23T14:00:00Z"}]}


def test_season_from_the_first_deadline():
    assert season.of(bs(2026)) == "2026-27" and season.of(bs(2099)) == "2099-00"


def test_rollover_moves_last_seasons_files(tmp_path, monkeypatch):
    monkeypatch.setattr(season, "REPORTS", tmp_path)
    monkeypatch.setattr(season, "MARKER", tmp_path / "season.json")
    monkeypatch.setattr(season, "CURRENT_SEASON", "2026-27")
    sent = []
    monkeypatch.setattr(autopilot, "_report", lambda log, title, lines=(), problems=(), kind=None: sent.append(
        (title, list(problems))) or True)
    (tmp_path / "gw01").mkdir()
    (tmp_path / "gw01" / "autopilot.json").write_text("{}")
    seasonal = ["gw01_review.md", "gw38_review.md", "gw38_improvement.md", "autopilot-state.json",
                "improve-state.json"]
    kept = ["status.md", "latest.md", "actions.log", "auto.log"]
    for name in seasonal + kept:
        (tmp_path / name).write_text("x")
    quiet = dict(log=lambda *_: None)

    assert season.rollover(bs(2026), **quiet) is None             # first run: records the season only
    assert season.recorded() == "2026-27" and (tmp_path / "gw38_review.md").exists()
    assert season.rollover(bs(2026), **quiet) is None and not sent

    assert season.rollover(bs(2027), **quiet) == "2026-27"
    old = tmp_path / "2026-27"
    assert (old / "gw01" / "autopilot.json").exists() and all((old / n).exists() for n in seasonal)
    assert not any((tmp_path / n).exists() for n in seasonal + ["gw01"])
    assert all((tmp_path / n).exists() for n in kept)
    assert json.loads((tmp_path / "season.json").read_text()) == {"season": "2027-28"}
    assert len(sent) == 1 and sent[0][0] == "New season 2027-28"
    assert "CURRENT_SEASON" in sent[0][1][0]                       # config.py not moved on yet
    assert season.rollover(bs(2027), **quiet) is None and len(sent) == 1
