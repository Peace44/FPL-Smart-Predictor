"""The unattended /fpl-improve run: when it is due, and the worktree flow end to end, with a fake
`claude` in a throwaway git repository. No network, no Claude Code."""
import json
import subprocess
import sys

import pytest

from fpl_smart import codechange, improve, tuning

FAKE_CLAUDE = r'''#!{python}
import json, os, pathlib, subprocess, sys, time
prompt = sys.stdin.read()
mode = os.environ["FAKE_MODE"]
cwd = pathlib.Path.cwd()
pathlib.Path(os.environ["FAKE_OUT"]).write_text(json.dumps({{
    "prompt": prompt, "args": sys.argv[1:], "cwd": str(cwd),
    "webhook": os.environ.get("DISCORD_WEBHOOK_URL"), "token": os.environ.get("FPL_REFRESH_TOKEN"),
    "pythonpath": os.environ.get("PYTHONPATH"), "env_file": (cwd / ".env").read_text(),
    "auth": (cwd / "data" / "auth").exists(), "raw_link": (cwd / "data" / "raw").is_symlink(),
    "ledger": (cwd / "tuning" / "ledger.md").read_text(), "brief": (cwd / "reports" / "gw11_improvement.md").exists()}}))
if mode == "sleep":
    time.sleep(60)
if mode in ("kept", "kept_failing"):
    (cwd / "src" / "new.py").write_text("X = 1\n")
    if mode == "kept_failing":
        (cwd / "tests" / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    with open(cwd / "tuning" / "ledger.md", "a") as f:
        f.write("- edited by the session\n")
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-qam", "Try an idea"], check=True)
outcome = "kept" if mode.startswith("kept") else "no_change"
print(json.dumps({{"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": 1.25,
                   "permission_denials": [{{"tool_name": "Bash", "tool_input": {{"command": "curl x"}}}}],
                   "structured_output": {{"outcome": outcome, "summary": "Did a thing (z 2.5).",
                                          "ledger_line": "- Code change: a thing (b): z 2.5; kept" if outcome == "kept" else "",
                                          "report": "## Account\n\nAll of it."}}}}))
'''


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A project with the layout codechange expects: committed code, ignored caches and secrets, and a
    live ledger line not yet committed."""
    root = tmp_path / "project"
    for d in ("src", "tests", "tuning", "data/raw", "data/processed", "data/auth", "reports", ".venv/bin"):
        (root / d).mkdir(parents=True)
    # The project's own ignore rules: they must cover the worktree's .venv and data/raw, which are symlinks.
    (root / ".gitignore").write_text((codechange.ROOT / ".gitignore").read_text())
    (root / "src" / "mod.py").write_text("A = 1\n")
    (root / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    (root / "tuning" / "params.json").write_text("{}\n")
    (root / "tuning" / "ledger.md").write_text("# Ledger\n")
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                 "GIT_COMMITTER_EMAIL": "t@t", "FPL_TEAM_ID": "123", "DISCORD_WEBHOOK_URL": "https://secret",
                 "FPL_REFRESH_TOKEN": "tok"}.items():
        monkeypatch.setenv(k, v)
    git("init", "-q", "-b", "main", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-qm", "init", cwd=root)
    with open(root / "tuning" / "ledger.md", "a") as f:
        f.write("## GW11: live entry\n")
    (root / ".env").write_text("FPL_TEAM_ID=123\nDISCORD_WEBHOOK_URL=https://secret\n")
    (root / "data" / "auth" / "token.json").write_text("{}")
    (root / "data" / "processed" / "cache.pkl").write_text("x")
    (root / "reports" / "gw11_improvement.md").write_text("# brief\n")
    py = root / ".venv" / "bin" / "python"
    py.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    py.chmod(0o755)
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_OUT", str(tmp_path / "seen.json"))
    return root, fake, tmp_path / "worktrees"


def run_work(repo, monkeypatch, mode, hours=1.0):
    root, fake, base = repo
    monkeypatch.setenv("FAKE_MODE", mode)
    head = git("rev-parse", "HEAD", cwd=root)
    res = codechange.work(11, hours, log=lambda *_: None, root=root, base=base, claude=str(fake))
    seen = json.loads((root.parent / "seen.json").read_text())
    assert git("rev-parse", "HEAD", cwd=root) == head                       # main never moves
    assert git("status", "--porcelain", cwd=root) == "M tuning/ledger.md"  # nor its working tree
    return res, seen


def test_a_kept_change_waits_on_its_branch_and_the_session_sees_no_secrets(repo, monkeypatch):
    root, _, base = repo
    res, seen = run_work(repo, monkeypatch, "kept")
    assert (res.outcome, res.branch, res.commits, res.files) == ("kept", "improve/gw11-auto", 1, ["src/new.py"])
    assert res.session == "s-1" and res.cost == 1.25 and res.ledger_line.startswith("- Code change:")
    assert res.notes == ["1 command(s) it tried were not permitted (Bash)."] and not res.problems
    # The session: the skill and its unattended section, refused rather than asked, no secrets, own code first.
    assert seen["prompt"].startswith("/fpl-improve reports/gw11_review.md") and "Unattended run" in seen["prompt"]
    assert "improve/gw11-auto" in seen["prompt"]
    assert seen["args"][seen["args"].index("--permission-mode") + 1] == "dontAsk"
    assert "Bash(git push*)" in seen["args"] and "Bash(.venv/bin/fpl apply*)" in seen["args"]
    assert f"Read(/{codechange.ROOT}/data/auth/**)" in seen["args"] and f"Edit(/{codechange.ROOT}/**)" in seen["args"]
    assert seen["webhook"] is None and seen["token"] is None and seen["env_file"] == "FPL_TEAM_ID=123\n"
    assert seen["pythonpath"] == str(base / "improve-gw11-auto" / "src") and seen["cwd"] == str(base / "improve-gw11-auto")
    assert not seen["auth"] and seen["raw_link"] and seen["brief"]
    assert "## GW11: live entry" in seen["ledger"]                         # the live settings, not HEAD's
    # The live ledger copy stays out of the branch; the worktree is gone, the branch stays.
    assert "tuning/ledger.md" not in git("show", "--name-only", "--format=", "improve/gw11-auto", cwd=root)
    assert not (base / "improve-gw11-auto").exists()
    assert git("branch", "--list", "improve/gw11-auto", cwd=root)
    # A second run on the same gameweek gets a branch of its own.
    res2, _ = run_work(repo, monkeypatch, "no_change")
    assert res2.branch == "improve/gw11-auto-2" and res2.outcome == "no_change"
    assert not git("branch", "--list", "improve/gw11-auto-2", cwd=root)      # nothing committed: no branch


def test_a_kept_change_whose_tests_fail_is_a_failure(repo, monkeypatch):
    _, _, base = repo
    res, _ = run_work(repo, monkeypatch, "kept_failing")
    assert res.outcome == "failed" and res.problems[0].startswith("The test suite fails on its branch")
    assert res.worktree == base / "improve-gw11-auto" and res.worktree.exists()    # kept for a look


def test_a_session_over_its_budget_is_stopped(repo, monkeypatch):
    res, _ = run_work(repo, monkeypatch, "sleep", hours=2 / 3600)
    assert res.outcome == "failed" and "budget" in res.summary and res.commits == 0


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.setattr(codechange, "REPORTS", tmp_path)
    monkeypatch.setattr(improve, "STATE", tmp_path / "improve-state.json")
    monkeypatch.setattr(improve, "REPORTS", tmp_path)
    monkeypatch.setattr(tuning, "DIR", tmp_path)
    monkeypatch.setattr(tuning, "LEDGER", tmp_path / "ledger.md")
    sent = []
    monkeypatch.setattr(codechange, "_report", lambda log, title, lines=(), problems=(), kind=None: sent.append(
        (title, kind, list(lines), list(problems))) or True)
    return tmp_path, sent


def test_record_writes_the_account_the_ledger_and_the_report(reports):
    tmp, sent = reports
    res = codechange.Result(11, "improve/gw11-auto", "kept", "Did a thing (z 2.5).", "## Account", "Code change: x; kept",
                            commits=2, files=["src/a.py", "tests/test_a.py"], notes=["note"], session="s-1", cost=0.5)
    codechange.record(res, log=lambda *_: None)
    text = (tmp / "gw11_code_change.md").read_text()
    assert "- Outcome: **kept**" in text and "## Account" in text and "`s-1`" in text
    assert (tmp / "ledger.md").read_text().rstrip().splitlines()[-1].startswith("- Code change: x; kept [unattended run")
    assert improve._past_structural()[0].startswith("x; kept")                # the next brief lists it as tried
    title, kind, lines, _ = sent[0]
    assert (title, kind) == ("GW11 code change ready to merge", "ok")
    assert any("git merge improve/gw11-auto" in x for x in lines) and "note" in lines


@pytest.mark.parametrize("state,free,runs", [
    ({"brief_gw": 11}, 30.0, (11, codechange.MAX_HOURS)),
    ({"brief_gw": 11}, 2.5, (11, 2.5)),                   # ends before the planning window opens
    ({"brief_gw": 11}, 0.5, None),                        # too little time: waits
    ({"brief_gw": 11, "code_gw": 11}, 30.0, None),         # already worked on
    ({"done_gw": 11}, 30.0, None),                         # the settings run changed one: no brief
])
def test_due(reports, monkeypatch, state, free, runs):
    improve._save_state(**state)
    monkeypatch.setattr(improve, "hours_free", lambda: (free, 12))
    assert codechange.due(log=lambda *_: None) == runs


def test_a_scheduled_run_starts_once_per_brief_and_can_be_switched_off(reports, monkeypatch):
    tmp, sent = reports
    (tmp / "gw11_improvement.md").write_text("# brief\n")
    improve._save_state(brief_gw=11)
    monkeypatch.setattr(improve, "hours_free", lambda: (30.0, 12))
    calls = []

    def boom(gw, hours, log):
        calls.append((gw, hours))
        raise RuntimeError("no claude")
    monkeypatch.setattr(codechange, "work", boom)
    monkeypatch.setenv("FPL_SELF_IMPROVE_CODE", "off")
    assert codechange.run(if_due=True, log=lambda *_: None) == 0 and not calls
    monkeypatch.delenv("FPL_SELF_IMPROVE_CODE")
    assert codechange.run(if_due=True, log=lambda *_: None) == 1
    assert sent[0][:2] == ("GW11 code change failed", "error") and "no claude" in sent[0][2][0]
    assert codechange.run(if_due=True, log=lambda *_: None) == 0 and len(calls) == 1   # not again by itself
    assert codechange.run(log=lambda *_: None) == 1 and len(calls) == 2                # by hand, yes


def test_the_settings_report_says_who_works_on_the_brief():
    out = [improve.Outcome("availability", skipped="needs 3; has 1")]
    auto = " ".join(improve.result_lines("selection", out, None, "reports/gw11_improvement.md", auto_code=True))
    hand = " ".join(improve.result_lines("selection", out, None, "reports/gw11_improvement.md"))
    assert "unattended" in auto and "To act on it" not in auto and "To act on it, run /fpl-improve" in hand


def test_claude_is_found_off_the_path(tmp_path, monkeypatch):
    """The Windows scheduler starts WSL without ~/.local/bin on PATH."""
    fake = tmp_path / ".local" / "bin" / "claude"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("FPL_CLAUDE", raising=False)
    monkeypatch.setattr(codechange.Path, "home", classmethod(lambda cls: tmp_path))
    assert codechange.claude_bin() == str(fake)
    monkeypatch.setenv("FPL_CLAUDE", "/nowhere/claude")       # a wrong setting falls through to what exists
    assert codechange.claude_bin() == str(fake)
