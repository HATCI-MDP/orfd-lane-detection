"""Tab completion: the committed scripts match the parsers, and both shells
complete options, choices and subcommands from them."""

import importlib.util
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH_PATH = ROOT / "completions/offroad-autonomy.bash"
POWERSHELL_PATH = ROOT / "completions/offroad-autonomy.ps1"


def _generator():
    path = ROOT / "scripts/generate_completions.py"
    spec = importlib.util.spec_from_file_location("generate_completions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_committed_completions_match_the_parsers():
    generator = _generator()
    specs = generator.collect()
    message = "Completions are stale: run python scripts/generate_completions.py"
    assert BASH_PATH.read_text(encoding="utf-8") == generator.render_bash(specs), message
    assert POWERSHELL_PATH.read_text(encoding="utf-8") == generator.render_powershell(specs), (
        message
    )


def test_every_script_and_subcommand_is_completed():
    specs = _generator().collect()
    scripts = {path.name for path in (ROOT / "scripts").glob("*.py")}
    assert scripts | {"offroad-autonomy"} <= set(specs)
    assert specs["diagnose_perception.py"].subcommands == ["capture", "analyze", "full"]


def test_a_word_the_shells_cannot_hold_is_refused():
    generator = _generator()
    parser = generator.argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["two words"])
    with pytest.raises(ValueError, match="two words"):
        generator.describe(parser, "tool.py", {})


def _bash(*words: str) -> list[str]:
    """COMPREPLY for the command line ``words``; the last word is the one
    being completed, so pass "" to complete after a space."""
    line = " ".join(shlex.quote(word) for word in words)
    script = (
        f"source {shlex.quote(str(BASH_PATH))}\n"
        f"COMP_WORDS=({line})\n"
        f"COMP_CWORD={len(words) - 1}\n"
        # The registered function, so a missing `complete` line fails too.
        'fn=$(complete -p "${COMP_WORDS[0]}" | sed -E "s/.*-F ([^ ]+).*/\\1/")\n'
        '"$fn"\n'
        'printf "%s\\n" "${COMPREPLY[@]}"\n'
    )
    out = subprocess.run(
        ["bash", "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return sorted(word for word in out.splitlines() if word)


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_bash_completes_options_by_prefix():
    assert _bash("offroad-autonomy", "--he") == ["--headless", "--help"]


@needs_bash
def test_bash_completes_choices_for_an_option_value():
    assert _bash("offroad-autonomy", "--log-level", "") == ["DEBUG", "ERROR", "INFO", "WARNING"]


@needs_bash
def test_bash_offers_nothing_for_a_numeric_value():
    assert _bash("offroad-autonomy", "--benchmark-seconds", "") == []


@needs_bash
def test_bash_completes_paths_for_a_path_value():
    assert _bash("offroad-autonomy", "--config", "configs/def") == ["configs/default.yaml"]


@needs_bash
def test_bash_completes_script_subcommands_and_their_options():
    script = "scripts/diagnose_perception.py"
    assert _bash("python", script, "") == ["analyze", "capture", "full"]
    assert _bash("python3", script, "analyze", "--s") == ["--side", "--src"]
    assert _bash("python", script, "analyze", "--side", "") == ["dashcam"]


@needs_bash
def test_bash_does_not_take_an_option_value_for_a_subcommand():
    assert _bash("python", "scripts/diagnose_perception.py", "--config", "full", "--") == [
        "--config",
        "--help",
    ]


@needs_bash
def test_bash_leaves_other_python_commands_to_file_completion():
    assert _bash("python", "elsewhere.py", "--") == []


def _powershell(line: str) -> list[str]:
    script = (
        f". '{POWERSHELL_PATH}'\n"
        f"(TabExpansion2 -inputScript '{line}' -cursorColumn {len(line)}).CompletionMatches"
        " | ForEach-Object { $_.CompletionText }\n"
    )
    out = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(word for word in out.splitlines() if word)


needs_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")


@needs_pwsh
def test_powershell_completes_options_choices_and_subcommands():
    assert _powershell("offroad-autonomy --he") == ["--headless", "--help"]
    assert _powershell("offroad-autonomy --log-level ") == ["DEBUG", "ERROR", "INFO", "WARNING"]
    script = "python scripts/diagnose_perception.py"
    assert _powershell(f"{script} ") == ["analyze", "capture", "full"]
    assert _powershell(f"{script} analyze --side ") == ["dashcam"]
    assert _powershell(f"{script} --config full --") == ["--config", "--help"]
