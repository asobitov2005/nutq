from __future__ import annotations

import sys

import pytest

from nutq_asr.cli import main as cli
from nutq_asr.cli.doctor import system_report


def test_top_level_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["nutq", "--help"])
    cli.main()
    assert "nutq train" in capsys.readouterr().out


def test_unknown_command(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["nutq", "unknown"])
    with pytest.raises(SystemExit, match="Unknown command"):
        cli.main()


def test_doctor_report_has_accelerator_state() -> None:
    report = system_report()
    assert isinstance(report["cuda_available"], bool)
    assert report["gpu_count"] == len(report["gpus"])
