"""Kiểm kịch bản thu âm bằng G2P (P2): mọi âm tiết phải phân tích được, liệt kê hết âm tiết lỗi.

API cần có trong ``parrot_tts.text.check_script``:
- ``ScriptError(line_id, text, piece, message)``.
- ``check_script(lines, dialect, normalize=None) -> list[ScriptError]``.
- ``main(argv) -> int``: CLI ``python -m parrot_tts.text.check_script <file.tsv> --dialect bac [--normalize]``.
"""

import subprocess
import sys

import pytest

from parrot_tts.text import check_script as cs
from parrot_tts.text.check_script import ScriptError, check_script, main

GOOD = [("s001", "ăn cơm chưa?"), ("s002", "xem tiktok nhé.")]


def write_script(tmp_path, body: str):
    path = tmp_path / "speaker.tsv"
    path.write_text(body, encoding="utf-8")
    return path


# --- check_script ---


@pytest.mark.parametrize("dialect", ["bac", "nam"])
def test_clean_script_has_no_errors(dialect):
    assert check_script(GOOD, dialect) == []


def test_collects_every_failure_across_and_within_lines():
    lines = [("s1", "ăn cơm"), ("s2", "xyz ăn 5 qwe."), ("s3", "tiktok hay"), ("s4", "bàn phím abc")]
    errors = check_script(lines, "bac")
    assert [(e.line_id, e.piece) for e in errors] == [("s2", "xyz"), ("s2", "5"), ("s2", "qwe"), ("s4", "abc")]
    assert all(isinstance(e, ScriptError) and e.message for e in errors)
    assert errors[0].text == "xyz ăn 5 qwe."


def test_empty_line_is_an_error():
    errors = check_script([("s1", "  ")], "nam")
    assert len(errors) == 1 and errors[0].line_id == "s1"


def test_bad_dialect_raises_instead_of_being_collected():
    with pytest.raises(ValueError, match="hanoi"):
        check_script(GOOD, "hanoi")


def test_normalize_is_applied_and_its_errors_collected():
    def fake_normalize(text: str, dialect: str) -> str:
        if "%" in text:
            raise ValueError("không đọc được %")
        return text.replace("5", "năm")

    lines = [("s1", "5 con gà"), ("s2", "tăng 5%")]
    errors = check_script(lines, "bac", normalize=fake_normalize)
    assert [(e.line_id, e.piece) for e in errors] == [("s2", "")]
    assert "%" in errors[0].message


# --- CLI ---


def test_cli_exit_0_on_clean_script(tmp_path, capsys):
    path = write_script(tmp_path, "s001\tăn cơm chưa?\n\ns002\txem tiktok nhé.\n")
    assert main([str(path), "--dialect", "bac"]) == 0
    out = capsys.readouterr().out
    assert "2 câu" in out


def test_cli_exit_1_and_lists_failures(tmp_path, capsys):
    path = write_script(tmp_path, "s001\tăn cơm\ns002\txyz ăn qwe\ns003\tok 7\n")
    assert main([str(path), "--dialect", "nam"]) == 1
    out = capsys.readouterr().out
    for needle in ["3 câu", "2 câu lỗi", "3 âm tiết lỗi", "s002", "xyz", "qwe", "s003", "7"]:
        assert needle in out, needle
    assert "s001" not in out


@pytest.mark.parametrize(
    "body, line",
    [
        ("s001\tăn cơm\nkhông có tab\n", 2),
        ("s001\tăn cơm\n\tthiếu id\n", 2),
        ("s001\tăn cơm\ns002\tcơm\ns001\tlặp id\n", 3),
    ],
)
def test_cli_exit_2_on_malformed_file(tmp_path, capsys, body, line):
    path = write_script(tmp_path, body)
    assert main([str(path), "--dialect", "bac"]) == 2
    err = capsys.readouterr().err
    assert f"dòng {line}" in err and "speaker.tsv" in err


def test_cli_exit_2_on_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "khong_co.tsv"), "--dialect", "bac"]) == 2
    assert "khong_co.tsv" in capsys.readouterr().err


def test_cli_rejects_unknown_dialect(tmp_path):
    path = write_script(tmp_path, "s001\tăn cơm\n")
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--dialect", "hue"])
    assert exc.value.code == 2


def test_cli_normalize_runs_p1_normalize(tmp_path, capsys):
    path = write_script(tmp_path, "s001\t5 con gà\n")
    assert main([str(path), "--dialect", "nam"]) == 1  # không chuẩn hoá thì "5" là lỗi
    capsys.readouterr()
    assert main([str(path), "--dialect", "nam", "--normalize"]) == 0


def test_cli_normalize_passes_dialect(tmp_path, monkeypatch):
    calls = []

    def fake_normalize(text: str, dialect: str) -> str:
        calls.append(dialect)
        return text

    monkeypatch.setattr(cs, "normalize_text", fake_normalize)
    path = write_script(tmp_path, "s001\tăn cơm\n")
    assert main([str(path), "--dialect", "bac", "--normalize"]) == 0
    assert calls == ["bac"]


def test_module_entry_point(tmp_path):
    path = write_script(tmp_path, "s001\tăn cơm\ns002\tqwe\n")
    run = subprocess.run(
        [sys.executable, "-m", "parrot_tts.text.check_script", str(path), "--dialect", "bac"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1, run.stderr
    assert "qwe" in run.stdout
