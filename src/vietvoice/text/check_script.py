"""Kiểm kịch bản thu âm bằng G2P (P2): mọi âm tiết phải phân tích được, liệt kê hết âm tiết lỗi.

Chạy: ``python -m vietvoice.text.check_script data/scripts/<speaker_id>.tsv --dialect bac``.
File kịch bản mỗi dòng ``id <TAB> câu`` (đặc tả mục 5), câu đã chuẩn hoá; thêm ``--normalize``
để chạy chuẩn hoá (``vietvoice.text.normalize``) trước. Mã thoát: 0 không lỗi, 1 có âm tiết
lỗi, 2 file kịch bản hoặc cấu hình sai.
"""

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from vietvoice.text.g2p import DIALECTS, PUNCTUATION, load_lexicon, split_text, word_tokens
from vietvoice.text.normalize import normalize as normalize_text

Normalizer = Callable[[str, str], str]


@dataclass(frozen=True)
class ScriptError:
    """Một chỗ G2P không đọc được trong kịch bản."""

    line_id: str
    text: str  # câu gốc trong kịch bản
    piece: str  # từ hoặc ký tự lỗi; "" khi lỗi cả câu (câu rỗng, chuẩn hoá lỗi)
    message: str


def check_script(
    lines: Iterable[tuple[str, str]], dialect: str, normalize: Normalizer | None = None
) -> list[ScriptError]:
    """Chạy G2P trên mọi câu ``(id, câu)``, gom tất cả chỗ lỗi thay vì dừng ở lỗi đầu tiên.

    Vùng sai hoặc từ điển ngoại lệ hỏng thì ném ``ValueError`` ngay, không tính là lỗi câu.
    """
    load_lexicon(dialect)
    errors: list[ScriptError] = []
    for line_id, text in lines:
        errors.extend(_check_line(line_id, text, dialect, normalize))
    return errors


def _check_line(line_id: str, text: str, dialect: str, normalize: Normalizer | None) -> list[ScriptError]:
    checked = text
    if normalize is not None:
        try:
            checked = normalize(text, dialect)
        except ValueError as err:
            return [ScriptError(line_id, text, "", f"chuẩn hoá lỗi: {err}")]
        if not isinstance(checked, str):
            raise TypeError(f"normalize phải trả về chuỗi, gặp {type(checked).__name__} ở câu {line_id!r}")

    pieces = [p for p in split_text(checked) if p not in PUNCTUATION]
    if not pieces:
        return [ScriptError(line_id, text, "", "câu không có âm tiết nào")]
    errors = []
    for piece in pieces:
        try:
            word_tokens(piece, dialect)
        except ValueError as err:
            errors.append(ScriptError(line_id, text, piece, str(err)))
    return errors


def read_script(path: Path) -> list[tuple[str, str]]:
    """Đọc file kịch bản ``id <TAB> câu``; bỏ dòng trống. Dòng sai hoặc id lặp thì ném ``ValueError``."""
    if not path.is_file():
        raise ValueError(f"không có file kịch bản {path}")
    lines: list[tuple[str, str]] = []
    first_line: dict[str, int] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name} dòng {number}"
        if "\t" not in line:
            raise ValueError(f"{where}: cần dạng id<TAB>câu, gặp {line!r}")
        line_id, text = line.split("\t", 1)
        line_id = line_id.strip()
        if not line_id:
            raise ValueError(f"{where}: thiếu id câu")
        if line_id in first_line:
            raise ValueError(f"{where}: id {line_id!r} bị lặp, đã có ở dòng {first_line[line_id]}")
        first_line[line_id] = number
        lines.append((line_id, text))
    return lines


def format_report(errors: Sequence[ScriptError], line_count: int, dialect: str) -> str:
    """Báo cáo tiếng Việt: số câu, số câu lỗi, số âm tiết lỗi, rồi từng lỗi kèm id câu."""
    bad_lines = len({e.line_id for e in errors})
    bad_pieces = sum(1 for e in errors if e.piece)
    head = f"Đã kiểm {line_count} câu (giọng {dialect}): "
    if not errors:
        return head + "không có lỗi."
    rows = [f"  {e.line_id}\t{e.piece or '(cả câu)'}\t{e.message}" for e in errors]
    return "\n".join([head + f"{bad_lines} câu lỗi, {bad_pieces} âm tiết lỗi.", *rows])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vietvoice.text.check_script",
        description="Chạy G2P trên kịch bản thu âm, liệt kê mọi âm tiết không đọc được.",
    )
    parser.add_argument("script", type=Path, help="file kịch bản id<TAB>câu")
    parser.add_argument("--dialect", required=True, choices=DIALECTS, help="vùng giọng")
    parser.add_argument("--normalize", action="store_true", help="chuẩn hoá câu trước khi chạy G2P")
    args = parser.parse_args(argv)

    try:
        normalize = normalize_text if args.normalize else None
        lines = read_script(args.script)
        errors = check_script(lines, args.dialect, normalize)
    except ValueError as err:
        print(f"lỗi: {err}", file=sys.stderr)
        return 2
    print(format_report(errors, len(lines), args.dialect))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
