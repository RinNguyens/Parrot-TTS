"""Thống kê sau mỗi lần chạy pipeline: số câu pass / review / fail ở từng bước (review P5)."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import get_args

from vietvoice.data.manifest import QcStatus

STATUSES: tuple[str, ...] = get_args(QcStatus)


class RunStats:
    """Đếm trạng thái theo bước; các bước giữ thứ tự lần đầu xuất hiện."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}

    def add(self, step: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"trạng thái {status!r} không hợp lệ; chỉ có: {', '.join(STATUSES)}")
        counts = self._counts.setdefault(step, dict.fromkeys(STATUSES, 0))
        counts[status] += 1

    def add_many(self, step: str, statuses: Iterable[str]) -> None:
        for status in statuses:
            self.add(step, status)

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {step: {**counts, "total": sum(counts.values())} for step, counts in self._counts.items()}

    def table(self) -> str:
        rows = [("Bước", *STATUSES, "Tổng")]
        rows += [(step, *(str(c[s]) for s in STATUSES), str(c["total"])) for step, c in self.to_dict().items()]
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
