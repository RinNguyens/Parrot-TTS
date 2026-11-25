"""Đếm lượt đọc theo giọng cho bảng xếp hạng giọng (trang web /giong).

Mỗi lần máy đọc đọc xong một đoạn (POST /tts thành công) là một lượt của giọng đó. Chỉ lưu SỐ ĐẾM theo ngày (UTC)
và tổng, không lưu văn bản người dùng gửi. Lưu vào một file JSON (ghi đè an toàn qua file tạm), giữ 60 ngày gần
nhất; tổng cộng dồn giữ mãi.

    u = Usage(Path("/workspace/usage.json"))
    u.add("fleurs")
    u.stats()  # {"updated": ..., "days": 7, "voices": {"fleurs": {"week": 1, "prev": 0, "total": 1}}}

Dùng chung cho `vietvoice serve` (vietvoice.server), infra/vast/tts_server.py và web/scripts/tts-stub.py
(hai file sau nạp qua infra/vast/usage.py).
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc  # noqa: UP017 - datetime.UTC chỉ có từ Python 3.11; máy Vast chạy 3.10

WINDOW = 7  # ngày: "tuần này" là 7 ngày tính cả hôm nay, "tuần trước" là 7 ngày trước đó
KEEP = 60  # ngày giữ số đếm theo ngày


def _today() -> date:
    return datetime.now(UTC).date()


class Usage:
    def __init__(self, path: Path | None, today: Callable[[], date] = _today) -> None:
        self.path = path
        self.today = today
        self.lock = threading.Lock()
        self.days: dict[str, dict[str, int]] = {}
        self.total: dict[str, int] = {}
        if path and path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                self.days = {k: {v: int(n) for v, n in day.items()} for k, day in d.get("days", {}).items()}
                self.total = {v: int(n) for v, n in d.get("total", {}).items()}
            except (OSError, ValueError, AttributeError):
                # file hỏng thì đếm lại từ đầu, không làm máy đọc ngừng
                self.days, self.total = {}, {}

    def add(self, voice: str, n: int = 1) -> None:
        with self.lock:
            key = self.today().isoformat()
            day = self.days.setdefault(key, {})
            day[voice] = day.get(voice, 0) + n
            self.total[voice] = self.total.get(voice, 0) + n
            cutoff = (self.today() - timedelta(days=KEEP)).isoformat()
            self.days = {k: v for k, v in self.days.items() if k > cutoff}
            self._save()

    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"days": self.days, "total": self.total}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def stats(self) -> dict:
        with self.lock:
            t = self.today()
            week = {(t - timedelta(days=i)).isoformat() for i in range(WINDOW)}
            prev = {(t - timedelta(days=i)).isoformat() for i in range(WINDOW, 2 * WINDOW)}
            voices: dict[str, dict[str, int]] = {}
            for v, total in self.total.items():
                voices[v] = {
                    "week": sum(self.days.get(d, {}).get(v, 0) for d in week),
                    "prev": sum(self.days.get(d, {}).get(v, 0) for d in prev),
                    "total": total,
                }
            return {"updated": datetime.now(UTC).isoformat(timespec="seconds"), "days": WINDOW, "voices": voices}
