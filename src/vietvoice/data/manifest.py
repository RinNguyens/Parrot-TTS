"""Schema metadata huấn luyện (đặc tả mục 4) và bước 12: chia tập, xuất manifest JSONL.

- ``Utterance``: một dòng của ``data/manifests/{train,val,test}.jsonl``.
- ``Speaker``: một mục của ``data/speakers.json`` (một mảng JSON).
- ``split_utterances``: train/val/test = 94/3/3, chia đều theo (người nói, văn phong), cố định theo seed.

Luật kiểm tra (thông số trong ``manifest.yaml``):
id = ``<speaker_id>_<6 chữ số>``; đường dẫn tương đối nằm trong ``data/``; câu ngoài 1–15 s không
được ``pass`` (phải ``review``/``fail``), câu ``pass`` còn cần SNR >= ngưỡng QA (``qa.yaml``) và
CER ASR <= ``max_asr_cer``; câu ``train`` phải có giấy phép trong danh sách cho phép;
số phần tử ``tones`` bằng số token thanh ``T1``..``T8`` trong ``phonemes``.
"""

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from vietvoice.data.config import CONFIG_DIR, FrozenModel, load_qa_config

Dialect = Literal["bac", "hue", "nam"]
Style = Literal["tin_tuc", "ke_chuyen", "quang_cao", "hoi_thoai"]
Emotion = Literal["trung_tinh", "vui", "tram", "hao_hung"]
QcStatus = Literal["pass", "review", "fail"]
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = get_args(Split)

_TONE_TOKEN = re.compile(r"T(\d+)")


class DurationRange(FrozenModel):
    min: float = Field(gt=0)
    max: float = Field(gt=0)


class SplitConfig(FrozenModel):
    train: float = Field(ge=0, le=1)
    val: float = Field(ge=0, le=1)
    test: float = Field(ge=0, le=1)
    seed: int

    @model_validator(mode="after")
    def _sums_to_one(self) -> Self:
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("tỉ lệ train + val + test phải bằng 1")
        return self


class ManifestConfig(FrozenModel):
    sample_rate: int = Field(gt=0)
    speaker_id_pattern: str
    utterance_digits: int = Field(ge=1)
    duration_s: DurationRange
    train_licenses: tuple[str, ...] = Field(min_length=1)
    max_asr_cer: float = Field(ge=0, le=1)
    min_snr_db: float  # không ghi trong manifest.yaml: lấy qa.min_snr_db từ qa.yaml
    split: SplitConfig


class ManifestError(ValueError):
    """File manifest hoặc speakers.json sai; thông báo nêu dòng / mục lỗi."""


def load_manifest_config(path: Path | None = None, qa_path: Path | None = None) -> ManifestConfig:
    """Đọc ``manifest.yaml``; ngưỡng SNR cho câu ``pass`` lấy từ ``qa.yaml`` để hai nơi không lệch nhau."""
    if path is None and qa_path is None:
        return _default_config()
    return _load_config(path, qa_path)


@cache
def _default_config() -> ManifestConfig:
    return _load_config(None, None)


def _load_config(path: Path | None, qa_path: Path | None) -> ManifestConfig:
    path = path or CONFIG_DIR / "manifest.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "min_snr_db" in raw:
        raise ValueError(f"{path.name} sai cấu trúc: cần một bảng khoá, không được ghi min_snr_db (lấy từ qa.yaml)")
    try:
        return ManifestConfig.model_validate({**raw, "min_snr_db": load_qa_config(qa_path).qa.min_snr_db})
    except ValidationError as err:
        raise ValueError(f"{path.name} sai cấu trúc: {err}") from err


def _config(info: ValidationInfo) -> ManifestConfig:
    context = info.context or {}
    return context.get("config") or _default_config()


def _check_speaker_id(value: str, info: ValidationInfo) -> str:
    if not re.fullmatch(_config(info).speaker_id_pattern, value):
        raise ValueError(f"speaker_id {value!r} sai mẫu {_config(info).speaker_id_pattern}")
    return value


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


NonEmpty = Annotated[str, Field(min_length=1)]


class Utterance(_Record):
    """Một câu thu âm (một dòng manifest). Thứ tự trường theo bảng đặc tả mục 4."""

    id: NonEmpty
    speaker_id: NonEmpty
    audio_path: NonEmpty
    duration_s: float = Field(gt=0, strict=True)
    sample_rate: int = Field(strict=True)
    text_raw: NonEmpty
    text_norm: NonEmpty
    phonemes: NonEmpty
    tones: list[Annotated[int, Field(ge=1, le=8, strict=True)]] = Field(min_length=1)
    dialect: Dialect
    province: NonEmpty
    style: Style
    emotion: Emotion | None = None
    speech_rate: float = Field(gt=0, strict=True)
    f0_mean_st: float = Field(strict=True)
    alignment_path: NonEmpty
    snr_db: float = Field(strict=True)
    loudness_lufs: float = Field(strict=True)
    asr_cer: float = Field(ge=0, strict=True)
    qc_status: QcStatus
    split: Split
    consent_id: NonEmpty
    license: NonEmpty
    recorded_at: date

    _speaker_id = field_validator("speaker_id")(_check_speaker_id)

    @field_validator("audio_path", "alignment_path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if "\\" in value or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"đường dẫn {value!r} phải là đường dẫn tương đối kiểu POSIX, không có '..'")
        return value

    @field_validator("sample_rate")
    @classmethod
    def _sample_rate(cls, value: int, info: ValidationInfo) -> int:
        want = _config(info).sample_rate
        if value != want:
            raise ValueError(f"sample_rate {value}, cần {want}")
        return value

    @model_validator(mode="after")
    def _cross_fields(self, info: ValidationInfo) -> Self:
        cfg = _config(info)
        if not re.fullmatch(rf"{re.escape(self.speaker_id)}_\d{{{cfg.utterance_digits}}}", self.id):
            raise ValueError(f"id {self.id!r} phải có dạng {self.speaker_id}_<{cfg.utterance_digits} chữ số>")
        lo, hi = cfg.duration_s.min, cfg.duration_s.max
        if self.qc_status == "pass":
            if not lo <= self.duration_s <= hi:
                raise ValueError(
                    f"duration_s {self.duration_s} ngoài {lo:g}–{hi:g} s thì qc_status phải là review/fail"
                )
            if self.snr_db < cfg.min_snr_db:
                raise ValueError(f"snr_db {self.snr_db} < {cfg.min_snr_db:g} dB thì qc_status phải là review/fail")
            if self.asr_cer > cfg.max_asr_cer:
                raise ValueError(f"asr_cer {self.asr_cer} > {cfg.max_asr_cer:g} thì qc_status phải là review/fail")
        if self.split == "train" and self.license not in cfg.train_licenses:
            raise ValueError(
                f"license {self.license!r} không được dùng để train; cho phép: {', '.join(cfg.train_licenses)}"
            )
        tone_tokens = [m for t in self.phonemes.split() if (m := _TONE_TOKEN.fullmatch(t))]
        bad = [m.group(0) for m in tone_tokens if not 1 <= int(m.group(1)) <= 8]
        if bad:
            raise ValueError(f"phonemes có token thanh ngoài T1..T8: {bad}")
        if len(tone_tokens) != len(self.tones):
            raise ValueError(f"tones có {len(self.tones)} phần tử nhưng phonemes có {len(tone_tokens)} âm tiết")
        return self


class Speaker(_Record):
    """Một người nói trong ``speakers.json`` (đặc tả mục 4)."""

    speaker_id: NonEmpty
    voice_name: NonEmpty
    gender: NonEmpty
    age_range: NonEmpty
    dialect: Dialect
    province: NonEmpty
    grew_up_in: NonEmpty
    voice_traits: NonEmpty
    f0_baseline_hz: float = Field(gt=0)
    mic: NonEmpty
    room: NonEmpty
    consent_id: NonEmpty
    rights: list[NonEmpty] = Field(min_length=1)
    hours_recorded: float = Field(ge=0)

    _speaker_id = field_validator("speaker_id")(_check_speaker_id)


def write_manifest(utterances: Iterable[Utterance], path: Path) -> None:
    """Ghi JSONL UTF-8, một câu mỗi dòng, theo thứ tự đưa vào; cùng đầu vào thì cùng byte."""
    utts = list(utterances)
    seen: set[str] = set()
    for utt in utts:
        if utt.id in seen:
            raise ManifestError(f"id trùng khi ghi {path}: {utt.id}")
        seen.add(utt.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for utt in utts:
            f.write(json.dumps(utt.model_dump(mode="json", exclude_none=True), ensure_ascii=False) + "\n")


def read_manifest(path: Path, config: ManifestConfig | None = None) -> list[Utterance]:
    """Đọc và kiểm tra từng dòng; dòng sai, dòng trống hoặc id trùng thì ném ``ManifestError`` nêu số dòng."""
    context = {"config": config or _default_config()}
    utts: list[Utterance] = []
    first_line: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ManifestError(f"{path} dòng {lineno}: dòng trống")
        try:
            utt = Utterance.model_validate_json(line, context=context)
        except ValidationError as err:
            raise ManifestError(f"{path} dòng {lineno}: {err}") from err
        if utt.id in first_line:
            raise ManifestError(f"{path} dòng {lineno}: id {utt.id} trùng với dòng {first_line[utt.id]}")
        first_line[utt.id] = lineno
        utts.append(utt)
    return utts


def write_speakers(speakers: Iterable[Speaker], path: Path) -> None:
    rows = [s.model_dump(mode="json") for s in speakers]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_speakers(path: Path) -> list[Speaker]:
    """Đọc ``speakers.json`` (mảng JSON); mục sai hoặc speaker_id trùng thì ném ``ManifestError``."""
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ManifestError(f"{path}: không phải JSON hợp lệ: {err}") from err
    if not isinstance(rows, list):
        raise ManifestError(f"{path}: cần một mảng JSON các người nói")
    speakers: list[Speaker] = []
    for index, row in enumerate(rows, start=1):
        try:
            speaker = Speaker.model_validate(row)
        except ValidationError as err:
            raise ManifestError(f"{path} mục {index}: {err}") from err
        if any(s.speaker_id == speaker.speaker_id for s in speakers):
            raise ManifestError(f"{path} mục {index}: speaker_id {speaker.speaker_id} trùng")
        speakers.append(speaker)
    return speakers


def check_consent(utterances: Iterable[Utterance], speakers: Iterable[Speaker]) -> None:
    """Mỗi câu phải có người nói trong speakers.json, cùng consent_id, dialect, province (review P4)."""
    by_id = {s.speaker_id: s for s in speakers}
    problems: list[str] = []
    for utt in utterances:
        speaker = by_id.get(utt.speaker_id)
        if speaker is None:
            problems.append(f"{utt.id}: người nói {utt.speaker_id} không có trong speakers.json")
            continue
        for name in ("consent_id", "dialect", "province"):
            if getattr(utt, name) != getattr(speaker, name):
                problems.append(
                    f"{utt.id}: {name} {getattr(utt, name)!r} khác speakers.json {getattr(speaker, name)!r}"
                )
    if problems:
        raise ValueError(f"{len(problems)} câu không khớp người nói / hợp đồng:\n" + "\n".join(problems))


@dataclass(frozen=True)
class SplitResult:
    utterances: tuple[Utterance, ...]  # đã gán split, sắp theo id
    dropped_duplicates: tuple[str, ...]  # id câu bị loại vì trùng văn bản của cùng người nói


def split_utterances(
    utterances: Sequence[Utterance], test_texts: Iterable[str], config: ManifestConfig | None = None
) -> SplitResult:
    """Gán train/val/test theo tỉ lệ trong ``manifest.yaml`` (bước 12, đặc tả mục 6).

    - Câu trùng văn bản của cùng một người nói: giữ một câu (ưu tiên ``pass``, rồi id nhỏ nhất).
    - Câu chung text_raw (cùng dòng kịch bản, kể cả khi text_norm khác theo vùng) hoặc chung text_norm
      luôn vào cùng một tập, để văn bản không rò giữa các tập.
    - Văn bản thuộc ``test_texts`` (so trên text_raw và text_norm, không phân biệt hoa thường,
      khoảng trắng) luôn vào ``test``, không bao giờ vào ``train``.
    - Trong mỗi nhóm (speaker_id, style), câu được xáo theo seed rồi rót vào tập còn thiếu nhiều
      nhất so với tỉ lệ, nên mỗi nhóm đều xấp xỉ 94/3/3. Kết quả không phụ thuộc thứ tự đầu vào.
    - Câu có giấy phép không cho train thì ném lỗi: cần lọc riêng trước khi chia tập.
    """
    cfg = config or _default_config()
    if not utterances:
        raise ValueError("danh sách câu rỗng, không có gì để chia tập")
    ids = [u.id for u in utterances]
    if len(set(ids)) != len(ids):
        raise ValueError(f"id trùng trong đầu vào: {sorted({i for i in ids if ids.count(i) > 1})}")
    blocked = sorted(u.id for u in utterances if u.license not in cfg.train_licenses)
    if blocked:
        raise ValueError(f"{len(blocked)} câu có giấy phép không cho train, cần lọc trước khi chia tập: {blocked}")

    kept, dropped = _drop_duplicates(utterances)
    test_keys = {_text_key(t) for t in test_texts}
    # Nhóm văn bản: câu chung text_raw (cùng dòng kịch bản, dù chuẩn hoá khác theo vùng) hoặc
    # chung text_norm thì cùng nhóm, và cả nhóm vào cùng một tập. Nhóm chứa câu test thì vào test.
    group = _text_groups(kept)
    by_group: dict[str, Split] = {}
    for utt in kept:
        if _text_key(utt.text_norm) in test_keys or _text_key(utt.text_raw) in test_keys:
            by_group[group[utt.id]] = "test"
    ratios = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}

    strata: dict[tuple[str, str], list[Utterance]] = {}
    for utt in kept:
        strata.setdefault((utt.speaker_id, utt.style), []).append(utt)

    assigned: dict[str, Split] = {}
    for (speaker, style), members in sorted(strata.items()):
        rng = random.Random(_derived_seed(cfg.split.seed, speaker, style))
        order = sorted(members, key=lambda u: u.id)
        rng.shuffle(order)
        counts = dict.fromkeys(SPLITS, 0)
        free: list[Utterance] = []
        for utt in order:
            forced = by_group.get(group[utt.id])
            if forced is None:
                free.append(utt)
            else:
                assigned[utt.id] = forced
                counts[forced] += 1
        targets = {s: ratios[s] * len(members) for s in SPLITS}
        for utt in free:
            split = by_group.get(group[utt.id]) or max(SPLITS, key=lambda s: targets[s] - counts[s])
            assigned[utt.id] = by_group[group[utt.id]] = split
            counts[split] += 1

    context = {"config": cfg}
    result = tuple(
        Utterance.model_validate({**u.model_dump(), "split": assigned[u.id]}, context=context)
        for u in sorted(kept, key=lambda u: u.id)
    )
    return SplitResult(utterances=result, dropped_duplicates=tuple(sorted(dropped)))


def _drop_duplicates(utterances: Iterable[Utterance]) -> tuple[list[Utterance], list[str]]:
    groups: dict[tuple[str, str], list[Utterance]] = {}
    for utt in utterances:
        groups.setdefault((utt.speaker_id, _text_key(utt.text_norm)), []).append(utt)
    kept, dropped = [], []
    for members in groups.values():
        best, *rest = sorted(members, key=lambda u: (u.qc_status != "pass", u.id))
        kept.append(best)
        dropped += [u.id for u in rest]
    return kept, dropped


def _text_groups(utterances: Sequence[Utterance]) -> dict[str, str]:
    """id câu -> khoá nhóm; hai câu chung text_raw hoặc chung text_norm (sau chuẩn hoá khoá) thì cùng nhóm."""
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        while parent.setdefault(key, key) != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for utt in utterances:
        a, b = find("raw:" + _text_key(utt.text_raw)), find("norm:" + _text_key(utt.text_norm))
        if a != b:
            parent[max(a, b)] = min(a, b)  # gốc là khoá nhỏ nhất: không phụ thuộc thứ tự đầu vào
    return {u.id: find("raw:" + _text_key(u.text_raw)) for u in utterances}


def _text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _derived_seed(seed: int, *parts: str) -> int:
    # Không dùng hash() của Python: bị xáo ngẫu nhiên giữa các lần chạy.
    digest = hashlib.sha256("|".join([str(seed), *parts]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
