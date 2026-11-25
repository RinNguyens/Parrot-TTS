"""Đọc file cấu hình YAML cạnh module và kiểm tra bằng Pydantic (sai khoá, sai kiểu thì ném lỗi)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_DIR = Path(__file__).parent


class FrozenModel(BaseModel):
    """Cấu hình bất biến, cấm khoá lạ để gõ sai tên khoá không bị bỏ qua im lặng."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def load_yaml_model[M: BaseModel](model: type[M], name: str, path: Path | None = None) -> M:
    """Đọc ``path`` (mặc định ``<name>.yaml`` trong thư mục này) thành ``model``."""
    path = path or CONFIG_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return model.model_validate(raw)
    except ValidationError as err:
        raise ValueError(f"{path.name} sai cấu trúc: {err}") from err


# --- qa.yaml: dùng chung cho ingest (bước 1), session_qa (P4) và loudness (giới hạn đỉnh) ---


class FormatSpec(FrozenModel):
    containers: tuple[str, ...] = Field(min_length=1)
    sample_rate: int = Field(gt=0)
    subtype: str
    channels: int = Field(ge=1)


class ClippingSpec(FrozenModel):
    lsb_tolerance: int = Field(ge=0)
    min_run: int = Field(ge=1)


class IngestConfig(FrozenModel):
    format: FormatSpec
    clipping: ClippingSpec


class QAThresholds(FrozenModel):
    min_snr_db: float
    max_peak_dbfs: float
    max_noise_floor_dbfs: float


class LevelAnalysis(FrozenModel):
    frame_ms: float = Field(gt=0)
    noise_percentile: float = Field(gt=0, le=100)
    speech_margin_db: float = Field(gt=0)
    db_floor: float
    max_zero_run_ms: float = Field(gt=0)


class QAConfig(FrozenModel):
    ingest: IngestConfig
    qa: QAThresholds
    analysis: LevelAnalysis


def load_qa_config(path: Path | None = None) -> QAConfig:
    return load_yaml_model(QAConfig, "qa", path)
