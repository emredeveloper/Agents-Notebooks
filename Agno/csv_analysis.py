from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import csv
import math
import os
from pathlib import Path

from pydantic import BaseModel, Field
from rich.pretty import pprint

from agno.agent import Agent
from agno.models.openrouter import OpenRouter


# ------------------------------
# 1) Basit CSV profil çıkarıcı
# ------------------------------

@dataclass
class ColumnStats:
    name: str
    non_null_count: int
    null_count: int
    inferred_type: str
    sample_values: List[str]
    unique_count: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None


def try_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def infer_type(values: List[str]) -> str:
    numeric_seen = 0
    text_seen = 0
    for v in values:
        if v == "" or v.lower() == "none" or v.lower() == "null":
            continue
        if try_float(v) is not None:
            numeric_seen += 1
        else:
            text_seen += 1
    if numeric_seen > 0 and text_seen == 0:
        return "numeric"
    if text_seen > 0 and numeric_seen == 0:
        return "text"
    if numeric_seen > 0 and text_seen > 0:
        return "mixed"
    return "unknown"


def profile_csv(file_path: str, max_rows: int = 5000) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV bulunamadı: {file_path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        column_values: Dict[str, List[str]] = {c: [] for c in fieldnames}

        total_rows = 0
        for row in reader:
            for c in fieldnames:
                column_values[c].append((row.get(c) or "").strip())
            total_rows += 1
            if total_rows >= max_rows:
                break

    columns: List[ColumnStats] = []
    for c in fieldnames:
        values = column_values[c]
        null_count = sum(1 for v in values if v == "" or v.lower() in {"none", "null"})
        non_null = len(values) - null_count
        inferred = infer_type(values)

        sample_values = [v for v in values if v][:10]
        stats = ColumnStats(
            name=c,
            non_null_count=non_null,
            null_count=null_count,
            inferred_type=inferred,
            sample_values=sample_values,
        )

        if inferred == "numeric":
            nums = [try_float(v) for v in values if v not in {"", "none", "null"}]
            nums = [n for n in nums if n is not None and not math.isnan(n)]
            if nums:
                stats.min_value = min(nums)
                stats.max_value = max(nums)
                stats.mean = sum(nums) / len(nums)
        else:
            # Kategorik/Metin: eşsiz sayısı mantıklı
            counter = Counter([v for v in values if v])
            stats.unique_count = len(counter)

        columns.append(stats)

    profile: Dict[str, Any] = {
        "file": str(path),
        "num_rows": total_rows,
        "num_columns": len(fieldnames),
        "columns": [vars(c) for c in columns],
        "note": "Bu profil ilk max_rows satır üzerinden çıkarılmıştır.",
    }
    return profile


# --------------------------------------
# 2) Yapılandırılmış çıktı için şema
# --------------------------------------

class ColumnAssessment(BaseModel):
    name: str = Field(..., description="Sütun adı")
    role: str = Field(..., description="Analizde önerilen rol (id, kategori, metrik, tarih vb.)")
    issues: List[str] = Field(default_factory=list, description="Saptanan kalite sorunları")
    actions: List[str] = Field(default_factory=list, description="Önerilen eylemler")


class CsvAnalysisReport(BaseModel):
    dataset_title: str = Field(..., description="Veri kümesi için kısa bir başlık")
    summary: str = Field(..., description="Genel özet ve içgörüler")
    key_metrics: List[str] = Field(default_factory=list, description="Öne çıkan metrikler")
    column_assessments: List[ColumnAssessment] = Field(default_factory=list, description="Sütun bazlı değerlendirmeler")
    data_quality_issues: List[str] = Field(default_factory=list, description="Genel veri kalitesi problemleri")
    recommendations: List[str] = Field(default_factory=list, description="Önceliklendirilmiş öneriler")


# --------------------------------------
# 3) Ajan kurulumu (structured output)
# --------------------------------------

def build_agent() -> Agent:
    api_key = "sk-or-v1-"
    model_id = "x-ai/grok-4-fast:free"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY bulunamadı. .env veya ortam değişkeni olarak ekleyin.")
    return Agent(
        model=OpenRouter(id=model_id, api_key=api_key),
        description="CSV verisi için veri profili üzerinden yapılandırılmış analiz raporu yazarsın.",
        output_schema=CsvAnalysisReport,
        markdown=True,
        instructions=(
            "Sana verilecek CSV profilini dikkatle incele. Veri kalitesi sorunlarını belirt, sütun rollerini öner, "
            "net ve uygulanabilir öneriler üret. Yalnızca şemada tanımlı alanları doldur."
        ),
    )


# --------------------------------------
# 4) Uçtan uca akış
# --------------------------------------

def run_csv_analysis(csv_path: str, max_rows: int = 5000) -> None:
    print("[1/4] CSV profili çıkarılıyor...")
    profile = profile_csv(csv_path, max_rows=max_rows)
    pprint({"profil_ozet": {k: v for k, v in profile.items() if k in {"file", "num_rows", "num_columns"}}})

    print("\n[2/4] Ajan hazırlanıyor...")
    agent = build_agent()

    print("\n[3/4] Ajan ile yapılandırılmış analiz alınıyor (akış)...\n")
    prompt = (
        "Aşağıda bir CSV veri profili var. Bu profile dayanarak, işlenebilir içgörüleri olan kısa ama kapsamlı "
        "bir analiz raporu hazırla. Eğer veri kalitesi sorunları tespit edersen açıkça yaz ve çözüm öner.")
    agent.print_response(
        f"{prompt}\n\nCSV_PROFIL_JSON=\n{profile}",
        stream=True,
    )

    print("\n[4/4] İşlem tamamlandı.")


if __name__ == "__main__":
    # Örnek kullanım: küçük bir CSV yolu verin
    # Örn: python Agno/csv_analysis.py
    sample_csv = os.environ.get("CSV_PATH", "monthly-car-sales.csv")
    try:
        run_csv_analysis(sample_csv, max_rows=5000)
    except Exception as e:
        print(f"Hata: {e}")

