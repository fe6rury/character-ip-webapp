"""
캐릭터 IP 유사도 판별 엔진 — FastAPI 서버

실행:
    uvicorn app:app --reload
    → http://127.0.0.1:8000        웹 화면
    → http://127.0.0.1:8000/docs   API 문서 (자동 생성)
"""

import io
import json
import math
from threading import Lock
from urllib.parse import quote
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModel

# ────────────────────────────── 설정 ──────────────────────────────

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OG_DIR = DATA / "og"

MODEL_ID = "facebook/dinov2-base"
IMG_SIZE = 224
HEAD_PATH = DATA / "character_head.npz"

YOLO_PATH = DATA / "character_print.pt"
INFERENCE_LOCK = Lock()
MAX_UPLOAD = 20 * 1024 * 1024
S: dict = {}          # 모델·임베딩을 담아두는 전역 상태


# ────────────────────────────── 전처리 ──────────────────────────────

def preprocess(im: Image.Image) -> np.ndarray:
    """알파→흰색 합성, 비율 유지 정사각 패딩, 224 리사이즈, 정규화.

    코랩에서 갤러리 임베딩을 뽑을 때와 **똑같은** 전처리여야 한다.
    하나라도 다르면 유사도가 통째로 틀어진다.
    """
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im)
    im = im.convert("RGB")

    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))

    arr = np.asarray(canvas.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.float32) / 255.0
    return ((arr - S["mean"]) / S["std"]).transpose(2, 0, 1)


def embed(im: Image.Image) -> np.ndarray:
    """이미지 1장 → L2 정규화된 768차원 벡터 (DINOv2 CLS 토큰)."""
    x = torch.from_numpy(preprocess(im)[None]).to(S["device"])
    with torch.no_grad():
        v = S["model"](pixel_values=x).last_hidden_state[:, 0]
    v = torch.nn.functional.normalize(v.float(), dim=1)
    return v.cpu().numpy()[0]


# ────────────────────────── 색상 특징 (형태와 별개) ──────────────────────────
#
# DINOv2는 색을 거의 무시한다 (색상 변형 유사도 0.989). 그래서 색은 따로 잰다.
# 판정 점수에는 섞지 않는다 — 색만 바꾼 도용을 놓치게 되기 때문. 검수자가
# "형태는 같은데 색을 바꿨다"를 읽을 수 있도록 나란히 보여주기만 한다.

HIST_BINS = (24, 8, 8)      # H, S, V
HIST_SMOOTH = 2             # 미세한 색 차이로 칸이 어긋나는 것을 완화


def _smooth(h3: np.ndarray, times: int = HIST_SMOOTH) -> np.ndarray:
    """히스토그램을 이웃 칸으로 번지게 한다.

    칸을 그대로 비교하면 주황과 살짝 다른 주황이 서로 다른 칸에 떨어져
    유사도가 0에 가까워진다. H축은 순환(빨강↔보라가 이웃), S·V축은 끝을 복제.
    """
    k = np.array([0.25, 0.5, 0.25], np.float32)
    for _ in range(times):
        h3 = np.roll(h3, 1, 0) * k[0] + h3 * k[1] + np.roll(h3, -1, 0) * k[2]
        for ax in (1, 2):
            pad = np.concatenate(
                [np.take(h3, [0], ax), h3, np.take(h3, [-1], ax)], axis=ax)
            n = h3.shape[ax]
            h3 = (np.take(pad, range(0, n), ax) * k[0]
                  + np.take(pad, range(1, n + 1), ax) * k[1]
                  + np.take(pad, range(2, n + 2), ax) * k[2])
    return h3


def color_hist(im: Image.Image) -> np.ndarray:
    """전경(배경 제외) 픽셀의 HSV 히스토그램. 합이 1인 벡터."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im)
    rgb = im.convert("RGB")
    rgb.thumbnail((256, 256), Image.BICUBIC)

    arr = np.asarray(rgb, np.float32)
    gray = arr @ np.array([0.299, 0.587, 0.114], np.float32)
    mask = gray < 245                       # 흰 배경 제외
    if mask.sum() < 50:                     # 전경을 못 찾으면 전체 사용
        mask = np.ones(gray.shape, bool)

    hsv = np.asarray(rgb.convert("HSV"))[mask]      # (N, 3)
    h, _ = np.histogramdd(hsv.astype(np.float32), bins=HIST_BINS,
                          range=((0, 256), (0, 256), (0, 256)))
    h = _smooth(h).ravel()
    return (h / (h.sum() + 1e-9)).astype(np.float32)


def color_similarity(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """바타차리야 계수. 0(전혀 다름) ~ 1(동일)."""
    return np.sqrt(h1 * h2).sum(axis=-1)


# ────────────────────────────── 기동 / 종료 ──────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    emb_path, meta_path = DATA / "og_emb.npy", DATA / "og_meta.csv"
    if not emb_path.exists() or not meta_path.exists():
        raise RuntimeError(
            f"등록 데이터가 없습니다.\n"
            f"  {emb_path}\n  {meta_path}\n"
            f"코랩의 '내보내기 셀'로 만든 zip을 data/ 에 풀어주세요."
        )

    cached_model = DATA / "dinov2-base"
    has_cache = all((cached_model / name).is_file() for name in (
        "config.json", "preprocessor_config.json", "model.safetensors",
    ))
    model_source = str(cached_model) if has_cache else MODEL_ID
    print(f"[1/4] DINOv2 로드 중… ({model_source})")
    proc = AutoImageProcessor.from_pretrained(model_source)
    S["mean"] = np.array(proc.image_mean, np.float32)
    S["std"] = np.array(proc.image_std, np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    S["model"] = AutoModel.from_pretrained(model_source).to(device).eval()
    S["device"] = device
    print(f"[2/4] 디바이스: {device}")

    S["emb"] = np.load(emb_path).astype(np.float32)
    S["meta"] = pd.read_csv(meta_path)
    n_char = S["meta"]["character"].nunique()
    if n_char != 10 or S["emb"].shape != (len(S["meta"]), 768):
        raise RuntimeError("10종 원본 목록과 DINOv2 임베딩이 일치하는지 확인하세요.")
    if not HEAD_PATH.is_file():
        raise RuntimeError(f"전체 팀 데이터 학습 가중치가 없습니다: {HEAD_PATH}")
    with np.load(HEAD_PATH, allow_pickle=False) as trained:
        classes = tuple(f"c_{code.lower()}" for code in trained["classes"].tolist())
        weights = trained["weights"].astype(np.float32)
        bias = trained["bias"].astype(np.float32)
        threshold = float(trained["review_threshold"])
    if (len(classes) != 10 or set(classes) != set(S["meta"]["character"])
            or weights.shape != (768, 10) or bias.shape != (10,)
            or not np.isfinite(weights).all() or not np.isfinite(bias).all()
            or not 0 < threshold < 1):
        raise RuntimeError("전체 학습 가중치의 10종 목록 또는 설정이 유효하지 않습니다.")
    training_report = json.loads((BASE / "전체학습_결과.json").read_text(encoding="utf-8"))
    training_rows = int(training_report["training_image_rows"])
    if training_rows != sum(training_report["source_groups"].values()) or training_rows <= 0:
        raise RuntimeError("전체 학습 결과 보고서와 학습 파일을 확인하세요.")
    S.update(training_rows=training_rows)
    S.update(head_columns={name: i for i,name in enumerate(classes)}, head_weights=weights,
             head_bias=bias, review_threshold=threshold)
    if not YOLO_PATH.is_file():
        raise RuntimeError(f"제출용 YOLO 가중치가 없습니다: {YOLO_PATH}")
    from ultralytics import YOLO
    S["detector"] = YOLO(str(YOLO_PATH))
    evaluation = BASE / "evaluation.json"
    S["detector_confidence"] = (
        float(json.loads(evaluation.read_text(encoding="utf-8")).get("deployment_confidence", 0.20))
        if evaluation.is_file() else 0.20
    )
    if not 0 < S["detector_confidence"] <= 1:
        raise RuntimeError("YOLO 검출 기준값을 확인하세요.")
    if list(S["detector"].names.values()) != ["character_print"]:
        raise RuntimeError("캐릭터 인쇄 전용 YOLO 가중치가 필요합니다.")
    print(f"[3/4] 등록 원본 {len(S['meta'])}장 / 캐릭터 {n_char}종")

    histograms = []
    for name in S["meta"]["file"]:
        with Image.open(OG_DIR / name) as original:
            histograms.append(color_hist(original))
    S["chist"] = np.stack(histograms)
    print(f"[4/4] 색상 히스토그램 {S['chist'].shape} — 준비 완료\n")
    print("    웹 화면  http://127.0.0.1:8000")
    print("    API 문서 http://127.0.0.1:8000/docs\n")

    try:
        yield
    finally:
        S.clear()


app = FastAPI(
    title="캐릭터 IP 유사도 판별 엔진",
    version="2.0.0",
    description=(
        "상품 이미지를 입력하면 등록된 공공 마스코트와의 유사도를 계산해 "
        "YOLO로 인쇄 영역을 찾고 사람이 검토할 유사 후보를 제시합니다. DINOv2-base로 임베딩을 추출하고 "
        "L2 정규화 후 코사인 유사도로 비교합니다."
    ),
    lifespan=lifespan,
)


# ────────────────────────────── 응답 스키마 ──────────────────────────────

class CharacterScore(BaseModel):
    rank: int
    character: str
    character_id: str
    recognition_score: float
    max_similarity: float     # DINOv2 원본 형태 유사도
    top3_mean: float
    mean: float
    color_similarity: float   # 색상 유사도 (참고용, 판정에 쓰지 않음)
    color_note: str
    best_origin: str          # 가장 닮은 원본의 파일명
    best_origin_url: str      # 그 썸네일 주소
    flagged: bool             # 임계값 이상이면 True
    verdict: str
    comparison_source: str
    detection_index: Optional[int] = None


class Detection(BaseModel):
    box: list[float]
    confidence: float


class IdentifyResponse(BaseModel):
    filename: Optional[str]
    region: Optional[list[float]] = None
    threshold: float
    flagged: bool             # 하나라도 걸렸는가
    summary: str
    detections: list[Detection]
    results: List[CharacterScore]


# ────────────────────────────── 엔드포인트 ──────────────────────────────

@app.get("/health", summary="서버 상태")
def health():
    return {
        "status": "ok",
        "device": S.get("device"),
        "registered_images": int(len(S["meta"])) if "meta" in S else 0,
        "characters": int(S["meta"]["char_name"].nunique()) if "meta" in S else 0,
        "threshold": S.get("review_threshold"),
        "yolo_ready": "detector" in S,
        "trained_recognition_ready": "head_weights" in S,
        "training_image_rows": S.get("training_rows"),
        "yolo_confidence": S.get("detector_confidence"),
    }


@app.get("/api/characters", summary="등록된 캐릭터 목록")
def characters():
    m = S["meta"]
    out = []
    for cid, g in m.groupby("character"):
        out.append({
            "character_id": cid,
            "character": g["char_name"].iloc[0],
            "n_origins": int(len(g)),
            "thumbnails": [f"/og/{quote(f)}" for f in g["file"].tolist()],
        })
    return out


@app.post("/api/identify", response_model=IdentifyResponse, summary="유사도 판정")
async def identify(
    file: UploadFile = File(..., description="검사할 이미지"),
    region: Optional[str] = Form(None, max_length=200),
):
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "20 MB 이하 이미지를 업로드하세요.")
    if not raw:
        raise HTTPException(400, "빈 파일입니다.")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            im = ImageOps.exif_transpose(source)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "이미지로 읽을 수 없는 파일입니다.") from exc
    selection = None
    if region is not None:
        try:
            selection = [float(v) for v in (region or "").split(",")]
            if len(selection) != 4 or not all(math.isfinite(v) for v in selection):
                raise ValueError
            left, top, right, bottom = selection
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError
        except ValueError:
            raise HTTPException(422, "사진에서 분석할 영역을 드래그해 선택해 주세요.") from None
        w, h = im.size
        box = (math.floor(left*w), math.floor(top*h), math.ceil(right*w), math.ceil(bottom*h))
        if box[2]-box[0] < 8 or box[3]-box[1] < 8:
            raise HTTPException(422, "가로와 세로가 각각 8픽셀 이상인 영역을 선택해 주세요.")
        im = im.crop(box)
    result = await run_in_threadpool(compare_image, im, file.filename)
    result["region"] = selection
    return result


def compare_image(im: Image.Image, filename: Optional[str]):
    detections = []
    candidates = [im]
    w, h = im.size
    with INFERENCE_LOCK:
        rgb = im.convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), rgb).convert("RGB")
        result = S["detector"].predict(
            rgb, imgsz=640, conf=S["detector_confidence"], max_det=5, device=S["device"], verbose=False,
        )[0]
        for xyxy, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = xyxy
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            margin = 0.10 * max(x2 - x1, y2 - y1)
            candidates.append(im.crop((
                max(0, int(x1 - margin)), max(0, int(y1 - margin)),
                min(w, int(x2 + margin + 1)), min(h, int(y2 + margin + 1)),
            )))
            detections.append({
                "box": [round(x1 / w, 6), round(y1 / h, 6), round(x2 / w, 6), round(y2 / h, 6)],
                "confidence": round(float(score), 4),
            })
        vectors = np.stack([embed(candidate) for candidate in candidates])
    similarities = vectors @ S["emb"].T
    logits = np.clip(vectors @ S["head_weights"] + S["head_bias"], -40, 40)
    recognition = 1 / (1 + np.exp(-logits))
    threshold = S["review_threshold"]
    hists = [color_hist(candidate) for candidate in candidates]
    meta = S["meta"]

    rows = []
    for cid, g in meta.groupby("character"):
        idx = g.index.to_numpy()
        column = S["head_columns"][cid]
        selected = int(recognition[:, column].argmax())
        score = float(recognition[selected, column])
        s = similarities[selected, idx]
        k = int(np.argmax(s))
        best = g.iloc[k]
        top3 = float(np.sort(s)[-3:].mean()) if len(s) >= 3 else float(s.mean())
        mx = float(s.max())
        cs = float(color_similarity(S["chist"][idx[k]], hists[selected]))           # 형태가 가장 닮은 그 원본과의 색상 유사도

        if score >= threshold and cs < 0.45:
            note = "판별 기준 이상 · 색상 차이 있음"
        elif score >= threshold:
            note = "판별 기준 이상 · 색상 차이 작음"
        elif cs >= 0.7:
            note = "색상만 유사"
        else:
            note = "—"

        rows.append({
            "character": str(best["char_name"]),
            "character_id": str(cid),
            "recognition_score": score,
            "max_similarity": round(mx, 4),
            "top3_mean": round(top3, 4),
            "mean": round(float(s.mean()), 4),
            "color_similarity": round(cs, 4),
            "color_note": note,
            "best_origin": str(best["file"]),
            "best_origin_url": f"/og/{quote(best['file'])}",
            "flagged": score >= threshold,
            "verdict": "검수 필요" if score >= threshold else "기준 미만",
            "comparison_source": "검출 영역" if selected else "전체 이미지",
            "detection_index": selected - 1 if selected else None,
        })

    rows.sort(key=lambda r: -r["recognition_score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["recognition_score"] = round(r["recognition_score"], 4)

    hit = [r for r in rows if r["flagged"]]
    if hit:
        names = ", ".join(r["character"] for r in hit)
        summary = f"{names} 와(과) 유사합니다. 사람의 검수가 필요합니다."
    else:
        top = rows[0]
        summary = (f"검수 기준을 넘은 후보가 없습니다. "
                   f"최고 판별 점수 {top['recognition_score']:.4f} ({top['character']}), "
                   f"기준 {threshold:.3f}.")

    return {
        "filename": filename,
        "threshold": threshold,
        "flagged": bool(hit),
        "summary": summary,
        "detections": detections,
        "results": rows,
    }


# ────────────────────────────── 정적 파일 ──────────────────────────────

if OG_DIR.is_dir():
    app.mount("/og", StaticFiles(directory=str(OG_DIR)), name="og")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE / "static" / "index.html")
