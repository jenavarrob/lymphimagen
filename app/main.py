from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .analyzer import AnalysisError, ImageAnalyzer
from .models import CoinMeasureResponse, CompareResponse, InteractiveCompareResponse

app = FastAPI(
    title="LymphImagen PoC",
    version="0.1.0",
    description="Explainable image-based limb comparison and coin calibration",
)
analyzer = ImageAnalyzer()
HOME_TEMPLATE = Path(__file__).parent / "templates" / "index.html"


async def _read_image(image: UploadFile) -> bytes:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Image upload is empty")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller")
    return data


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HOME_TEMPLATE.read_text(encoding="utf-8")


@app.post("/api/v1/measure/compare", response_model=CompareResponse)
async def compare(image: UploadFile = File(...)) -> CompareResponse:
    try:
        return analyzer.compare(await _read_image(image))
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/measure/interactive-compare", response_model=InteractiveCompareResponse)
async def interactive_compare(image: UploadFile = File(...)) -> InteractiveCompareResponse:
    try:
        return analyzer.interactive_compare(await _read_image(image))
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/measure/coin", response_model=CoinMeasureResponse)
async def coin_measure(image: UploadFile = File(...)) -> CoinMeasureResponse:
    try:
        return analyzer.coin_measure(await _read_image(image))
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
