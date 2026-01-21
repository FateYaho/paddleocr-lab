import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
from PIL import Image

os.environ.setdefault("HOME", "/app")

app = FastAPI()

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="korean",
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    
    try:
        # 원본 파일 저장
        suffix = os.path.splitext(file.filename or "")[1] or ".png"
        original_path = os.path.join(tmp_dir, "input" + suffix)
        
        with open(original_path, "wb") as f:
            f.write(await file.read())
        
        # TIFF 등 지원 안되는 포맷은 PNG로 변환
        if suffix.lower() in [".tif", ".tiff", ".bmp", ".webp"]:
            img = Image.open(original_path).convert("RGB")
            converted_path = os.path.join(tmp_dir, "input.png")
            img.save(converted_path, "PNG")
            ocr_path = converted_path
        else:
            ocr_path = original_path
        
        result = ocr_engine.ocr(ocr_path, cls=True)
        return {"result": result}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
