import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR

os.environ.setdefault("HOME", "/app")

app = FastAPI()

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="korean",
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "input" + suffix)

    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        result = ocr.ocr(path, cls=True)
        return {"result": result}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
