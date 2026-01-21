import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageFilter

os.environ.setdefault("HOME", "/app")

app = FastAPI()

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="korean",
    det_db_thresh=0.2,
    det_db_box_thresh=0.4,
    det_db_unclip_ratio=1.8,
    rec_batch_num=8,
)

def preprocess_fax_image(img):
    """팩스/스캔 문서 전처리"""
    img = img.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)
    img = img.filter(ImageFilter.SHARPEN)
    return img.convert("RGB")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...), preprocess: bool = True):
    tmp_dir = tempfile.mkdtemp()
    
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".png"
        original_path = os.path.join(tmp_dir, "input" + suffix)
        
        with open(original_path, "wb") as f:
            f.write(await file.read())
        
        img = Image.open(original_path).convert("RGB")
        
        if preprocess:
            img = preprocess_fax_image(img)
        
        ocr_path = os.path.join(tmp_dir, "processed.png")
        img.save(ocr_path, "PNG")
        
        result = ocr_engine.ocr(ocr_path)
        
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append({
                    "text": line[1][0],
                    "confidence": line[1][1],
                    "box": line[0]
                })
        
        return {
            "full_text": "\n".join([t["text"] for t in texts]),
            "details": texts
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
