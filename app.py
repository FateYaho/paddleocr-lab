import os
import tempfile
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR

os.environ.setdefault("HOME", "/app")

app = FastAPI()

# 서식 문서 최적화 설정
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="korean",
    rec_model_dir=None, # 기본 모델 사용
    det_db_thresh=0.4,   # 임계값을 더 높여 노이즈 차단
    det_db_box_thresh=0.6,
    det_db_unclip_ratio=1.5,
    show_log=False
)

def advanced_preprocess(image_path):
    # 1. 이미지 읽기 및 그레이스케일
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 대비 극대화 (CLAHE 적용)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. 가우시안 블러로 미세 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 4. 적응형 이진화 (Otsu와 조합)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. 모폴로지 연산 (글자 획을 선명하게 하고 워터마크 잔상 제거)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    processed_path = image_path.replace(".", "_adv_processed.")
    cv2.imwrite(processed_path, processed)
    return processed_path

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...), preprocess: bool = True):
    tmp_dir = tempfile.mkdtemp()
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".png"
        original_path = os.path.join(tmp_dir, "input" + suffix)
        
        with open(original_path, "wb") as f:
            f.write(await file.read())
        
        target_path = original_path
        if preprocess:
            # 더 강력한 전처리 호출
            target_path = advanced_preprocess(original_path)
        
        result = ocr_engine.ocr(target_path)
        
        texts = []
        if result and result[0]:
            # Y좌표 우선, X좌표 차선 정렬로 표 읽기 순서 최적화
            sorted_lines = sorted(result[0], key=lambda x: (x[0][0][1], x[0][0][0]))
            
            for line in sorted_lines:
                texts.append({
                    "text": line[1][0],
                    "confidence": float(line[1][1]),
                    "box": line[0]
                })
        
        return {
            "full_text": "\n".join([t["text"] for t in texts]),
            "details": texts
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
