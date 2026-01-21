import os
import tempfile
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR

os.environ.setdefault("HOME", "/app")

app = FastAPI()

# 1. 엔진 파라미터 재조정 (가장 중요)
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="korean",
    det_db_thresh=0.4,        # 노이즈 차단을 위해 더 높임
    det_db_box_thresh=0.6,    # 확실한 글자만 박스로 잡음
    det_db_unclip_ratio=1.5,  # 박스를 너무 키우지 않음 (워터마크 간섭 방지)
    rec_batch_num=1,          # 정확도를 위해 배치 사이즈 축소
    show_log=False
)

def final_preprocess(image_path):
    # 이미지를 읽어서 워터마크만 흐리게 만듦
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # [핵심] 적응형 히스토그램 평활화 (글자만 진하게 만듦)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 이진화를 하지 않고, 대신 노이즈만 살짝 제거 (워터마크 무시 전략)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    processed_path = image_path.replace(".", "_final.")
    cv2.imwrite(processed_path, denoised)
    return processed_path

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".png"
        path = os.path.join(tmp_dir, "input" + suffix)
        
        with open(path, "wb") as f:
            f.write(await file.read())
        
        # 새로운 전처리 적용
        target_path = final_preprocess(path)
        
        result = ocr_engine.ocr(target_path)
        
        texts = []
        if result and result[0]:
            # 줄바꿈 정렬 로직 강화 (Y좌표 오차범위 10px 허용)
            result[0].sort(key=lambda x: (x[0][0][1] // 10, x[0][0][0]))
            
            for line in result[0]:
                texts.append({
                    "text": line[1][0],
                    "conf": float(line[1][1])
                })
        
        full_text = "\n".join([t["text"] for t in texts])
        return {"full_text": full_text}
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
