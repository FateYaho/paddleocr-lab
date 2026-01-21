import os
import tempfile
import shutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
from PIL import Image

os.environ.setdefault("HOME", "/app")

app = FastAPI()

# PaddleOCR 엔진 설정 최적화
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="korean",
    # 서식 문서의 경우 박스 검출 임계값을 약간 높여 노이즈를 방지합니다.
    det_db_thresh=0.3,         
    det_db_box_thresh=0.5,     
    det_db_unclip_ratio=1.6,   
    show_log=False
)

def preprocess_fax_image(image_path):
    """
    OpenCV를 사용한 고성능 전처리: 워터마크 제거 및 대비 강화
    """
    # 1. 이미지 읽기 및 그레이스케일 변환
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 노이즈 제거 (비등방성 확산 필터와 유사한 효과)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 3. 적응형 이진화 (Adaptive Thresholding)
    # 배경의 워터마크를 날리고 글자 획을 뚜렷하게 만듭니다.
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 8
    )
    
    # 4. 결과 저장
    processed_path = image_path.replace(".", "_processed.")
    cv2.imwrite(processed_path, binary)
    return processed_path

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
            content = await file.read()
            f.write(content)
        
        # 전처리 수행 여부 결정
        target_path = original_path
        if preprocess:
            target_path = preprocess_fax_image(original_path)
        
        # OCR 실행
        result = ocr_engine.ocr(target_path)
        
        texts = []
        if result and result[0]:
            # 추출된 텍스트들을 Y좌표(행) 기준으로 정렬하여 읽기 순서를 보정합니다.
            # line[0]은 박스 좌표 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            sorted_lines = sorted(result[0], key=lambda x: x[0][0][1])
            
            for line in sorted_lines:
                texts.append({
                    "text": line[1][0],
                    "confidence": float(line[1][1]), # JSON 직렬화를 위해 float 변환
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
