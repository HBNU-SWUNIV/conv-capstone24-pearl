import os
import cv2
import pickle
import easyocr
import glob

def extract_text(reader, img_path,min_confidence=0):
    # 이미지 읽기 및 RGB 변환
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OCR 텍스트 추출
    ocr_results = reader.readtext(img_rgb)
    # 텍스트 값만 추출
    texts = [res[1] for res in ocr_results]
    return texts

def save_ocr_text_index(image_dir, output_file='ocr_text_index.pkl'):
    # easyocr
    reader = easyocr.Reader(['ko', 'en'])

    # 텍스트를 키로, 이미지 이름 목록을 값으로 저장할 딕셔너리 초기화
    ocr_index = {}

    # 모든 이미지 파일 경로 가져오기
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

    # 이미지마다 OCR 텍스트 저장
    for img_path in image_paths:
        img_name = os.path.basename(img_path)  # 이미지 이름 추출
        texts = extract_text(reader, img_path)  # OCR 텍스트 추출

        # OCR로 추출한 각 텍스트를 인덱스에 추가
        for text in texts:
            if text in ocr_index:
                ocr_index[text].append(img_name)
            else:
                ocr_index[text] = [img_name]

        print(f"Processed {img_name}")

    # OCR 텍스트 인덱스를 피클 파일로 저장
    with open(output_file, 'wb') as f:
        pickle.dump(ocr_index, f)
    print(f'OCR text index saved to {output_file}')

if __name__ == '__main__':
    # 이미지 디렉토리 설정
    image_dir = 'train_image'
    save_ocr_text_index(image_dir)





##############################################################

def extract_text(reader, img_path, min_confidence=0.5):
    # 이미지 읽기 및 RGB 변환
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OCR 텍스트 추출
    ocr_results = reader.readtext(img_rgb)
    # 신뢰도 기준 이상의 텍스트만 추출
    texts = [res[1] for res in ocr_results if res[2] >= min_confidence]
    return texts

def save_ocr_text_index(image_dir, output_file='reduced_ocr_text_index.pkl', min_confidence=0.5):
    # easyocr 모델 초기화
    reader = easyocr.Reader(['ko', 'en'])

    # 텍스트를 키로 하고, 이미지 이름 목록을 값으로 저장할 딕셔너리 초기화
    ocr_index = {}

    # 모든 이미지 파일 경로 가져오기
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

    # 이미지마다 OCR 텍스트 저장
    for img_path in image_paths:
        img_name = os.path.basename(img_path)  # 이미지 이름 추출
        texts = extract_text(reader, img_path, min_confidence)  # OCR 텍스트 추출

        # OCR로 추출한 각 텍스트를 인덱스에 추가
        for text in texts:
            if text in ocr_index:
                ocr_index[text].append(img_name)
            else:
                ocr_index[text] = [img_name]

        print(f"Processed {img_name}")

    # OCR 텍스트 인덱스를 피클 파일로 저장
    with open(output_file, 'wb') as f:
        pickle.dump(ocr_index, f)
    print(f'OCR text index saved to {output_file}')

if __name__ == '__main__':
    # 이미지 디렉토리 설정
    image_dir = 'train_image'
    save_ocr_text_index(image_dir, min_confidence=0.5)