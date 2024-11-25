from image_vector import read_img, reduce_dimensionality, load_model
import torch
import numpy as np
import pickle
import time
import cv2
import re
import easyocr
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(reader, img_path, min_confidence=0.5):
    # 이미지 읽기 및 RGB 변환
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # OCR 텍스트 추출
    ocr_results = reader.readtext(img_rgb)
    # 신뢰도 기준 이상의 텍스트만 추출
    texts = [(res[1],res[2]) for res in ocr_results if res[2] >= min_confidence]
    return texts

def load_image_labels(label_file='image_label.txt'):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  
            parts = line.split()
            filename = parts[0]
            point = (float(parts[1]), float(parts[2]), float(parts[3]))  
            label_dict[filename] = point
    return label_dict

def determine_floor(filtered_texts):
    floor = None
    for text, confidence in filtered_texts: 
        if "SW" in text or (any(200 <= int(num) < 300 for num in re.findall(r'\b\d+\b', text)) or str('2F') in text):
            floor = "2F"
            break
        elif "AI" in text or (any(300 <= int(num) < 400 for num in re.findall(r'\b\d+\b', text)) or str('3F') in text):
            floor = "3F"
            break
    return floor
def get_image_point(image_name, labels):
    return labels.get(image_name, "N/A")

def find_most_similar_images(target_image_path, model, reader, labels, top_k=10, ocr_data='ocr_text_index.pkl', pca_path='pca_model.pkl', data_path='reduced_image_vectors_list.pkl', cuda=False):
    with open(ocr_data, 'rb') as f:
        ocr_index = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(data_path, 'rb') as f:
        saved_vectors = pickle.load(f)

    target_image_vector = model.pool(model.encoder(read_img(target_image_path, cuda)))
    target_image_vector = target_image_vector.cpu().detach().numpy().flatten()
    target_image_vector = reduce_dimensionality([target_image_vector], pca_path=pca_path, train_pca=False)[0]

    # OCR 텍스트 추출 및 층 구별
    ocr_results = extract_text(reader=reader, img_path=target_image_path, min_confidence=0.5)
    floor = determine_floor(ocr_results)
    구별번호 = 3  # 기본값: 전체 실패로 설정

    # OCR 텍스트 추출
    filtered_ocr_index = {key: value for key, value in ocr_index.items() if len(value) <= 20}
    valid_ocr_texts = sorted([(res[0], res[1]) for res in ocr_results if res[0] in filtered_ocr_index and res[1] >= 0.5], key=lambda x: x[1], reverse=True)


    if floor:
        구별번호 = 1  # 층 구별 성공, OCR 성공 여부는 아직 판단되지 않음
        if valid_ocr_texts:
            # 각 텍스트에 매칭되는 이미지 목록 수집
            matched_images = set()
            for text, _ in valid_ocr_texts:
                matched_images.update(filtered_ocr_index.get(text, []))

            # 층 필터링
            matched_images = [img for img in matched_images if img.startswith(floor)]
            if matched_images:
                구별번호 = 0  # 층과 OCR 모두 성공

                print("OCR로 매칭된 이미지 존재, 코사인 유사도 계산 시작")
                matched_vectors = [vec for name, vec in saved_vectors if name in matched_images]

                matched_vectors = np.array(matched_vectors)
                similarities = cosine_similarity([target_image_vector], matched_vectors)[0]
                most_similar_index = np.argmax(similarities)
                most_similar_image = (matched_images[most_similar_index], similarities[most_similar_index], labels.get(matched_images[most_similar_index], "N/A"), 구별번호)
                return [most_similar_image]
        
        # 층 구별 성공, OCR 매칭 실패한 경우
        print(f"{floor}에 대한 OCR 매칭 실패, 해당 층 전체 이미지에서 유사도 계산")
        filtered_images = [(name, vec) for name, vec in saved_vectors if name.startswith(floor)]
        filtered_names, filtered_vectors = zip(*filtered_images)
        filtered_vectors = np.array(filtered_vectors)
        similarities = cosine_similarity([target_image_vector], filtered_vectors)[0]
        most_similar_index = np.argmax(similarities)
        most_similar_image = (filtered_names[most_similar_index], similarities[most_similar_index], labels.get(filtered_names[most_similar_index], "N/A"), 구별번호)
        return [most_similar_image]

    # 층 구별 실패, OCR 검색만 성공하는 경우
    if not floor and valid_ocr_texts:
        print("층 구별 실패 및 OCR 성공, 전체 이미지에서 OCR,유사도 계산")
        구별번호 = 2  # OCR만 성공
        matched_images = set()
        for text, _ in valid_ocr_texts:
            matched_images.update(filtered_ocr_index.get(text, []))
        matched_images = list(matched_images)

        if matched_images:
            matched_vectors = [vec for name, vec in saved_vectors if name in matched_images]
            matched_vectors = np.array(matched_vectors)
            similarities = cosine_similarity([target_image_vector], matched_vectors)[0]
            most_similar_index = np.argmax(similarities)
            most_similar_image = (matched_images[most_similar_index], similarities[most_similar_index], labels.get(matched_images[most_similar_index], "N/A"), 구별번호)
            return [most_similar_image]

    # 층 구별 및 OCR 모두 실패한 경우, 전체 데이터에서 유사도 계산
    print("층 구별 및 OCR 모두 실패, 전체 이미지에서 유사도 계산")
    image_names = [item[0] for item in saved_vectors]
    vectors = np.array([item[1] for item in saved_vectors])
    similarities = cosine_similarity([target_image_vector], vectors)[0]
    most_similar_index = np.argmax(similarities)
    most_similar_image = (image_names[most_similar_index], similarities[most_similar_index], labels.get(image_names[most_similar_index], "N/A"), 구별번호)
    return [most_similar_image]




if __name__ == '__main__':

    start_time = time.time()

    weight_path = 'pretrain-model.pth.tar'
    cuda = torch.cuda.is_available()
    model = load_model(weight_path, cuda)
    reader = easyocr.Reader(['ko', 'en'])

    labels = load_image_labels('image_label.txt')
    label2 = load_image_labels('test_image_label.txt')

    target_image = 'test_image/test(73).jpg'
    target_image2 = target_image.replace('test_image/', '')
    point = get_image_point(target_image2, label2)
    print(f"'{target_image}'의 포인트 값: {point}")

    print()
    a = find_most_similar_images(target_image, model,reader,labels, top_k=1, ocr_data='reduced_ocr_text_index.pkl', pca_path='pca_model.pkl', data_path='reduced_image_vectors_list.pkl', cuda=cuda)
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")
    print(a)