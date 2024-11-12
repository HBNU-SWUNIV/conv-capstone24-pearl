import torch
import numpy as np
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity
from image_vector import read_img, reduce_dimensionality, load_model
import re
import cv2 
import easyocr 
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

## 한글 폰트 설정 
#font_path = "C:/Windows/Fonts/malgun.ttf"
#fontprop = font_manager.FontProperties(fname=font_path)
#rc('font', family=fontprop.get_name())

def load_image_labels(label_file='image_label.txt'):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue  
            parts = line.split()
            filename = parts[0]
            point = int(parts[4])  
            label_dict[filename] = point
    return label_dict

def get_image_point(image_name, labels):
    return labels.get(image_name, "N/A")

def filter_text(results):
    filtered_results = []
    for res in results:
        text = res[1]  # OCR에서 추출한 텍스트
        # 특정 텍스트 패턴("SW", "AI") 찾기
        if "SW" in text or "AI" in text:
            filtered_results.append(text)
        else:
            # 숫자 범위 (200~299, 300~399)에 해당하는 숫자 찾기
            numbers = re.findall(r'\b\d+\b', text)  # 텍스트에서 모든 숫자 추출
            for number in numbers:
                num = int(number)
                if 200 <= num < 300 or 300 <= num < 400:
                    filtered_results.append(text)
                    break  # 하나라도 범위에 맞으면 추가하고 다음 텍스트로 넘어감
    return filtered_results

def find_most_similar_images(target_image_path, model, reader, labels, top_k=10, pca_path='pca_model.pkl', data_path='reduced_image_vectors_list.pkl', cuda=False):
    # 저장된 PCA 모델과 벡터 데이터를 불러오기
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(data_path, 'rb') as f:
        saved_vectors = pickle.load(f)

    # 타겟 이미지 벡터화 및 PCA 차원 축소
    target_image_vector = model.pool(model.encoder(read_img(target_image_path, cuda)))
    target_image_vector = target_image_vector.cpu().detach().numpy().flatten()
    target_image_vector = reduce_dimensionality([target_image_vector], pca_path=pca_path, train_pca=False)[0]

    # 코사인 유사도를 통해 유사한 이미지 찾기
    image_names = [item[0] for item in saved_vectors]
    vectors = np.array([item[1] for item in saved_vectors])
    similarities = cosine_similarity([target_image_vector], vectors)[0]

    # 상위 top_k개의 유사한 이미지 인덱스 가져오기
    most_similar_indices = np.argsort(similarities)[-top_k:][::-1]
    most_similar_images = [(image_names[i], similarities[i]) for i in most_similar_indices]
    
	# OCR 실행
    img = cv2.imread(target_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb) 
    filtered_texts = filter_text(results)     
    floor = None
    
    for text in filtered_texts:
        if "SW" in text or any(200 <= int(num) < 300 for num in re.findall(r'\b\d+\b', text)):
            floor = "2F"
            break
        elif "AI" in text or any(300 <= int(num) < 400 for num in re.findall(r'\b\d+\b', text)):
            floor = "3F"
            break

    if floor:
        filtered_images = [(name, sim) for name, sim in most_similar_images if name.startswith(floor)]
        print(f"'{target_image_path}'와 가장 유사한 이미지 {top_k}개 중 '{floor}'층만 남긴 결과:")
        for idx, (image_name, similarity) in enumerate(filtered_images, start=1):
            point = labels.get(image_name, "N/A")
            print(f"{idx}. {image_name} (유사도: {similarity:.4f}, 포인트: {point})")
    else:

        print(f"'{target_image_path}'와 가장 유사한 이미지 {top_k}개:")
        for idx, (image_name, similarity) in enumerate(most_similar_images, start=1):
            point = labels.get(image_name, "N/A")
            print(f"{idx}. {image_name} (유사도: {similarity:.4f}, 포인트: {point})")

    return most_similar_images

if __name__ == '__main__':

    start_time = time.time()

    weight_path = 'pretrain-model.pth.tar'
    cuda = torch.cuda.is_available()
    model = load_model(weight_path, cuda)
    reader = easyocr.Reader(['ko', 'en'])

    labels = load_image_labels('image_label.txt')

    target_image = 'image_data/2F_day(50).jpg'
    target_image2 = target_image.replace('image_data/', '')
    point = get_image_point(target_image2, labels)
    print(f"'{target_image}'의 포인트 값: {point}")

    print()
    find_most_similar_images(target_image, model,reader,labels, top_k=10, pca_path='pca_model.pkl', data_path='reduced_image_vectors_list.pkl', cuda=cuda)
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.4f} seconds")


