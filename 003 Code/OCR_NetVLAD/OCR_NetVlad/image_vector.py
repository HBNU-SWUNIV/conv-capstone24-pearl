import argparse
import torch
import torchvision
import os
import time
import numpy as np
import cv2
import glob
import pickle
import netvlad
from sklearn.decomposition import PCA

def reduce_dimensionality(vectors, pca_path='pca_model.pkl', output_dim=1044, train_pca=True):
    if train_pca:
        # PCA 학습 및 저장
        pca = PCA(n_components=output_dim)
        reduced_vectors = pca.fit_transform(vectors)
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f'PCA model saved to {pca_path}')
    else:
        # 저장된 PCA 모델 불러오기 및 차원 축소
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        reduced_vectors = pca.transform(vectors)
    return reduced_vectors

def read_img(img_path, cuda):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    inp = preprocess(img).unsqueeze(0)
    if cuda:
        inp = inp.cuda()
    return inp

def load_model(weight_path, cuda):
    encoder_dim = 1280
    encoder = torchvision.models.mobilenet_v2(pretrained=False)
    layers = list(encoder.features.children())
    encoder = torch.nn.Sequential(*layers)
    model = torch.nn.Module()
    model.add_module('encoder', encoder)
    net_vlad = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim, vladv2=False)
    model.add_module('pool', net_vlad)

    if cuda:
        checkpoints = torch.load(weight_path)
        model.load_state_dict(checkpoints['state_dict'])
        model = model.cuda()
    else:
        checkpoints = torch.load(weight_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoints['state_dict'])
    model.eval()
    return model

def save_to_pickle(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Data saved to {save_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MobileNet_v2-Netvlad Demo.')
    parser.add_argument('--image_dir', type=str, default='image_data/',
                        help='Directory containing images.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use cuda GPU to speed up network processing speed (default: False)')
    parser.add_argument('--pca_path', type=str, default='pca_model.pkl',
                        help='Path to save/load PCA model')
    parser.add_argument('--train_pca', action='store_true',
                        help='Train and save a new PCA model if set, otherwise load existing one')
    opt = parser.parse_args()

    cuda = opt.cuda
    image_dir = opt.image_dir
    pca_path = opt.pca_path
    train_pca = opt.train_pca

    if cuda:
        print('=> Using GPU!')
    else:
        print('=> Using CPU!')

    weight_path = 'pretrain-model.pth.tar'
    model = load_model(weight_path, cuda)

    vector_list = []

    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    for img_path in image_paths:
        print(f'Processing {img_path}')
        inp = read_img(img_path, cuda)

        s1 = time.time()
        image_encoding = model.encoder(inp)
        vlad_encoding = model.pool(image_encoding)
        print('====> Infer time:', time.time() - s1)

        # 벡터를 flatten하여 1D 배열로 변환
        vector_list.append(vlad_encoding.cpu().detach().numpy().flatten())

    # 전체 벡터를 2D 배열로 변환
    all_vlad_vectors = np.array(vector_list)

    # PCA를 사용해 벡터 차원 축소 (PCA 모델을 학습하고 저장 또는 로드하여 사용)
    reduced_vectors = reduce_dimensionality(all_vlad_vectors, pca_path=pca_path, output_dim=1044, train_pca=True)

    # 이미지 이름과 축소된 벡터값을 결합
    result = [(os.path.basename(image_paths[i]), reduced_vectors[i]) for i in range(len(image_paths))]

    # 축소된 벡터 리스트를 피클 파일로 저장
    save_to_pickle(result, 'reduced_image_vectors_list.pkl')
