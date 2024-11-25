from tkinter import Tk, Label, Button, filedialog, Frame, Canvas
from PIL import Image, ImageTk
from image_ocr import extract_text
from image_vector import read_img, reduce_dimensionality, load_model
import torch
import easyocr
from ocr_netvlad import load_image_labels, find_most_similar_images

selected_image = None  # 선택된 이미지를 저장할 변수
dxdy = [(0.0,0.0), (0.0,10.8), (0.0,23.94), (0.0,27.54), (0.0,31.14), (0.0,40.86), (0.0,50.58), (1.8,32.94), (10.44,32.94), (15.84,34.38), (26.28,15.84), (34.2,23.4), (33.12,30.6),
        (31.68,34.92), (30.6,40.32), (29.52,47.88), (28.08,55.8), (26.64,64.44),]
xy = [(262,93), (262,183), (262,305), (262,334), (262,370), (262,453), (262,534), (277,387), (350,386), (400,394), (494,235),
      (483,300), (473,368), (462,406), (454,455), (442,523), (430,595), (420,663)]



def circle(canvas, cx, cy, r):
    canvas.create_oval(cx-r, cy-r, cx+r, cy+r, width=5, outline='blue', tags='first')

def inference(img_path):
    weight_path = 'pretrain-model.pth.tar'
    cuda = torch.cuda.is_available()
    model = load_model(weight_path, cuda)
    reader = easyocr.Reader(['ko', 'en'])
    labels = load_image_labels('image_label.txt')
    r = find_most_similar_images(
        img_path, model, reader, labels,
        top_k=5, ocr_data='reduced_ocr_text_index.pkl',
        pca_path='pca_model.pkl', data_path='reduced_image_vectors_list.pkl',
        cuda=cuda
    )
    return r[0]


def select_image():
    global selected_image, img_label, info_label, file_path, progressbar

    # 파일 선택 창 열기
    file_path = filedialog.askopenfilename(
        title="이미지 선택",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )

    map_canvas.delete("first")

    if file_path:
        # 이미지 로드
        selected_image = Image.open(file_path)

        # 이미지 크기를 라벨 크기에 맞게 조정
        rotated_image = selected_image.rotate(-90, expand=True)
        img_width, img_height = 400, 400  # 원하는 라벨 크기
        resized_image = rotated_image.resize((img_width, img_height), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)

        # 이미지 출력
        img_label.config(image=tk_image, text="")  # 텍스트 제거 및 이미지 추가
        img_label.image = tk_image                 # 참조 유지 (GC 방지)

        # info_label에 로딩바 표시
        info_label.config(text="")  # 기존 텍스트 제거

        # 유사도 검색 실행 (비동기 처리)
        process_inference(file_path)


def process_inference(img_path):
    global  info_label

    # 유사도 검색 및 결과 업데이트
    image_name, similarity, image_point, _ = inference(img_path)

    # 장소명 및 유사도 업데이트
    info_label.config(text=f"좌표: {image_point[:2]} {int(image_point[2])}층")

    for i in range(len(dxdy)):
        if dxdy[i] == image_point[:2]:
            idx = i + 1
            break
    circle(map_canvas,xy[i][0], xy[i][1], 50)
    

# Tkinter GUI 설정
root = Tk()
root.title("국립한밭대학교 세종캠퍼스 위치추정")
root.geometry('1280x720')

# 프레임 생성
map_frame = Frame(root, width=680, height=720, bg='lightblue')
map_frame.place(x=0, y=0)

ui_frame = Frame(root, width=600, height=720, bg="white")
ui_frame.place(x=680, y=0)

# 지도 이미지 표시
image = Image.open('Sejong campus.png')
resized_image = image.crop((300, 0, 980, 720))
tk_image = ImageTk.PhotoImage(resized_image)
map_canvas = Canvas(map_frame, width=680, height=720, bg='white', bd=2)
map_canvas.place(x=0, y=0)
map_canvas.create_image(0,0, anchor="nw", image=tk_image)


# 이미지 표시 라벨
img_label = Label(ui_frame, text="이미지를 선택하세요", bg="gray", fg="white")
img_label.place(relx=0.5, rely=0.3, anchor="center", width=400, height=400)


# 장소명 및 유사도 라벨
info_label = Label(ui_frame, text="", font=("Arial", 12), bg="white")  # 초기에는 아무것도 표시하지 않음
info_label.place(relx=0.5, y=550, anchor="center")


# 이미지 선택 버튼
btn_select = Button(ui_frame, text="이미지 선택", command=select_image)
btn_select.place(x=250, y=650)

root.mainloop()
