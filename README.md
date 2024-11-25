# 한밭대학교 인공지능소프트웨어학과 PEARL팀

**팀 구성**
- 20221080 최진 
- 20231051 김현빈
  

## <u>Teamate</u> Project Background
- ### 필요성
  - 실내에서는 GPS 신호가 약하거나 도달하지 않기 때문에 정확한 위치 추정이 어렵다.
    따라서 실내에서 로봇이나 드론과 같은 자율 이동 장치는 위치추정이 어렵기 때문에 이미지 만으로 위치를 추정하는 기술이 필요하다.
    
- ### 기존 해결책의 문제점
  - 이미지로 위치를 추정하는 기술로는 NetVLAD가 있는데 기존 기술은 낮과 밤, 조명 조건에 따라 위치추정에 어려움이 생기고, 공간적 정보로만 위치추정을 하기에 공간이 조금이라도 바뀌면 위치추정에 어려움이 생긴다. 그렇기에 우리는 낮과 밤에 robustness하며 공간적 정보뿐만 아니라 문맥적 정보도 활용하는 OCRNetVLAD layer를 설계한다.

## System Design
- ### 알고리즘 구조
  ![순서도](https://github.com/HBNU-SWUNIV/conv-capstone24-pearl/blob/5071ef711248050f9cf71e31b9e58282c3d39cc4/002%20Presentation/%EC%88%9C%EC%84%9C%EB%8F%84.png)
  - ### System Requirements
    - python
    - pytorch
    - easyocr
    - sklearn
    - pillow
    - numpy
    
  
## Conclusion
  - ### 성능(R@1)
|모델|전체 적중률|좌표 적중률|층 적중률| 
|------|---|---|---|
|NetVLAD|40.74|67.13|57.41|
|OCR + NetVLAD|54.17|70|73.5|


![결과사진1](https://github.com/HBNU-SWUNIV/conv-capstone24-pearl/blob/c56c160e8e096318ef817f4c005f9e8bc19d26ab/002%20Presentation/%EC%B6%94%EB%A1%A0%EC%84%B1%EA%B3%B51.png)

![결과사진2](https://github.com/HBNU-SWUNIV/conv-capstone24-pearl/blob/c56c160e8e096318ef817f4c005f9e8bc19d26ab/002%20Presentation/%EC%B6%94%EB%A1%A0%EC%84%B1%EA%B3%B52.png)


  
## Project Outcome
- ### 20XX 년 OO학술대회 

