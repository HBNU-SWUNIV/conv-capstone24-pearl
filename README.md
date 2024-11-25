# 한밭대학교 인공지능소프트웨어학과 PEARL팀

**팀 구성**
- 20221080 최진 
- 20231051 김현빈
  

## <u>Teamate</u> Project Background
- ### 필요성
  - 실내에서는 GPS 신호가 약하거나 도달하지 않기 때문에 정확한 위치 추정이 어렵다.
    그래서 실내에서 로봇이나 드론과 같은 자율 이동 장치는 위치추정이 어렵기 때문에 이미지 만으로 위치를 추정하는 기술이 필요하다.
    
- ### 기존 해결책의 문제점
  - 이미지로 위치를 추정하는 기술로는 NetVLAD가 있는데 기존 기술은 낮과 밤, 조명 조건에 따라 위치추정하는게 달라지고, 공간적 정보로만 위치추정을 하기에 공간이 조금이라도 바뀌면 위치추정에 어려움이 생긴다. 그래서 우리는 낮과 밤에 robustness하며 공간적 정보뿐만 아니라 문맥적 정보도 활용하는 NetVLAD layer를 만들것이다.
  
## System Design
  - ### System Requirements
    - python
    - pytorch
    - easyocr
    - sklearn
    - pillow
    - numpy
    
  
## Conclusion
  - ### 성능차이
|모델|전체 적중률|좌표 적중률|층 적중률| 
|------|---|---|---|
|NetVLAD|40.74|67.13|57.41|
|OCR + NetVLAD|54.17|70|73.5|


  
## Project Outcome
- ### 20XX 년 OO학술대회 

