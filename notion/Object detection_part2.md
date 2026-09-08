# Object detection_part2

개요

본 문서는 Object Detection의 두 번째 축인 1-Stage Detector 계열(YOLO v1~v3, SSD, FPN)을 다룹니다. Region Proposal 단계 없이 단일 네트워크로 위치와 클래스를 동시에 예측하는 접근법의 발전 과정과, 다중 스케일 검출을 가능케 한 SSD·FPN의 구조적 아이디어, 그리고 YOLO 시리즈의 세대별 개선 포인트를 정리하였습니다.

Part 3. 1-Stage Detection (YOLO / SSD 계열)

YOLO v1 (You Only Look Once)

YOLO v1 구조

YOLO v1 장단점

SSD (Single Shot MultiBox Detector)

SSD – Default Boxes & Feature Map별 스케일

SSD Loss

SSD 장단점

YOLO v2

YOLO v2 개선 단계별 성능 (VOC mAP)

- **차원 클러스터(Dimension Clusters):** 앵커 박스 크기를 수작업 대신 k-평균 군집화로 자동 산출
- **Fine-Grained Features:** 13×13 특징 맵의 한계를 보완하기 위해 26×26 저층 특징을 가져오는 passthrough layer 추가
- **Darknet-19:** 19개 conv + 5개 pooling 계층, ImageNet Top-1 72.9%/Top-5 91.2%
- **동작 원리:** 이미지 분할(그리드) → 앵커 기반 박스 예측 → NMS로 최종 감지
- PASCAL VOC2012 테스트 기준 YOLOv2는 Faster R-CNN(ResNet)·SSD512와 대등한 성능이면서 2~10배 빠름

YOLO v2 장단점

YOLO 9000

YOLO 9000 장단점

YOLOv3

FPN (Feature Pyramid Network)

YOLOv3 네트워크 구조

YOLOv1 / v2 / v3 비교

YOLO 시리즈 계보

참고 링크

YOLO v1 논문: https://arxiv.org/pdf/1506.02640

YOLO v1 구조: https://medium.com/@saptarshimt/yolo-v1-pascal-voc-simplistic-pytorch-implementation-from-scratch-961fa36f4d4d

SSD 논문: https://arxiv.org/pdf/1512.02325

SSD 개념/스케일: https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/

SSD 출력 채널: https://wikidocs.net/142280

YOLO v2 논문: https://arxiv.org/pdf/1612.08242

YOLO v2 앵커/구조: https://wikidocs.net/167664

YOLO 9000 해설: https://ffighting.net/deep-learning-paper-review/object-detection/yolo-v2/

FPN 개념: https://herbwood.tistory.com/18

YOLOv3 구조: https://wikidocs.net/163583, https://wikidocs.net/163607

YOLOv3 로지스틱 분류: https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/

YOLO 시리즈 계보: https://devocean.sk.com/blog/techBoardDetail.do?ID=166976

