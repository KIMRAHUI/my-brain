# Object detection_part1

개요

본 문서는 Object Detection(객체인식)의 기본 개념과 성능평가지표, 그리고 딥러닝 기반 객체 검출의 초기 패러다임인2-Stage Detector 계열(R-CNN → Fast R-CNN → Faster R-CNN)의 구조와 동작 원리를 정리한 강의노트입니다. Region Proposal 방식의 발전 과정과 RPN(Region Proposal Network)의 핵심 메커니즘까지 체계적으로 다룹니다.

Part 1. Object Detection 기초와 성능평가

Object Detection이란?

1-Stage vs 2-Stage Detector

Object Detection이 어려운 이유

객체 인식 Dataset

성능평가지표 – IoU (Intersection over Union)

성능평가지표 – Precision / Recall / Accuracy

Object Detection에서의 TP, FP, FN

정밀도-재현율 트레이드오프

Average Precision (AP)

mean Average Precision (mAP) & F1-Score

IoU Threshold별 mAP

Part 2. 2-Stage Detection (R-CNN 계열)

Object Localization

Bounding Box 회귀를 바로 적용할 때 발생하는 문제

Region Proposal

R-CNN (Regions with CNN features)

Region Proposal – Selective Search

R-CNN 동작 단계

R-CNN Bounding Box Regression

NMS (Non-Max Suppression)

R-CNN 장단점

Fast R-CNN

ROI Pooling (Region of Interest Pooling)

Fast R-CNN Loss 함수

Fast R-CNN 장단점

Faster R-CNN

Region Proposal Network (RPN)

RPN Loss 및 미니배치 구성

RPN 최종 Proposal 생성 및 후속 처리

참고 링크

Object Detection 개요: https://dacon.io/forum/405806, https://wikidocs.net/167508

1-stage/2-stage 구조 비교: https://www.researchgate.net/figure/Basic-deep-learning-based-one-stage-vs-two-stage-object-detection-model-architectures_fig2_358362847

COCO Dataset: https://cocodataset.org/#home

Open Images: https://storage.googleapis.com/openimages/web/factsfigures_v7.html

Pascal VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

IoU: https://oniss.tistory.com/36

Precision/Recall/mAP: https://herbwood.tistory.com/2

Precision-Recall Trade-off: https://datascience-george.medium.com/the-precision-recall-trade-off-aa295faba140

mAP 계산: https://github.com/rafaelpadilla/Object-Detection-Metrics

Object Localization/Detection: https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/object_localization_and_detection

Region Proposal 개요: https://velog.velcdn.com/images/qtly_u/post/c8552acf-3671-4bed-a307-d4fdde78b71f/image.webp

R-CNN 논문: https://arxiv.org/pdf/1311.2524

R-CNN 정리: https://wikidocs.net/148633

Selective Search: https://developer-lionhong.tistory.com/31, http://www.huppelen.nl/publications/selectiveSearchDraft.pdf

Bounding Box Regression: https://better-tomorrow.tistory.com/entry/Bounding-box-regression

NMS: https://wikidocs.net/142645

Fast R-CNN 논문: https://arxiv.org/pdf/1504.08083

Fast R-CNN 개념: https://seongkyun.github.io/papers/2019/01/06/Object_detection/

ROI Pooling: https://herbwood.tistory.com/8

Faster R-CNN 논문: https://arxiv.org/pdf/1506.01497

Faster R-CNN 개념: https://viso.ai/deep-learning/faster-r-cnn-2/

RPN/Anchor Box: https://herbwood.tistory.com/10

Faster R-CNN 정리(nttuan8): https://nttuan8.com/bai-11-object-detection-voi-faster-r-cnn/

