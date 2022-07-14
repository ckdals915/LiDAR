# PointPillars: Fast Encoders for Object Detection from Point Clouds

**Date**: 							2022.07.13

**Name**:  						ChangMin An

**Github**: 						[Link](https://github.com/ckdals915/LiDAR)



## I. Introduction

PointPillars는 downstream detection pipeline에 적합한 pillar인 point cloud encoder이다. 기존에 사용되는 encoder은 크게 두 가지로 나뉘는데, 빠르지만 정확도가 떨어지는 fixed encoder와 느리지만 정확도가 더 높은 encoders that are learned from data로 나뉜다. Pillars는 기존의 2D Conv를 사용할 수 있어 빠르고, 정확도 또한 다른 encoder에 비해 높은 장점을 가지고 있다.

PointPillars는 pillar마다 feature를 구하여 object의 3D oriented bounding box를 구하는 방법이다. PointPillars의 장점은 첫째로 fixed encoder를 통해 point cloud로부터 얻을 수 있는 모든 information을 활용할 수 있다. 두번째로 voxel대신 pillars를 사용함으로써 hyper parameter로 tuning하던 vertical direction의 binning을 할 필요가 없어졌다. 마지막으로 모든 key operation이 2D Conv로 이루어지기에 매우 빠르다.

즉 voxel 단위로 feature를 나눈 후 각 voxel마다 PointNet을 적용한다. 이후 3D Conv를 진행하여 수직축을 통합하고 그 이후 2D Conv를 진행한다.

<img src="https://github.com/ckdals915/LiDAR/blob/main/docs/pictures/Bird's_Eye_View_Performance_PointPillars.jpg?raw=true?raw=true?raw=true?raw=true" style="zoom:80%;" />

**Figure 1. KITTI dataset에서 PointPillars(PP)를 비롯한 여러 알고리즘의 성능 정리(A: AVOD, M: MV3D, C: Contfuse, V: VoxelNet, F: Frustum PointNet, S: SECOND, P+: PIXOR++)**

**Main Contribution**

* End-to-end 3D object detection에 쓰이는 새로운 point cloud encoder인 PointPillars를 제시하였다.
* 다른 3D object detection 알고리즘에 비해 2~4배 빠르게 pillars에 2D Conv를 사용하는지 보여주었다.
* KITTI Dataset에서 실험을 수행하여 3D와 BEV benchmarks에 대해 SOTA의 성능을 보여주었다.
* Ablation study를 통해 strong detection performance를 보여주는 key factor가 무엇인지 알 수 있다.



## II. PointPillars Network

<img src="https://github.com/ckdals915/LiDAR/blob/main/docs/pictures/PointPillars_Architecture.jpg?raw=true?raw=true?raw=true?raw=true" style="zoom:80%;" />

**Figure 2. PointPillars Architecture**

PointPillars는 크게 3가지 단계로 나뉜다.

* Point cloud를 sparse pseudo-image로 바꾸는 feature encoder network
* Pseudo-image를 high-level representation으로 바꿔주는 2D conv network
* 3D boxes를 regression하는 detection head



### 1. Pointcloud to Pseudo-Image

2D convolutional architecture를 구현하기 위해 point cloud를 pseudo-image 형태로 변환해야 한다. 먼저 point cloud를 x-y 평면 위의 grid로 discretized 하여 size가 B인 pillar의 set P를 만들어 준다. 이는 z 평면으로 spatial limit이 존재하지 않아 이에 대한 hyper parameter가 필요없다. 이 때 cluster center의 x, y, z값을 얻게 되고, pillar center x, y 값을 얻게 된다. 이렇게 총 9 dimension을 얻게 된다. 
$$
(x, y, z, r, x_C, y_C, z_C, x_P, y_P)
$$
생성된 pillar는 LiDAR point가 sparse하기 때문에 대부분 비어있게 된다. 이 sparsity는 비어 있지 않은 pillar의 수(P)와 pillar당 들어있는 point 수(N)에 제한을 두어 (D, P, N)크기의 dense한 tensor를 정의하여 처리한다. 만약 data양이 많아 tensor의 크기를 초과한다면 data는 random하게 sample되어 들어간다. 너무 적은 data를 갖게 되면 zero padding을 사용하여 해결한다.

그 이후 PointNet을 사용한다. 먼저 각 point마다 linear layer, batch norm, ReLU를 적용하여 (C, P, N) 크기의 tensor를 생성한다. 그리고 각 channel에 대해 Max Pooling을 하여 (C, P) 크기의 tensor를 생성한다. Encoding 이후 결과적으로 (C, H, W)의 pseudo-image를 생성한다.



### 2. Backbone

<img src="https://github.com/ckdals915/LiDAR/blob/main/docs/pictures/Backbone_Network.jpg?raw=true?raw=true?raw=true?raw=true" style="zoom:80%;" />

**Figure 3. Backbone Network(2D conv)**





## III. PointNet Architecture

<img src="https://github.com/ckdals915/LiDAR/blob/main/docs/pictures/PointNet_Architecture.jpg?raw=true?raw=true?raw=true?raw=true" style="zoom:80%;" />

Architecture에는 3가지 주요 모듈이 있다. **max pooling layer**는 모든 점들로부터 정보를 모으기 위한 symmetric 함수이며, **지역 및 전역 정보 조합 구조**, 입력 점군과 점 특징들을 정렬하는 2개의 **joint alignment network**로 구성된다. 



### 1. Symmetry Function for Unordered Input

정렬은 2차원에서 좋은 solution이지만 높은 차원(3D)의 자료 정렬은 존재하지 않는다. 예를 들어, 고차원 공간의 점들을 1차원 실수 선으로 projection한 후 정렬할 수 있으나, 이에 대한 역변환으로 원 데이터를 복구할 수 없다. 이를 해결하기 위한 PointNet의 아이디어는 변환된 요소에 대한 **symmetric function**(max-pooling)을 적용한 점군을 정의하는 것이다. 이는 순서에 상관없이 결과가 일정하게 나오기 위함이다. 

f({x1, ..., xn}) = g(h(x1), ..., h(xn))

이 때 g가 max pooling을 해주는 symmetric function이다.



### 2. A Local and Global Information Combination Structure

벡터 f1, ..., fk 형태의 출력은 입력 집합에 대한 global information이다. SVM이나 MLP를 이용해 형상의 전역 특징을 학습하는 것은 쉽지만, local 및 global information을 구분해 얻는 것이 필요하다. **전역 특징이 계산된 후, 각 점들의 특징과 전역 특징을 연결하여 포인트 특징을 얻는다.** 



### 3. Joint Alignment Network

점군의 labeling은 형상 변환(translation, rotation)에 대해 불변이어야 한다. 이를 위해 mini-network(T-net)을 정의하고 적용한다. 이 때 T-net에서 transformation된 matrix를 추정하여 사용한다. 

