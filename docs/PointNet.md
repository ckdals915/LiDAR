# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

**Date**: 							2022.07.13

**Name**:  						ChangMin An

**Github**: 						[Link](https://github.com/ckdals915/LiDAR)



## I. Introduction

센서 스캔 결과로 얻어지는 3차원 point cloud를 인식하기 위해, 많은 연구자들은 point cloud를 Voxel 형식으로 변환한다. 복셀은 요약하기 좋으나, 빈 공간이 많이 발생하여 비효율적인 측면이 있다. PointNet은 입력 시 포인트 순열의 불변성을 이용한다. PointNet을 통해, Classification, Segmentation, 의미 해석 등에 필요한 아키텍처를 제공한다.



## II. Properties of Point sets

PCD(point cloud data)는 {Pi | i = 1, ... , n}, P = (x, y, z)로 정의한다. PointNet은 k개의 output을 가지도록 class를 정의하며, 이 모델을 통해 n x m scores output을 가질 수 있다. 이 때, n: PCD m: semantic subcategory이다. PointNet의 input은 Euclidean space에서 포인트들의 subset이다.

* Unordered: 점들 사이에 순서가 없다.
* Interaction among points: 점들간 상호영향을 받는다.
* Invariance under transformations: 변환에 불변이어야 한다.



## III. PointNet Architecture



 