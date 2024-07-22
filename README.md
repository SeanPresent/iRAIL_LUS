# The Project LUS | Lung Ultrasound
## 업데이트 내용 및 자료
**미팅 및 업데이트:** ```iRAIL-SNU/LungUS/meeting/ 에서 확인``` 가능 <br>
**미팅 외 자료들 :** 발표자료 및 기타 데이터 [링크](https://drive.google.com/drive/folders/1e69s8RKwfRnDcHjZtiBGJsKwJrl_PKdH?usp=sharing)

## 프로젝트 소개
**소개 :** Deep Learning을 이용하여 lung ultrasound finding을 BLUE protocol에 적용한 연구 <br>
**목적 :** 급성호흡곤란을 일으키는 환자들에게서 원인을 찾기 위해 고안된 BLUE protocol은 90.5% 이상의 높은 진단적 정확도를 보이지만, 초음파 진단의 특성 상```시술자에 따라 해석이 달라질 수 있다는 단점이 있기에 숙련된 검사자와 비슷한 정도의 정확도를 갖출 수 있도록 보조```해주는 모델 연구 
<p align="center">
<img src="image/LUS_example.png" width="512">

## 해당 프로젝트 모델 : <br>
**주제** : Multi-label Classification in Blue Protocol setting. <br>

* 1\. **데이터**
    * 1.1\. **Label :** A-line, B-line, Consolidation, Effusion
    * 1.2\. **Case :** 3,429명 [기존 188명 + 새로운 환장 3,241명 추가]
* 2\. **모델**
    * 2.1\. **Task :** Multi-label Classification
<p align="center">
<img src="image/model_example.png" width="512"> <br>
(eg) Roy, Subhankar et al. “Deep Learning for Classification and Localization of COVID-19 Markers in Point-of-Care Lung Ultrasound.” IEEE Transactions on Medical Imaging 39 (2020): 2676-2687.
   
## 참고 문헌
 
1 "Automated Lung Ultrasound B-line Assessment Using a Deep Learning Algorithm." IEEEonUFFC2020 <br>
2 "Quantifying lung ultrasound comets with a convolutional neural network: Initial clinical results.“CBM2019 <br>
3 "Localizing B-Lines in Lung Ultrasonography by Weakly Supervised Deep Learning, In-Vivo Results.“ IEEEonBHI2019 <br>
4 "Deep learning for classification and localization of COVID-19 markers in point-of-care lung ultrasound.“ IEEEonMI2020 <br>
5 "POCOVID-Net: automatic detection of COVID-19 from a new lung ultrasound imaging dataset (POCUS).“ POCOVID2020 <br>
6 "Lung mass density analysis using deep neural network and lung ultrasound surface wave elastography.“ Ultrasonics2018 <br>
7 "Predicting lung mass density of patients with interstitial lung disease and healthy subjects using deep neural network and lung ultrasound surface wave elastography.“JMBBM2020 <br>
8 "Ultrasound-based detection of lung abnormalities using single shot detection convolutional neural networks.“ MICCAI2018 <br>
9 "Deep Learning-Based Pneumothorax Detection in Ultrasound Videos." SUSI2019 <br>
10 "Enhanced Point‐of‐Care Ultrasound Applications by Integrating Automated Feature‐Learning Systems Using Deep Learning.“ JUM2019 <br>
