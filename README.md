# LG_Aimers_4th 

![image](https://github.com/BaekJunehong/LG_Aimers_Hackathon/assets/101456289/8f923c8d-a8ac-4028-9c85-01062742610e)

## 0. 개요
LG Aimers에서 주관한 프로그램으로서 1달간의 온라인교육기간과 이후 온라인 해커톤을 1달간 진행하였습니다.  

교육과정으로 머신러닝,딥러닝등의 AI관련 내용과 함께 해커톤과 관련된 도메인 지식으로 B2B 마케팅에 대한 강의가 제공되어 해커톤에 대비하여 전문적인 지식을 얻어갈수 있었습니다.  

1달간의 교육기간이 끝나고 온라인 해커톤이 진행되었습니다.  

## 1. 프로젝트 소개 

최근 영업 과정에서 전환 가능성이 높은 고객에게 영업 자원을 집중을 목표로 합니다.  
이 과정에서 고객의 전환 여부를 예측하기 위해 기계학습 모델을 사용하려고 하는 동향성을 보이고 있습니다.  

이러한 부분에 있어 고객의 다양한 정보를 통해 해당 고객이 전환 고객인지 아닌지를 판단하는 모델을 구현하는게 이번 프로젝트의 과제였습니다.  

## 2. 프로젝트 과정  
### Ch1. data info
전반적인 데이터 구성에 대해 확인하는 과정을 가졌습니다.  

각 변수들에 대해서 결측치,이상치,통계량을 확인하며 전반적인 전처리 흐름에 대해 파악하였습니다.

### Ch2. data processing
각 변수별로 확인하며 전처리가 필요한 부분에 대해서 확인 및 처리하는 과정을 진행하였습니다.  

1) 결측치에 대한 처리
   -> 다른변수들을 이용해 값을 구하거나 특정값으로 변환
   
2) 이상치에 대한 처리
   -> 특정 빈도수 이하의 값에 대하여 값변환
   
3) 문자형 자료에 대한 처리
   -> 일정한 기준에 의해 범주화 진행 또는 타겟변수의 비율계산을 이용한 범주화를 진행하여 새로운 변수 생성

### Ch3. feature_engineering  
모델학습을 하기위한 준비하는 과정으로서 라벨링을 하는 과정과 함께  
모델 성능을 높이기위한 작업으로 파생변수 생성 및 성능저하에 영향을 주는 변수 제거를 진행하였습니다.

1) 타겟인코딩을 이용하여 문자형 데이터에 대해서 수치형으로 변환
2) 변수간의 관계를 확인하며 관련 변수들에 대하여 파생변수를 생성
3) 상관관계 및 다중공산성을 확인하며 높은 수치를 나타내는 변수에 대하여 제거

### Ch4. modeling   
학습을 진행하기전 전반적으로 타겟변수의 비율이 극단적으로 분포하여 이에 대해 샘플링을 진행하였습니다.  

1) 오버샘플링 -> SMOTE
2) 다운샘플링 -> RandomUnderSampler

다음으로 본격적 모델링 작업을 진행하였습니다.  

1) Optuna
  * 전반적인 분류모델에 대하여 Optuna를 이용하여 모델에 대한 하이퍼 파라미터를 찾음

2) Voting & Stacking
  * 개별 모델들에 대해서 Voting 과 Stacking 기법을 이용하여 성능을 극대화 시키려는 작업을 진행
  * 성능 비교과정을 통해 최적의 모델을 찾는 과정을 진행

최종모델로는 LightGBM, XGBoost, RandomForest, ExtraTree 모델들을 Soft Voting 기법을 이용하여 모델링을 진행하였고 예측된 결과에 대해서 cutoff-value 값을 조절하여 조금더 True 값을 더 잘 예측할수 있도록 모델링을 하였습니다.

## 3. 프로젝트 결과
<img src="https://github.com/BaekJunehong/LG_Aimers_Hackathon/assets/101456289/7bd0607c-21d7-4ccd-b88d-dbb364ba952f" alt="image" width="500" height="280">
<img src="https://github.com/BaekJunehong/LG_Aimers_Hackathon/assets/101456289/2534857e-1e8b-4abe-84d5-5fddaa9f5d71" alt="image" width="500" height="280">

Final 점수기준으로 30등(팀) 까지 본선에 진출하였는데 결과적으로 64등을 하면서 본선까지의 여정은 이어나가지 못하고 마무리 하게 되었습니다.  

## 4. 맺음글  
이전까지는 정제되어 있는 데이터를 이용하여 전처리를 진행하였었는데 이번 기회에 실무적인 데이터를 접하면서 상대적으로 날것의 데이터를 다루며 초반에는 당황을 하며 진행하였던거 같습니다.  

각 변수별로 결측값이 없는 변수들도 있는 반면에 때로는 결측값이 대부분인 변수들을 접하며 어떻게 처리를 할지에 대해서 많은 고민들을 했었고 훈련데이터의 타겟변수의 값의 비율을 통한 범주화의 방법을 이번 대회를 통해 겪으면서 이러한 방법을 활용할수도 있다는 것을 알게되며 때로는 신선한 충격이었던거 같습니다.  

모델링 부분에서는 전처리에 대한 힘을 많이 들인것에 대비하여 좋은 성능을 내지 못하는 부분에 있어 조금은 좌절감을 느끼기도 했지만 모델링을 진행하면서 cutoff-value를 조절할는 방안에 대해서 고려하게 되면서 어느정도 만족스러운 성능값을 나오는 부분에서 성취감을 느꼈던거 같습니다.  

비록 본선에는 진출하지는 못하였지만 나름의 성장할수 있는 기회의 장이였다 생각이 들며 여러번의 전처리 방법들의 변경과 여러 모델링 기법들을 활용해 보면서 많은 공부를 하며 배울수 있는 시간이었다고 생각합니다. 
















