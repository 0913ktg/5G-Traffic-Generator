# 5G Traffic Generation for Practical Simulations Using Open Datasets

<img width="1076" alt="5GT-GAN" src="https://user-images.githubusercontent.com/57590655/198483530-cb958da0-4b29-45c1-8556-94aa4f4be211.png">
<img width="1019" alt="N-HiTS-5G" src="https://user-images.githubusercontent.com/57590655/198483553-84d4dd52-a69f-4f1c-a979-a1f94937c7ca.png">

## Getting Started 

### Environment
* Ubuntu 20.04 LTS
* Docker 
* Pytorch & Pytorch Lightning

### Prerequisites 
* Recommend install CUDA, CUDNN

### Installing

```
pip install requirements.txt
```

## Running the tests / 테스트의 실행

어떻게 테스트가 이 시스템에서 돌아가는지에 대한 설명을 합니다

### N-HiTS-5G

First run model_train.py

select dataset name
select datatype (ul, dl)

```
python3 model_train.py --dataset afreeca --datatype dl --hyperopt_max_evals 10 --experiment_id test_1
```

### 테스트는 이런 식으로 작성하시면 됩니다

```
예시
```

## Deployment / 배포

Add additional notes about how to deploy this on a live system / 라이브 시스템을 배포하는 방법

## Built With / 누구랑 만들었나요?

* [이름](링크) - 무엇 무엇을 했어요
* [Name](Link) - Create README.md

## Contributiong / 기여

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. / [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) 를 읽고 이에 맞추어 pull request 를 해주세요.

## License / 라이센스

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/PurpleBooth/LICENSE.md) file for details / 이 프로젝트는 MIT 라이센스로 라이센스가 부여되어 있습니다. 자세한 내용은 LICENSE.md 파일을 참고하세요.

## Acknowledgments / 감사의 말

* Hat tip to anyone whose code was used / 코드를 사용한 모든 사용자들에게 팁
* Inspiration / 영감
* etc / 기타
