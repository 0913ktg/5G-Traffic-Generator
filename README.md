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

### N-HiTS-5G

  ```
  cd N-HiTS-5G
  pip install - r requirements.txt
  ```

## Running the tests

### N-HiTS-5G

  1. Run model_train.py
    #### Options
    + dataset : Select one name for the dataset want to learn.
    + datatype : Select one name for the dataset type want to learn (ul or dl).
    + hyperopt_max_evals : Enter the maximum number of evaluations for hyperparameter tuning.
    + experiment_id : Enter a title for the current experiment.

    ```
    python3 model_train.py --dataset afreeca --datatype dl --hyperopt_max_evals 10 --experiment_id test_1
    ```

  2. Run inference.py
    #### Option
    + dataset : Select one name for the dataset want to learn.
    + datatype : Select one name for the dataset type want to learn (ul or dl).
    + experiment_id : Enter a title for the current experiment.
    + size : Enter the size of the traffic you want to generate (output length = horizon * size).

    ```
    python3 inference.py --dataset afreeca --datatype dl --experiment_id test_1 --size 10
    ```

  3. Run evaluation.py

    ```
    python3 evaluation.py
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
