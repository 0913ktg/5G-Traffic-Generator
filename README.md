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
pip install -r requirements.txt
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

## Built With

* [DAEGYEOM KIM](https://github.com/0913ktg) - Implementation of the N-HiTS-5G model
* [MYEONGJIN KO](https://github.com/KoMyeongjin) - Implementation of the 5GT-GAN model

## License

This project is licensed under the MIT License

## Acknowledgments

* Date of submission October 31, 2022.  
* This work was supported by the In-stitute for Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (No. 2021-0-00092); and in part by the National Research Foundation of Korea (NRF) grant funded by the Korea government Ministry of Science and ICT (No. 2021R1F1A1064080). 
