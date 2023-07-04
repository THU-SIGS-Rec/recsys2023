
# RecSys Challenge 2023

This project is a solution of Recsys Challenge 2023 provided by the team ```SSL```. For more details about this challenge, please visit the official website (https://sharechat.com/recsys2023). Our code is based on [FuxiCTR](https://github.com/xue-pai/FuxiCTR) and (BARS)[https://openbenchmark.github.io/]. 

***Rank***: 2th at Acadamic Leardboard

***Team Member***: Zhutian Lin(Tsinghua University), [Liangcai Su(Tsinghua University)](https://liangcaisu.github.io/), Xiaoteng Shen(Tsinghua University). 

*Note: The contributions of all of us are equal.*


## Environment Setup 
Please refer to `requirements.txt` for environment installation.

## Project Structure
```
.
├── config              # model config & dataset config
├── data                # dataset
├── enumerate_expid_list.py
├── final_merge.py
├── fuxictr
├── fuxictr_version.py
├── inference.py        # get the submission of a trained model
├── README.md
├── requirements.txt
├── run_expid_list.py
├── run_expid.py
├── run_param_tuner.py
├── src                 # model implementation
├── step_1_run_preprocess_s1.sh
├── step_2_train_models.sh
├── step_3_inference.sh
├── step_4_result_ensemble.sh
└── submission
```
## Steps to Reproduce
In this challenge, we employed multiple MMoE-based multi-task models. Below, we will provide a detailed description of the process for reproducing our competition results.

### Download Dataset 
Download the dataset from the official website and place it in the `data/raw_data` directory. Then, unzip the dataset files.
```
├── raw_data
│   ├── 2a161f8e_1679936280892_sc.zip
│   └── sharechat_recsys2023_data
│       ├── README.txt
│       ├── test
│       └── train
```
### Dataset Preprocess 
``` 
bash step_1_run_preprocess_s1.sh
```

### Model Config & Dataset Config 
Please refer to 'config/dataset_config.yaml' and 'config/model_config.yaml' for more details. 

| Model   	| Online Score 	| Dataset Config 	| Model Config 	    | Other 	|
|---------	|--------------	|----------------	|--------------	    |-------	|
| Model 1 	|   6.244622    |    sharechatx1 	|MMoE_sharechat_x1 	|       	|
| Model 2 	|   6.24423   	|    sharechatx2	|MMoE_sharechat_x2  |       	|
| Model 3 	|   6.269848   	|    sharechatx3	|MMoE_sharechat_x3v1|       	|
| Model 4 	|   6.282074   	|    sharechatx3	|MMoE_sharechat_x3v2|       	|
| Model 5 	|   6.205853   	|    sharechatx3	|MMoE_sharechat_x3v3|       	|
| Model 6 	|   6.21848     |    sharechatx3	|MMoE_sharechat_x3v4|       	|

### Train Models 
```
bash step_2_train_models.sh 
```
### Inference 
```
bash step_3_inference.sh 
```
### Merge Results
```
bash step_4_merge_result.sh 
```

Final Online Score: 6.159096

## Citation 
If you find our code helpful in your research, please kindly cite the following papers.

> Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, Xiuqiang He. [Open Benchmarking for Click-Through Rate Prediction](https://arxiv.org/abs/2009.05794). *The 30th ACM International Conference on Information and Knowledge Management (CIKM)*, 2021. [[Bibtex](https://dblp.org/rec/conf/cikm/ZhuLYZH21.html?view=bibtex)]

> Jieming Zhu, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Xi Xiao, Rui Zhang. [BARS: Towards Open Benchmarking for Recommender Systems](https://arxiv.org/abs/2205.09626). *The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)*, 2022. [[Bibtex](https://dblp.org/rec/conf/sigir/ZhuDSMLCXZ22.html?view=bibtex)]


## Concat 
If you have any questions, please feel free to reach out to Liangcai Su(sulc21@mails.tsinghua.edu.cn). 