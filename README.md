# WindowMixer 

This is the official implementation of paper "WindowMixe:Intra-Window and Inter-Window Modeling for Time Series Forecasting". WindowMixer stems from a simple observation: Time series are recorded in a continuous manner, and the information at any given time point relies on the preceding and succeeding time points for a complete representation.

## Main Experiment
![image](https://github.com/user-attachments/assets/abf3ea74-b451-45cc-9d8c-c76fa4b56cda)

## Start
1. Install Python 3.10. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing). Then place the downloaded data in the folder`./dataset`.

3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/WidowMixer_ETTh1.sh
```

## Contact
If you have any questions or suggestions, feel free to contact our maintenance team:
- Quangao Liu (liuquangao@sia.cn)
- Ruiqi Li (liruiqi1@sia.cn)
- Maowei Jiang (jiangmaowei@sia.cn)

Or describe it in Issues.


## Citation

If you find this repo useful, please cite our paper
```
@inproceedings{liu2024windowmixer,
  title={WindowMixe:Intra-Window and Inter-Window Modeling for Time Series Forecasting},
  author={Quangao Liu and Ruiqi Li and Maowei Jiang and Wei Yang and Cheng Liang and Zhuozhang Zou},
  year={2024},
}
```
## Acknowledgement

Our code is based on Time Series Library (TSLib)：https://github.com/thuml/Time-Series-Library

