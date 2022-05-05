# Cooperative Navigation Control with Hierarchical and Stable MARL

This repository is an implementation of cooperative navigation control based on [Hierarchical and Stable Multiagent Reinforcement
Learning for Cooperative Navigation Control](https://ieeexplore.ieee.org/abstract/document/9466421).

@article{  
  jin2021hierarchical,  
  title={Hierarchical and Stable Multiagent Reinforcement Learning for Cooperative Navigation Control},  
  author={Jin, Yue and Wei, Shuangqing and Yuan, Jian and Zhang, Xudong},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},  
  year={2021},  
  publisher={IEEE}  
}

## Pretrain
Pretrain target selection policy:
 
```
cd pretrain
python pretrain_ts_run.py --agent_num=3 --mode=train --save_path=./result
```

## Train

```
python main.py --agent_num=3 --mode=train --pretrain_path=./pretrain --save_path=./result
```

## Test

```
python main.py --agent_num=3 --mode=test --pretrain_path=./pretrain --load_path=./result
```
