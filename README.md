# Neural Robust Control for Multi-drone Slung Payload Manipulation with Control Contraction Metrics
Pytorch and Matlab implementation of the ACC2026 paper "Neural Robust Control for Multi-drone Slung Payload Manipulation with Control Contraction Metrics", by Xinyuan Liang, Longhao Qian, Yi Lok Lo and Hugh H.T. Liu

## Acknowledgement
The codes are developed based on the CoRL'20 paper "[Learning Certified Control Using Contraction Metric](https://arxiv.org/abs/2011.12569)", by Dawei Sun, Susmit Jha, and Chuchu Fan.

## System dynamics
...

## Requirements
Dependencies include ```torch```, ```tqdm```, ```numpy```, and ```matplotlib```. You can install them using the following command.
```bash
pip install -r requirements.txt
```

## Usage
The script ```main.py``` can be used for learning the controller. U

Optional arguments:
```
  --bs BS               Batch size.
  --num_train NUM_TRAIN
                        Number of samples for training.
  --num_test NUM_TEST   Number of samples for testing.
  --lr LEARNING_RATE    Base learning rate.
  --epochs EPOCHS       Number of training epochs.
  --lr_step LR_STEP
  --lambda _LAMBDA      Convergence rate: lambda
  --w_ub W_UB           Upper bound of the eigenvalue of the dual metric.
  --w_lb W_LB           Lower bound of the eigenvalue of the dual metric.
  --log LOG             Path to a directory for storing the log.
```

Run the following command to learn a controller for the 3 drone slung payload system.
```
python3 main.py
```

Run the following command to evaluate the learned controller and plot the results.
```
python3 plot.py
```
It produces plots and csv files...


## Matlab visualization
...
Simulation package
save csv files as .mat
plot 3D plots with drone figures in ...
plot UDE performance in ...

cite longhao...