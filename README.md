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

## Dual training of CCM and controller
The script ```main.py``` can be used for learning the controller. U

Optional arguments:

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


Run the following command to learn a controller for the 3 drone slung payload system.
```
python3 main.py
```

## Simulation of closed-loop system
Run the following command to evaluate the learned controller and plot the results.
```
python3 plot.py
```

Things to change:
trajectory type
plot type
plot dimensions
number of trajectories
Disturbance switch
sigma
UDE switch
attitude tracker switch

It produces plots and csv files...
plots in results/plots...
csv in results/csvs...
con_i.csv, sim_i.csv, ude_i.csv
i being the i^th simulated trajectory

## Matlab visualization and plots
...
Simulation package 
cite longhao

Open examples/quadrotor slung payload model/quadrotorPayload.m


save csv files as .mat using save_files.m
Change struct varaible name and ... in
(photo) 

Set struct variable in quadrotor.m
(photo)
run quadrotorPayload.m to plot

3D plot with drone figures in figure(1)
UDE performance plot in figure(2)

For further instruction on other visualization, refer to ...