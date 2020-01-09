
# Grid HP search

## NN
learning rate : Adam takes care of it
Batch size ... maybe
n_steps : 10000 makes it converge without overfitting...

python run_array.py --gpu 1 --partition titanic --xp-name NN-grid-search --skip-minuit --model NN --n-steps 10000 15000 20000 

## ANN
learning rate : Adam takes care of it
Batch size ... maybe
n_steps : 10000 makes it converge without overfitting...
n-augment = 1 2 4
width = 1 2 3 

python run_array.py --gpu 1 --partition titanic --xp-name ANN-grid-search --skip-minuit --model ANN --n-steps 10000 --n-augment 1 2 4 --width 1 2 3

## GB
n_estimators : 1000
max_depth : control overfitting : 3 5 8 ?
learning-rate : main HP ! 0.1 0.01 1 0.5 0.05

python run_array.py --gpu 1 --partition titanic --xp-name GB-grid-search --skip-minuit --model GB --n-estimators 1000 --learning-rate 0.1 0.01 1 0.5 0.05 --max-depth 3 5 8

## TP
learning rate : Adam takes care of it
Batch size ... maybe
n_steps : 10000 makes it converge without overfitting...
alpha : maybe but 1e-2 makes it Ok already
trade-off : main HP = 10.0 1.0 0.1 0.01 0.001 0.0

python run_array.py --gpu 1 --partition titanic --xp-name TP-grid-search --skip-minuit --model TP --n-steps 10000 --trade-off 10.0 1.0 0.1 0.01 0.001 0.0

## ATP
learning rate : Adam takes care of it
Batch size ... maybe
n_steps : 10000 makes it converge without overfitting...
alpha : maybe but 1e-2 makes it Ok already
trade-off : main HP = 10.0 1.0 0.1 0.01 0.001 0.0
width = 1 2 3 
n-augment = 1 2 4

python run_array.py --gpu 1 --partition titanic --xp-name ATP-grid-search --skip-minuit --model ATP --n-steps 10000 --trade-off 10.0 1.0 0.1 0.01 0.001 0.0 --n-augment 1 2 4 --width 1 2 3

## PAN
learning rate : Adam takes care of it
Batch size ... maybe
n-clf-pre-training-steps = 3000
n-adv-pre-training-steps = 3000
n_steps : 10000 makes it converge without overfitting...
n-recovery-steps : main HP = 1 5 10 100
trade-off : main HP = 10.0 1.0 0.1 0.01 0.001 0.0
width = 1 2 3 

python run_array.py --gpu 1 --partition titanic --xp-name PAN-grid-search --skip-minuit --model PAN --n-steps 10000 --n-recovery-steps 1 5 10 --trade-off 10.0 1.0 0.1 0.01 0.001 0.0 --width 1 2 3

## APAN
learning rate : Adam takes care of it
Batch size ... maybe
n-clf-pre-training-steps = 3000
n-adv-pre-training-steps = 3000
n_steps : 10000 makes it converge without overfitting...
n-recovery-steps : main HP = 1 5 10 100
trade-off : main HP = 10.0 1.0 0.1 0.01 0.001 0.0
width = 1 2 3 
n-augment = 1 2 4

python run_array.py --gpu 1 --partition titanic --xp-name APAN-grid-search --skip-minuit --model APAN --n-steps 10000 --n-recovery-steps 1 5 10 --trade-off 10.0 1.0 0.1 0.01 0.001 0.0 --n-augment 1 2 4 --width 1 2 3
