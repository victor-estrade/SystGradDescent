# All_cmd

# TOY 1D
#========
python run_array.py benchmark.GG.CALIB-Rescale --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name GG-CALIB-Rescale

python run_array.py benchmark.GG.GB-Prior --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Prior
python run_array.py benchmark.GG.GB-Calib --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Calib

python run_array.py benchmark.GG.NN-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-NN-Prior
python run_array.py benchmark.GG.NN-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-NN-Calib

python run_array.py benchmark.GG.DA-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Prior
python run_array.py benchmark.GG.DA-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Calib

python run_array.py benchmark.GG.PIVOT-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Prior
python run_array.py benchmark.GG.PIVOT-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Calib

python run_array.py benchmark.GG.TP-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Prior
python run_array.py benchmark.GG.TP-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Calib

python run_array.py benchmark.GG.INF-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Prior
python run_array.py benchmark.GG.INF-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Calib

python run_array.py benchmark.GG.likelihood --gpu 1 --xp-name GG-likelihood
python run_array.py benchmark.GG.likelihood --gpu 1 --xp-name GG-bayes

python run_array.py benchmark.GG.REG-Marginal --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Marginal
python run_array.py benchmark.GG.REG-Marginal --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Marginal-bouncing

python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Prior
python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Prior-bouncing

python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Calib
python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Calib-bouncing

# === PLOT ===
python run_array.py benchmark.COMPARE.GG.GB-Prior --gpu 1 --xp-name COMPARE-GG-GB-Prior
python run_array.py benchmark.COMPARE.GG.GB-Calib --gpu 1 --xp-name COMPARE-GG-GB-Calib

python run_array.py benchmark.COMPARE.GG.NN-Prior --gpu 1 --xp-name COMPARE-GG-NN-Prior
python run_array.py benchmark.COMPARE.GG.NN-Calib --gpu 1 --xp-name COMPARE-GG-NN-Calib

python run_array.py benchmark.COMPARE.GG.DA-Prior --gpu 1 --xp-name COMPARE-GG-DA-Prior
python run_array.py benchmark.COMPARE.GG.DA-Calib --gpu 1 --xp-name COMPARE-GG-DA-Calib

python run_array.py benchmark.COMPARE.GG.PIVOT-Prior --gpu 1 --xp-name COMPARE-GG-PIVOT-Prior
python run_array.py benchmark.COMPARE.GG.PIVOT-Calib --gpu 1 --xp-name COMPARE-GG-PIVOT-Calib

python run_array.py benchmark.COMPARE.GG.TP-Prior --gpu 1 --xp-name COMPARE-GG-TP-Prior
python run_array.py benchmark.COMPARE.GG.TP-Calib --gpu 1 --xp-name COMPARE-GG-TP-Calib

python run_array.py benchmark.COMPARE.GG.INF-Prior --gpu 1 --xp-name COMPARE-GG-INF-Prior
python run_array.py benchmark.COMPARE.GG.INF-Calib --gpu 1 --xp-name COMPARE-GG-INF-Calib

python run_array.py benchmark.COMPARE.GG.REG-Prior --gpu 1 --xp-name COMPARE-GG-REG-Prior
python run_array.py benchmark.COMPARE.GG.REG-Calib --gpu 1 --xp-name COMPARE-GG-REG-Calib

python run_array.py benchmark.COMPARE.GG.REG-Marginal --gpu 1 --xp-name COMPARE-GG-REG-Marginal


