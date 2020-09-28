# All_cmd

# TOY 1D
#========

# CALIBRATION / MARGINAL
python run_array.py benchmark.GG.CALIB-Rescale --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name GG-CALIB-Rescale --partition besteffort

python run_array.py benchmark.GG.likelihood --gpu 1 --xp-name GG-likelihood --partition besteffort
python run_array.py benchmark.GG.bayes --gpu 1 --xp-name GG-bayes --partition besteffort

python run_array.py benchmark.GG.REG-Marginal --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Marginal --partition besteffort
python run_array.py benchmark.GG.REG-Marginal --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Marginal-bouncing --partition besteffort


# PRIOR RUN
python run_array.py benchmark.GG.GB-Prior --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Prior --partition besteffort
python run_array.py benchmark.GG.NN-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-NN-Prior --partition besteffort
python run_array.py benchmark.GG.DA-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Prior --partition besteffort

python run_array.py benchmark.GG.PIVOT2-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --n-recovery-steps 1 --xp-name GG-PIVOT2-Prior --partition besteffort

python run_array.py benchmark.GG.PIVOT-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Prior --partition besteffort
python run_array.py benchmark.GG.TP-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Prior --partition besteffort

python run_array.py benchmark.GG.INF-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Prior --partition besteffort

python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Prior --partition besteffort
python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Prior-bouncing --partition besteffort


# CALIB RUN
python run_array.py benchmark.GG.GB-Calib --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Calib --partition besteffort
python run_array.py benchmark.GG.NN-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-NN-Calib --partition besteffort
python run_array.py benchmark.GG.DA-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Calib --partition besteffort

python run_array.py benchmark.GG.PIVOT2-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --n-recovery-steps 1 --xp-name GG-PIVOT2-Calib --partition besteffort

python run_array.py benchmark.GG.PIVOT-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Calib --partition besteffort
python run_array.py benchmark.GG.TP-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Calib --partition besteffort

python run_array.py benchmark.GG.INF-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Calib --partition besteffort

python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Calib --partition besteffort
python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Calib-bouncing --partition besteffort



# === PLOT ===

# CALIBRATION / MARGINAL
python run_array.py benchmark.COMPARE.GG.REG-Marginal --gpu 1 --xp-name COMPARE-GG-REG-Marginal --partition besteffort


# PRIOR RUN
python run_array.py benchmark.COMPARE.GG.GB-Prior --gpu 1 --xp-name COMPARE-GG-GB-Prior --partition besteffort
python run_array.py benchmark.COMPARE.GG.NN-Prior --gpu 1 --xp-name COMPARE-GG-NN-Prior --partition besteffort
python run_array.py benchmark.COMPARE.GG.DA-Prior --gpu 1 --xp-name COMPARE-GG-DA-Prior --partition besteffort

python run_array.py benchmark.COMPARE.GG.PIVOT-Prior --gpu 1 --xp-name COMPARE-GG-PIVOT-Prior --partition besteffort
python run_array.py benchmark.COMPARE.GG.TP-Prior --gpu 1 --xp-name COMPARE-GG-TP-Prior --partition besteffort

python run_array.py benchmark.COMPARE.GG.INF-Prior --gpu 1 --xp-name COMPARE-GG-INF-Prior --partition besteffort

python run_array.py benchmark.COMPARE.GG.REG-Prior --gpu 1 --xp-name COMPARE-GG-REG-Prior --partition besteffort


# CALIB RUN

python run_array.py benchmark.COMPARE.GG.GB-Calib --gpu 1 --xp-name COMPARE-GG-GB-Calib --partition besteffort
python run_array.py benchmark.COMPARE.GG.NN-Calib --gpu 1 --xp-name COMPARE-GG-NN-Calib --partition besteffort
python run_array.py benchmark.COMPARE.GG.DA-Calib --gpu 1 --xp-name COMPARE-GG-DA-Calib --partition besteffort

python run_array.py benchmark.COMPARE.GG.PIVOT-Calib --gpu 1 --xp-name COMPARE-GG-PIVOT-Calib --partition besteffort
python run_array.py benchmark.COMPARE.GG.TP-Calib --gpu 1 --xp-name COMPARE-GG-TP-Calib --partition besteffort

python run_array.py benchmark.COMPARE.GG.INF-Calib --gpu 1 --xp-name COMPARE-GG-INF-Calib --partition besteffort

python run_array.py benchmark.COMPARE.GG.REG-Calib --gpu 1 --xp-name COMPARE-GG-REG-Calib --partition besteffort

# BEST METHODS
python run_array.py benchmark.COMPARE.GG.compare_models --gpu 1 --xp-name COMPARE-GG-MODELS --partition besteffort





