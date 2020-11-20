# All_cmd

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


# CHEAT RUN
python run_array.py benchmark.COMPARE.GG.REG-Cheat --gpu 1 --xp-name COMPARE-GG-REG-Cheat --partition besteffort










# TOY 3D
#========


# === PLOT ===

# CALIBRATION / MARGINAL
python run_array.py benchmark.COMPARE.S3D2.REG-Marginal --gpu 1 --xp-name COMPARE-S3D2-REG-Marginal --partition besteffort


# PRIOR RUN
python run_array.py benchmark.COMPARE.S3D2.GB-Prior --gpu 1 --xp-name COMPARE-S3D2-GB-Prior --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.NN-Prior --gpu 1 --xp-name COMPARE-S3D2-NN-Prior --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.DA-Prior --gpu 1 --xp-name COMPARE-S3D2-DA-Prior --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.PIVOT-Prior --gpu 1 --xp-name COMPARE-S3D2-PIVOT-Prior --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.TP-Prior --gpu 1 --xp-name COMPARE-S3D2-TP-Prior --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.INF-Prior --gpu 1 --xp-name COMPARE-S3D2-INF-Prior --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.REG-Prior --gpu 1 --xp-name COMPARE-S3D2-REG-Prior --partition besteffort


# CALIB RUN

python run_array.py benchmark.COMPARE.S3D2.GB-Calib --gpu 1 --xp-name COMPARE-S3D2-GB-Calib --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.NN-Calib --gpu 1 --xp-name COMPARE-S3D2-NN-Calib --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.DA-Calib --gpu 1 --xp-name COMPARE-S3D2-DA-Calib --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.PIVOT-Calib --gpu 1 --xp-name COMPARE-S3D2-PIVOT-Calib --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.TP-Calib --gpu 1 --xp-name COMPARE-S3D2-TP-Calib --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.INF-Calib --gpu 1 --xp-name COMPARE-S3D2-INF-Calib --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.REG-Calib --gpu 1 --xp-name COMPARE-S3D2-REG-Calib --partition besteffort

# BEST METHODS
python run_array.py benchmark.COMPARE.S3D2.compare_models --gpu 1 --xp-name COMPARE-S3D2-MODELS --partition besteffort





# HIGGS
#========


# === PLOT ===

# CALIBRATION / MARGINAL
python run_array.py benchmark.COMPARE.HIGGS.REG-Marginal --gpu 1 --xp-name COMPARE-HIGGS-REG-Marginal --partition besteffort


# PRIOR RUN
python run_array.py benchmark.COMPARE.HIGGS.GB-Prior --gpu 1 --xp-name COMPARE-HIGGS-GB-Prior --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.NN-Prior --gpu 1 --xp-name COMPARE-HIGGS-NN-Prior --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.DA-Prior --gpu 1 --xp-name COMPARE-HIGGS-DA-Prior --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.PIVOT-Prior --gpu 1 --xp-name COMPARE-HIGGS-PIVOT-Prior --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.TP-Prior --gpu 1 --xp-name COMPARE-HIGGS-TP-Prior --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.INF-Prior --gpu 1 --xp-name COMPARE-HIGGS-INF-Prior --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.REG-Prior --gpu 1 --xp-name COMPARE-HIGGS-REG-Prior --partition besteffort


# CALIB RUN

python run_array.py benchmark.COMPARE.HIGGS.GB-Calib --gpu 1 --xp-name COMPARE-HIGGS-GB-Calib --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.NN-Calib --gpu 1 --xp-name COMPARE-HIGGS-NN-Calib --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.DA-Calib --gpu 1 --xp-name COMPARE-HIGGS-DA-Calib --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.PIVOT-Calib --gpu 1 --xp-name COMPARE-HIGGS-PIVOT-Calib --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.TP-Calib --gpu 1 --xp-name COMPARE-HIGGS-TP-Calib --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.INF-Calib --gpu 1 --xp-name COMPARE-HIGGS-INF-Calib --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.REG-Calib --gpu 1 --xp-name COMPARE-HIGGS-REG-Calib --partition besteffort

# BEST METHODS
python run_array.py benchmark.COMPARE.HIGGS.compare_models --gpu 1 --xp-name COMPARE-HIGGS-MODELS --partition besteffort
