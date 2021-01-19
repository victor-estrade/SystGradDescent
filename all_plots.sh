# All_cmd

# === PLOT ===

# CALIBRATION / MARGINAL
python run_array.py benchmark.COMPARE.GG.REG-Marginal --gpu 1 --xp-name COMPARE-GG-REG-Marginal --partition besteffort


# RUNs
python run_array.py benchmark.COMPARE.GG.GB --gpu 1 --xp-name COMPARE-GG-GB --partition besteffort
python run_array.py benchmark.COMPARE.GG.NN --gpu 1 --xp-name COMPARE-GG-NN --partition besteffort
python run_array.py benchmark.COMPARE.GG.DA --gpu 1 --xp-name COMPARE-GG-DA --partition besteffort

python run_array.py benchmark.COMPARE.GG.PIVOT --gpu 1 --xp-name COMPARE-GG-PIVOT --partition besteffort
python run_array.py benchmark.COMPARE.GG.TP --gpu 1 --xp-name COMPARE-GG-TP --partition besteffort

python run_array.py benchmark.COMPARE.GG.INF --gpu 1 --xp-name COMPARE-GG-INF --partition besteffort

python run_array.py benchmark.COMPARE.GG.REG --gpu 1 --xp-name COMPARE-GG-REG --partition besteffort


# BEST METHODS
python run_array.py benchmark.COMPARE.GG.compare_models --gpu 1 --xp-name COMPARE-GG-MODELS --partition besteffort


# CHEAT RUN
python run_array.py benchmark.COMPARE.GG.REG-Cheat --gpu 1 --xp-name COMPARE-GG-REG-Cheat --partition besteffort





# TOY 3D
#========


# === PLOT ===

# CALIBRATION / MARGINAL
python run_array.py benchmark.COMPARE.S3D2.REG-Marginal --gpu 1 --xp-name COMPARE-S3D2-REG-Marginal --partition besteffort


# RUNS
python run_array.py benchmark.COMPARE.S3D2.GB --gpu 1 --xp-name COMPARE-S3D2-GB --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.NN --gpu 1 --xp-name COMPARE-S3D2-NN --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.DA --gpu 1 --xp-name COMPARE-S3D2-DA --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.PIVOT --gpu 1 --xp-name COMPARE-S3D2-PIVOT --partition besteffort
python run_array.py benchmark.COMPARE.S3D2.TP --gpu 1 --xp-name COMPARE-S3D2-TP --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.INF --gpu 1 --xp-name COMPARE-S3D2-INF --partition besteffort

python run_array.py benchmark.COMPARE.S3D2.REG --gpu 1 --xp-name COMPARE-S3D2-REG --partition besteffort


# BEST METHODS
python run_array.py benchmark.COMPARE.S3D2.compare_models --gpu 1 --xp-name COMPARE-S3D2-MODELS --partition besteffort





# HIGGS
#========


# === PLOT ===

# CALIBRATION / MARGINAL
python run_array.py benchmark.COMPARE.HIGGS.REG-Marginal --gpu 1 --xp-name COMPARE-HIGGS-REG-Marginal --partition besteffort


# RUN
python run_array.py benchmark.COMPARE.HIGGS.FF --gpu 1 --xp-name COMPARE-HIGGS-FF --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.GB --gpu 1 --xp-name COMPARE-HIGGS-GB --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.NN --gpu 1 --xp-name COMPARE-HIGGS-NN --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.DA --gpu 1 --xp-name COMPARE-HIGGS-DA --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.PIVOT --gpu 1 --xp-name COMPARE-HIGGS-PIVOT --partition besteffort
python run_array.py benchmark.COMPARE.HIGGS.TP --gpu 1 --xp-name COMPARE-HIGGS-TP --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.INF --gpu 1 --xp-name COMPARE-HIGGS-INF --partition besteffort

python run_array.py benchmark.COMPARE.HIGGS.REG --gpu 1 --xp-name COMPARE-HIGGS-REG --partition besteffort



# BEST METHODS
python run_array.py benchmark.COMPARE.HIGGS.compare_models --gpu 1 --xp-name COMPARE-HIGGS-MODELS --partition besteffort
