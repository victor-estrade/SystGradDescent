# All_cmd
python run_array.py benchmark.HIGGS.assert_torch_generator --xp-name HIGGS-assert_torch_generator --partition besteffort
python run_array.py benchmark.HIGGS.speedy --xp-name HIGGS-speedy --partition besteffort

# TOY 1D
#========

#  LEANRING CURVE
python run_array.py benchmark.GG.GB_learning_curve --gpu 1 --n-estimators 300  --max-depth 3 --learning-rate 0.1 --xp-name GG-GB_learning_curve --partition besteffort
python run_array.py benchmark.GG.NN_learning_curve --gpu 1 --n-steps 5000 --n-unit 200 --batch-size 1000 --xp-name GG-NN_learning_curve --partition besteffort


# CALIBRATION / MARGINAL
python run_array.py benchmark.GG.CALIB-Rescale --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name GG-CALIB-Rescale --partition besteffort

python run_array.py benchmark.GG.likelihood --gpu 1 --xp-name GG-likelihood --partition besteffort
python run_array.py benchmark.GG.bayes --gpu 1 --xp-name GG-bayes --partition besteffort

python run_array.py benchmark.GG.REG-Marginal --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Marginal --partition besteffort
python run_array.py benchmark.GG.REG-Marginal --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Marginal-bouncing --partition besteffort


# PRIOR RUN
python run_array.py benchmark.GG.GB-Prior --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Prior --partition besteffort
python run_array.py benchmark.GG.NN-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 1000 --xp-name GG-NN-Prior --partition besteffort
python run_array.py benchmark.GG.DA-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Prior --partition besteffort

python run_array.py benchmark.GG.PIVOT-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Prior --partition besteffort
python run_array.py benchmark.GG.TP-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Prior --partition besteffort

python run_array.py benchmark.GG.INF-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Prior --partition besteffort

python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Prior --partition besteffort
python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Prior-bouncing --partition besteffort



# CALIB RUN
python run_array.py benchmark.GG.GB-Calib --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Calib --partition besteffort
python run_array.py benchmark.GG.NN-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 1000 --xp-name GG-NN-Calib --partition besteffort
python run_array.py benchmark.GG.DA-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Calib --partition besteffort

python run_array.py benchmark.GG.PIVOT-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Calib --partition besteffort
python run_array.py benchmark.GG.TP-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Calib --partition besteffort

python run_array.py benchmark.GG.INF-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Calib --partition besteffort

python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Calib --partition besteffort
python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Calib-bouncing --partition besteffort



# PRIOR PLUS

python run_array.py benchmark.GG.REG-Prior-Plus --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Prior-Plus --partition besteffort




# TOY 3D
#========

# CALIBRATION / MARGINAL
python run_array.py benchmark.S3D2.CALIB-LAM --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name S3D2-CALIB-LAM --partition besteffort
python run_array.py benchmark.S3D2.CALIB-R --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name S3D2-CALIB-R --partition besteffort

python run_array.py benchmark.S3D2.likelihood --gpu 1 --xp-name S3D2-likelihood --partition besteffort
python run_array.py benchmark.S3D2.bayes --gpu 1 --xp-name S3D2-bayes --partition besteffort

python run_array.py benchmark.S3D2.REG-Marginal --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 5000 --xp-name S3D2-REG-Marginal --partition besteffort
python run_array.py benchmark.S3D2.REG-Marginal --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name S3D2-REG-Marginal-bouncing --partition besteffort


# PRIOR RUN
python run_array.py benchmark.S3D2.GB-Prior --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name S3D2-GB-Prior --partition besteffort
python run_array.py benchmark.S3D2.NN-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 1000 --xp-name S3D2-NN-Prior --partition besteffort

python run_array.py benchmark.S3D2.DA-Prior --gpu 1 --start-cv 0 --end-cv 6 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Prior --partition besteffort
python run_array.py benchmark.S3D2.DA-Prior --gpu 1 --start-cv 6 --end-cv 12 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Prior --partition besteffort
python run_array.py benchmark.S3D2.DA-Prior --gpu 1 --start-cv 12 --end-cv 18 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Prior --partition besteffort
python run_array.py benchmark.S3D2.DA-Prior --gpu 1 --start-cv 18 --end-cv 24 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Prior --partition besteffort
python run_array.py benchmark.S3D2.DA-Prior --gpu 1 --start-cv 24 --end-cv 30 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Prior --partition besteffort

python run_array.py benchmark.S3D2.PIVOT-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-PIVOT-Prior --partition besteffort
python run_array.py benchmark.S3D2.TP-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-TP-Prior --partition besteffort

python run_array.py benchmark.S3D2.INF-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-INF-Prior --partition besteffort

python run_array.py benchmark.S3D2.REG-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-REG-Prior --partition besteffort
python run_array.py benchmark.S3D2.REG-Prior --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name S3D2-REG-Prior-bouncing --partition besteffort


# CALIB RUN
python run_array.py benchmark.S3D2.GB-Calib --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name S3D2-GB-Calib --partition besteffort
python run_array.py benchmark.S3D2.NN-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 1000 --xp-name S3D2-NN-Calib --partition besteffort
python run_array.py benchmark.S3D2.DA-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Calib --partition besteffort

python run_array.py benchmark.S3D2.PIVOT-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-PIVOT-Calib --partition besteffort
python run_array.py benchmark.S3D2.TP-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-TP-Calib --partition besteffort

python run_array.py benchmark.S3D2.INF-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-INF-Calib --partition besteffort

python run_array.py benchmark.S3D2.REG-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-REG-Calib --partition besteffort
python run_array.py benchmark.S3D2.REG-Calib --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name S3D2-REG-Calib-bouncing --partition besteffort




# HIGGS
#========

# CALIBRATION / MARGINAL
# python run_array.py benchmark.HIGGS.CALIB-LAM --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name HIGGS-CALIB-LAM --partition besteffort
# python run_array.py benchmark.HIGGS.CALIB-R --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name HIGGS-CALIB-R --partition besteffort

# python run_array.py benchmark.HIGGS.likelihood --gpu 1 --xp-name HIGGS-likelihood --partition besteffort
# python run_array.py benchmark.HIGGS.bayes --gpu 1 --xp-name HIGGS-bayes --partition besteffort

# python run_array.py benchmark.HIGGS.REG-Marginal --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 5000 --xp-name HIGGS-REG-Marginal --partition besteffort
# python run_array.py benchmark.HIGGS.REG-Marginal --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name HIGGS-REG-Marginal-bouncing --partition besteffort


# PRIOR RUN
python run_array.py benchmark.HIGGS.GB-Prior --gpu 1 --n-estimators 300  --max-depth 3 --learning-rate 0.1 --xp-name HIGGS-GB-Prior --partition besteffort --estimate-only
# python run_array.py benchmark.HIGGS.GB-Prior --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name HIGGS-GB-Prior --partition besteffort
# python run_array.py benchmark.HIGGS.NN-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 1000 --xp-name HIGGS-NN-Prior --partition besteffort

# python run_array.py benchmark.HIGGS.DA-Prior --gpu 1 --start-cv 0 --end-cv 6 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-DA-Prior --partition besteffort
# python run_array.py benchmark.HIGGS.DA-Prior --gpu 1 --start-cv 6 --end-cv 12 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-DA-Prior --partition besteffort
# python run_array.py benchmark.HIGGS.DA-Prior --gpu 1 --start-cv 12 --end-cv 18 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-DA-Prior --partition besteffort
# python run_array.py benchmark.HIGGS.DA-Prior --gpu 1 --start-cv 18 --end-cv 24 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-DA-Prior --partition besteffort
# python run_array.py benchmark.HIGGS.DA-Prior --gpu 1 --start-cv 24 --end-cv 30 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-DA-Prior --partition besteffort

# python run_array.py benchmark.HIGGS.PIVOT-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HIGGS-PIVOT-Prior --partition besteffort
# python run_array.py benchmark.HIGGS.TP-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HIGGS-TP-Prior --partition besteffort

# python run_array.py benchmark.HIGGS.INF-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-INF-Prior --partition besteffort

# python run_array.py benchmark.HIGGS.REG-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-REG-Prior --partition besteffort
# python run_array.py benchmark.HIGGS.REG-Prior --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name HIGGS-REG-Prior-bouncing --partition besteffort


# CALIB RUN
# python run_array.py benchmark.HIGGS.GB-Calib --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name HIGGS-GB-Calib --partition besteffort
# python run_array.py benchmark.HIGGS.NN-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 1000 --xp-name HIGGS-NN-Calib --partition besteffort
# python run_array.py benchmark.HIGGS.DA-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-DA-Calib --partition besteffort

# python run_array.py benchmark.HIGGS.PIVOT-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HIGGS-PIVOT-Calib --partition besteffort
# python run_array.py benchmark.HIGGS.TP-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HIGGS-TP-Calib --partition besteffort

# python run_array.py benchmark.HIGGS.INF-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-INF-Calib --partition besteffort

# python run_array.py benchmark.HIGGS.REG-Calib --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name HIGGS-REG-Calib --partition besteffort
# python run_array.py benchmark.HIGGS.REG-Calib --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name HIGGS-REG-Calib-bouncing --partition besteffort





# HIGGS TES ONLY
#===============

# PRIOR RUN
python run_array.py benchmark.HIGGSTES.GB-Prior --gpu 1 --n-estimators 300  --max-depth 3 --learning-rate 0.1 --xp-name HIGGSTES-GB-Prior --partition besteffort
python run_array.py benchmark.HIGGSTES.NN-Prior --gpu 1 --n-steps 5000 --n-unit 500 --batch-size 10000 --xp-name HIGGSTES-NN-Prior --partition besteffort
python run_array.py benchmark.HIGGSTES.DA-Prior --gpu 1 --n-steps 5000 --n-unit 500 --batch-size 10000 --xp-name HIGGSTES-DA-Prior --partition besteffort

python run_array.py benchmark.HIGGSTES.INF-Prior --gpu 1 --n-steps 5000 --n-unit 500 --sample-size 10000 --xp-name HIGGSTES-INF-Calib --partition besteffort

python run_array.py benchmark.HIGGSTES.REG-Prior --gpu 1 --n-steps 5000 --n-unit 500 --sample-size 10000 --xp-name HIGGSTES-REG-Prior --partition besteffort



# OTHERS
#=========

python run_array.py benchmark.S3D2.GB-Prior --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name S3D2-GB-Prior --partition besteffort
