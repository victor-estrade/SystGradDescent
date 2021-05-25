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
python run_array.py benchmark.GG.FF-Prior --gpu 1 --feature-id 0 --xp-name GG-FF-Prior --partition besteffort --estimate-only
python run_array.py benchmark.GG.GB-Prior --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Prior --partition besteffort --estimate-only
python run_array.py benchmark.GG.NN-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name GG-NN-Prior --partition besteffort --estimate-only
python run_array.py benchmark.GG.DA-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Prior --partition besteffort --estimate-only

python run_array.py benchmark.GG.PIVOT-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Prior --partition besteffort --estimate-only
python run_array.py benchmark.GG.TP-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Prior --partition besteffort --estimate-only

python run_array.py benchmark.GG.INF-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Prior --partition besteffort --estimate-only

python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Prior --partition besteffort --estimate-only
python run_array.py benchmark.GG.REG-Prior --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Prior-bouncing --partition besteffort --estimate-only



# CALIB RUN
python run_array.py benchmark.GG.FF-Calib --gpu 1 --feature-id 0 --xp-name GG-FF-Calib --partition besteffort --estimate-only
python run_array.py benchmark.GG.GB-Calib --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name GG-GB-Calib --partition besteffort --estimate-only
python run_array.py benchmark.GG.NN-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name GG-NN-Calib --partition besteffort --estimate-only
python run_array.py benchmark.GG.DA-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-DA-Calib --partition besteffort --estimate-only

python run_array.py benchmark.GG.PIVOT-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-PIVOT-Calib --partition besteffort --estimate-only
python run_array.py benchmark.GG.TP-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name GG-TP-Calib --partition besteffort --estimate-only

python run_array.py benchmark.GG.INF-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-INF-Calib --partition besteffort --estimate-only

python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Calib --partition besteffort --estimate-only
python run_array.py benchmark.GG.REG-Calib --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name GG-REG-Calib-bouncing --partition besteffort --estimate-only



# PRIOR PLUS

python run_array.py benchmark.GG.REG-Prior-Plus --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name GG-REG-Prior-Plus --partition besteffort

# VAR RUN
python run_array.py benchmark.VAR.GG.GB --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name VAR-GG-GB --partition besteffort
python run_array.py benchmark.VAR.GG.NN --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name VAR-GG-NN --partition besteffort
python run_array.py benchmark.VAR.GG.DA --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name VAR-GG-DA --partition besteffort

python run_array.py benchmark.VAR.GG.PIVOT --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name VAR-GG-PIVOT --partition besteffort
python run_array.py benchmark.VAR.GG.TP --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name VAR-GG-TP --partition besteffort


# AMS RUN
python run_array.py benchmark.AMS.GG.GB --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name AMS-GG-GB --partition besteffort
python run_array.py benchmark.AMS.GG.NN --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name AMS-GG-NN --partition besteffort
python run_array.py benchmark.AMS.GG.DA --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name AMS-GG-DA --partition besteffort

python run_array.py benchmark.AMS.GG.PIVOT --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name AMS-GG-PIVOT --partition besteffort
python run_array.py benchmark.AMS.GG.TP --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name AMS-GG-TP --partition besteffort





# TOY 1D HARD MODE
#=================

#  LEANRING CURVE
python run_array.py benchmark.HARDGG.GB_learning_curve --gpu 1 --n-estimators 300  --max-depth 3 --learning-rate 0.1 --xp-name HARDGG-GB_learning_curve --partition besteffort
python run_array.py benchmark.HARDGG.NN_learning_curve --gpu 1 --n-steps 5000 --n-unit 200 --batch-size 1000 --xp-name HARDGG-NN_learning_curve --partition besteffort


# CALIBRATION / MARGINAL
python run_array.py benchmark.HARDGG.CALIB-Rescale --gpu 1  --n-steps 1000 2000 5000 --n-unit 80 200 --xp-name HARDGG-CALIB-Rescale --partition besteffort

python run_array.py benchmark.HARDGG.likelihood --gpu 1 --xp-name HARDGG-likelihood --partition besteffort
python run_array.py benchmark.HARDGG.bayes --gpu 1 --xp-name HARDGG-bayes --partition besteffort

python run_array.py benchmark.HARDGG.REG-Marginal --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-REG-Marginal --partition besteffort
python run_array.py benchmark.HARDGG.REG-Marginal --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name HARDGG-REG-Marginal-bouncing --partition besteffort


# PRIOR RUN
python run_array.py benchmark.HARDGG.FF-Prior --gpu 1 --feature-id 0 --xp-name HARDGG-FF-Prior --partition besteffort
python run_array.py benchmark.HARDGG.GB-Prior --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name HARDGG-GB-Prior --partition besteffort
python run_array.py benchmark.HARDGG.NN-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name HARDGG-NN-Prior --partition besteffort
python run_array.py benchmark.HARDGG.DA-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-DA-Prior --partition besteffort

python run_array.py benchmark.HARDGG.PIVOT-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HARDGG-PIVOT-Prior --partition besteffort
python run_array.py benchmark.HARDGG.TP-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HARDGG-TP-Prior --partition besteffort

python run_array.py benchmark.HARDGG.INF-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-INF-Prior --partition besteffort

python run_array.py benchmark.HARDGG.REG-Prior --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-REG-Prior --partition besteffort
python run_array.py benchmark.HARDGG.REG-Prior --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name HARDGG-REG-Prior-bouncing --partition besteffort



# CALIB RUN
python run_array.py benchmark.HARDGG.GB-Calib --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name HARDGG-GB-Calib --partition besteffort
python run_array.py benchmark.HARDGG.NN-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name HARDGG-NN-Calib --partition besteffort
python run_array.py benchmark.HARDGG.DA-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-DA-Calib --partition besteffort

python run_array.py benchmark.HARDGG.PIVOT-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HARDGG-PIVOT-Calib --partition besteffort
python run_array.py benchmark.HARDGG.TP-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name HARDGG-TP-Calib --partition besteffort

python run_array.py benchmark.HARDGG.INF-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-INF-Calib --partition besteffort

python run_array.py benchmark.HARDGG.REG-Calib --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-REG-Calib --partition besteffort
python run_array.py benchmark.HARDGG.REG-Calib --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name HARDGG-REG-Calib-bouncing --partition besteffort



# PRIOR PLUS

python run_array.py benchmark.HARDGG.REG-Prior-Plus --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name HARDGG-REG-Prior-Plus --partition besteffort

# VAR RUN
python run_array.py benchmark.VAR.HARDGG.GB --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name VAR-HARDGG-GB --partition besteffort
python run_array.py benchmark.VAR.HARDGG.NN --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name VAR-HARDGG-NN --partition besteffort
python run_array.py benchmark.VAR.HARDGG.DA --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name VAR-HARDGG-DA --partition besteffort

python run_array.py benchmark.VAR.HARDGG.PIVOT --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name VAR-HARDGG-PIVOT --partition besteffort
python run_array.py benchmark.VAR.HARDGG.TP --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name VAR-HARDGG-TP --partition besteffort


# AMS RUN
python run_array.py benchmark.AMS.HARDGG.GB --gpu 1 --n-estimators 100 300 1000  --max-depth 3 5 10 --learning-rate 0.1 0.05 0.01 --xp-name AMS-HARDGG-GB --partition besteffort
python run_array.py benchmark.AMS.HARDGG.NN --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --batch-size 20 --xp-name AMS-HARDGG-NN --partition besteffort
python run_array.py benchmark.AMS.HARDGG.DA --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --xp-name AMS-HARDGG-DA --partition besteffort

python run_array.py benchmark.AMS.HARDGG.PIVOT --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name AMS-HARDGG-PIVOT --partition besteffort
python run_array.py benchmark.AMS.HARDGG.TP --gpu 1 --n-steps 2000 5000 --n-unit 50 100 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name AMS-HARDGG-TP --partition besteffort



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
python run_array.py benchmark.S3D2.GB-Prior --retrain --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name S3D2-GB-Prior --partition besteffort --estimate-only
python run_array.py benchmark.S3D2.NN-Prior --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 1000 --xp-name S3D2-NN-Prior --partition besteffort --estimate-only

python run_array.py benchmark.S3D2.DA-Prior --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Prior --partition besteffort --estimate-only

python run_array.py benchmark.S3D2.PIVOT-Prior --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-PIVOT-Prior --partition besteffort --estimate-only
python run_array.py benchmark.S3D2.TP-Prior --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-TP-Prior --partition besteffort --estimate-only

python run_array.py benchmark.S3D2.INF-Prior --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-INF-Prior --partition besteffort --estimate-only

python run_array.py benchmark.S3D2.REG-Prior --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-REG-Prior --partition besteffort --estimate-only
python run_array.py benchmark.S3D2.REG-Prior --retrain --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name S3D2-REG-Prior-bouncing --partition besteffort --estimate-only


# CALIB RUN
python run_array.py benchmark.S3D2.GB-Calib --retrain --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name S3D2-GB-Calib --partition besteffort --estimate-only
python run_array.py benchmark.S3D2.NN-Calib --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 1000 --xp-name S3D2-NN-Calib --partition besteffort --estimate-only
python run_array.py benchmark.S3D2.DA-Calib --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-DA-Calib --partition besteffort --estimate-only

python run_array.py benchmark.S3D2.PIVOT-Calib --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-PIVOT-Calib --partition besteffort --estimate-only
python run_array.py benchmark.S3D2.TP-Calib --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 1e-2 1e-3 --xp-name S3D2-TP-Calib --partition besteffort --estimate-only

python run_array.py benchmark.S3D2.INF-Calib --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-INF-Calib --partition besteffort --estimate-only

python run_array.py benchmark.S3D2.REG-Calib --retrain --gpu 1 --n-steps 5000 --n-unit 200 500 --xp-name S3D2-REG-Calib --partition besteffort --estimate-only
python run_array.py benchmark.S3D2.REG-Calib --retrain --gpu 1 --n-steps 2000 --n-unit 200 --beta1 0.9 --beta2 0.999 --xp-name S3D2-REG-Calib-bouncing --partition besteffort --estimate-only




# HIGGS
#========

# CALIBRATION / MARGINAL
python run_array.py benchmark.HIGGS.CALIB-TES --retrain --gpu 1 --n-steps 5000 --n-unit 100 300 500 --sample-size 10000 --xp-name HIGGS-CALIB-TES --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.CALIB-JES --retrain --gpu 1 --n-steps 5000 --n-unit 100 300 500 --sample-size 10000 --xp-name HIGGS-CALIB-JES --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.CALIB-LES --retrain --gpu 1 --n-steps 5000 --n-unit 100 300 500 --sample-size 10000 --xp-name HIGGS-CALIB-LES --partition besteffort --estimate-only

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







# EASY HIGGS
#===========

# CALIBRATION / MARGINAL

# PRIOR RUN


# CALIB RUN






# BALANCED HIGGS
#===============

# CALIBRATION / MARGINAL

# PRIOR RUN


# CALIB RUN








# HIGGS TES ONLY
#===============

# CALIBRATION / MARGINAL
python run_array.py benchmark.HIGGS.CALIB-TES --retrain --gpu 1 --n-steps 15000 25000 --n-unit 100 300 500 --sample-size 10000 --xp-name HIGGSTES-CALIB-TES --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.CALIB-JES --retrain --gpu 1 --n-steps 15000 25000 --n-unit 100 300 500 --sample-size 10000 --xp-name HIGGSTES-CALIB-JES --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.CALIB-LES --retrain --gpu 1 --n-steps 15000 25000 --n-unit 100 300 500 --sample-size 10000 --xp-name HIGGSTES-CALIB-LES --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.REG-Marginal --gpu 1 --n-steps 15000 25000 --n-unit 200 500 --sample-size 10000 50000 --xp-name HIGGSTES-REG-Marginal --partition besteffort --estimate-only

# PRIOR RUN
python run_array.py benchmark.HIGGS.FF-Prior --gpu 1 --tolerance 100 --feature-id 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --xp-name HIGGSTES-FF-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.GB-Prior --retrain --gpu 1 --tolerance 100 --n-estimators 300 800  --max-depth 3 6 --learning-rate 0.1 0.01 --xp-name HIGGSTES-GB-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.NN-Prior --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --batch-size 10000 --xp-name HIGGSTES-NN-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.DA-Prior --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --batch-size 10000 --xp-name HIGGSTES-DA-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.PIVOT-Prior --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --trade-off 1 0.1 --xp-name HIGGSTES-PIVOT-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.TP-Prior --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --trade-off 1 0.1 --batch-size 200 --xp-name HIGGSTES-TP-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.INF-Prior --retrain --gpu 1 --tolerance 100 --n-steps 2000 --n-unit 200 500 --sample-size 10000 --xp-name HIGGSTES-INF-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.REG-Prior --retrain --gpu 1 --n-steps 15000 25000 --n-unit 200 500 --sample-size 10000 50000 --xp-name HIGGSTES-REG-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.FREG-Prior --retrain --gpu 1 --n-steps 15000 25000 --n-unit 200 500 --sample-size 10000 50000 --xp-name HIGGSTES-FREG-Prior --partition besteffort --estimate-only

# CALIB RUN
python run_array.py benchmark.HIGGS.FF-Calib --retrain --gpu 1 --tolerance 100 --feature-id 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --xp-name HIGGSTES-FF-Calib --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.GB-Calib --retrain --gpu 1 --tolerance 100 --n-estimators 300 800  --max-depth 3 6 --learning-rate 0.1 0.01 --xp-name HIGGSTES-GB-Calib --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.NN-Calib --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --batch-size 10000 --xp-name HIGGSTES-NN-Calib --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.DA-Calib --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --batch-size 10000 --xp-name HIGGSTES-DA-Calib --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.PIVOT-Calib --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --trade-off 1 0.1 --xp-name HIGGSTES-PIVOT-Calib --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.TP-Calib --retrain --gpu 1 --tolerance 100 --n-steps 15000 25000 --n-unit 200 500 --batch-size 200 --xp-name HIGGSTES-TP-Calib --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.INF-Calib --retrain --gpu 1 --tolerance 100 --n-steps 2000 --n-unit 200 500 --sample-size 10000 --xp-name HIGGSTES-INF-Calib --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.REG-Calib --retrain --gpu 1 --n-steps 15000 25000 --n-unit 200 500 --sample-size 10000 50000 --xp-name HIGGSTES-REG-Calib --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.FREG-Calib --retrain --gpu 1 --n-steps 15000 25000 --n-unit 200 500 --sample-size 10000 50000 --xp-name HIGGSTES-FREG-Calib --partition besteffort --estimate-only


# VAR RUN
python run_array.py benchmark.VAR.HIGGSTES.FF --gpu 1 --feature-id 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --xp-name VAR-HIGGSTES-FF --partition besteffort --estimate-only
python run_array.py benchmark.VAR.HIGGSTES.GB --gpu 1 --n-estimators 300 800  --max-depth 3 6 --learning-rate 0.1 0.01 --xp-name VAR-HIGGSTES-GB --partition besteffort --end-cv 5
python run_array.py benchmark.VAR.HIGGSTES.NN --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name VAR-HIGGSTES-NN --partition besteffort --end-cv 5
python run_array.py benchmark.VAR.HIGGSTES.DA --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name VAR-HIGGSTES-DA --partition besteffort --end-cv 5

python run_array.py benchmark.VAR.HIGGSTES.PIVOT --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 --xp-name VAR-HIGGSTES-PIVOT --partition besteffort --end-cv 5
python run_array.py benchmark.VAR.HIGGSTES.TP --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 200 --xp-name VAR-HIGGSTES-TP --partition besteffort --end-cv 5


# AMS RUN
python run_array.py benchmark.AMS.HIGGSTES.FF --gpu 1 --feature-id 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --xp-name AMS-HIGGSTES-FF --partition besteffort --estimate-only
python run_array.py benchmark.AMS.HIGGSTES.GB --gpu 1 --n-estimators 300 800  --max-depth 3 6 --learning-rate 0.1 0.01 --xp-name AMS-HIGGSTES-GB --partition besteffort --end-cv 5
python run_array.py benchmark.AMS.HIGGSTES.NN --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name AMS-HIGGSTES-NN --partition besteffort --end-cv 5
python run_array.py benchmark.AMS.HIGGSTES.DA --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name AMS-HIGGSTES-DA --partition besteffort --end-cv 5

python run_array.py benchmark.AMS.HIGGSTES.PIVOT --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 --xp-name AMS-HIGGSTES-PIVOT --partition besteffort --end-cv 5
python run_array.py benchmark.AMS.HIGGSTES.TP --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 200 --xp-name AMS-HIGGSTES-TP --partition besteffort --end-cv 5





# EASY HIGGS TES ONLY
#====================

# CALIBRATION / MARGINAL
python run_array.py benchmark.HIGGS.REG-Marginal --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 10000 50000 --xp-name EASY_HIGGSTES-REG-Marginal --partition besteffort --estimate-only

# PRIOR RUN
python run_array.py benchmark.HIGGS.FF-Prior --gpu 1 --feature-id 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --xp-name EASY_HIGGSTES-FF-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.GB-Prior --gpu 1 --n-estimators 300 800  --max-depth 3 6 --learning-rate 0.1 0.01 --xp-name EASY_HIGGSTES-GB-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.NN-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name EASY_HIGGSTES-NN-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.DA-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name EASY_HIGGSTES-DA-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.PIVOT-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 --xp-name EASY_HIGGSTES-PIVOT-Prior --partition besteffort
python run_array.py benchmark.HIGGS.TP-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 200 --xp-name EASY_HIGGSTES-TP-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.INF-Prior --gpu 1 --n-steps 2000 --n-unit 200 500 --sample-size 10000 --xp-name EASY_HIGGSTES-INF-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.REG-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 10000 50000 --xp-name EASY_HIGGSTES-REG-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.FREG-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 10000 50000 --xp-name EASY_HIGGSTES-FREG-Prior --partition besteffort --estimate-only


# CALIB RUN







# BALANCED HIGGS TES ONLY
#========================

# CALIBRATION / MARGINAL
python run_array.py benchmark.HIGGS.REG-Marginal --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 10000 50000 --xp-name BALANCED_HIGGSTES-REG-Marginal --partition besteffort --estimate-only

# PRIOR RUN
python run_array.py benchmark.HIGGS.FF-Prior --gpu 1 --feature-id 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --xp-name BALANCED_HIGGSTES-FF-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.GB-Prior --gpu 1 --n-estimators 300 800  --max-depth 3 6 --learning-rate 0.1 0.01 --xp-name BALANCED_HIGGSTES-GB-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.NN-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name BALANCED_HIGGSTES-NN-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.DA-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 10000 --xp-name BALANCED_HIGGSTES-DA-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.PIVOT-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --trade-off 1 0.1 --xp-name BALANCED_HIGGSTES-PIVOT-Prior --partition besteffort
python run_array.py benchmark.HIGGS.TP-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --batch-size 200 --xp-name BALANCED_HIGGSTES-TP-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.INF-Prior --gpu 1 --n-steps 2000 --n-unit 200 500 --sample-size 10000 --xp-name BALANCED_HIGGSTES-INF-Prior --partition besteffort --estimate-only

python run_array.py benchmark.HIGGS.REG-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 10000 50000 --xp-name BALANCED_HIGGSTES-REG-Prior --partition besteffort --estimate-only
python run_array.py benchmark.HIGGS.FREG-Prior --gpu 1 --n-steps 5000 --n-unit 200 500 --sample-size 10000 50000 --xp-name BALANCED_HIGGSTES-FREG-Prior --partition besteffort --estimate-only

# CALIB RUN





# OTHERS
#=========

python run_array.py benchmark.S3D2.GB-Prior --gpu 1 --n-estimators 300 1000  --max-depth 3 5 --learning-rate 0.1 0.05 0.01 --xp-name S3D2-GB-Prior --partition besteffort
