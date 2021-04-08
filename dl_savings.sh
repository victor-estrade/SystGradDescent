# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/savings/ ./savings
# scp -r titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/Playground.ipynb ./
# scp titanic:/home/tao/vestrade/workspace/SystML/INFERNO/paper-inferno/code/*.ipynb ./
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/savings/EASYHIGGS-marginal ./savings/EASYHIGGS-marginal
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/ ./OUTPUT


rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/MODELS/ ./OUTPUT/MODELS
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/COMPARE/ ./OUTPUT/COMPARE

rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG/ ./OUTPUT/GG
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-calib/ ./OUTPUT/GG-calib
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/ ./OUTPUT/GG-prior
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-marginal/ ./OUTPUT/GG-marginal

rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HARDGG-prior/ ./OUTPUT/HARDGG-prior
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HARDGG-calib/ ./OUTPUT/HARDGG-calib

rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2/ ./OUTPUT/S3D2
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-calib/ ./OUTPUT/S3D2-calib
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-prior/ ./OUTPUT/S3D2-prior
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-marginal/ ./OUTPUT/S3D2-marginal

rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/BALANCEDHIGGSTES-marginal/ ./OUTPUT/BALANCEDHIGGSTES-marginal
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/BALANCEDHIGGSTES-prior/ ./OUTPUT/BALANCEDHIGGSTES-prior
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/EASYHIGGSTES-marginal/ ./OUTPUT/EASYHIGGSTES-marginal
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/EASYHIGGSTES-prior/ ./OUTPUT/EASYHIGGSTES-prior
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HIGGSTES-marginal/ ./OUTPUT/HIGGSTES-marginal
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HIGGSTES-prior/ ./OUTPUT/HIGGSTES-prior
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HIGGSTES-calib/ ./OUTPUT/HIGGSTES-calib

rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HIGGSTES-prior-10.0/ ./OUTPUT/HIGGSTES-prior-10.0
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HIGGSTES-prior-100.0/ ./OUTPUT/HIGGSTES-prior-100.0
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HIGGSTES-prior-1000.0/ ./OUTPUT/HIGGSTES-prior-1000.0


rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/HIGGS-prior/ ./OUTPUT/HIGGS-prior

rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/VAR-GG/ ./OUTPUT/VAR-GG
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/VAR-HIGGSTES/ ./OUTPUT/VAR-HIGGSTES


# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/PivotClassifier2/ ./OUTPUT/GG-prior/PivotClassifier2
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-calib/Calib_lam/ ./OUTPUT/S3D2-calib/Calib_lam/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-calib/Calib_r/ ./OUTPUT/S3D2-calib/Calib_r/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-prior/PivotClassifier/ ./OUTPUT/S3D2-prior/PivotClassifier/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/MODELS/S3D2/PivotClassifier/ ./OUTPUT/MODELS/S3D2/PivotClassifier/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/MODELS/S3D2/Calib_lam ./OUTPUT/MODELS/S3D2/Calib_lam/

# rsync -rtv --include='valid_roc_concat.png' --include='*/' --exclude='*' titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/PivotClassifier/ ./OUTPUT/GG-prior/PivotClassifier


# rsync -rtv ./hessian titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/hessian/


# rsync -rtv "titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/VAR-HIGGSTES/NeuralNetClassifier/NeuralNetClassifier-L4x500-Adam-0.001-\(0.9-0.999\)-5000-10000/threshold.csv" OUTPUT/VAR-HIGGSTES/NeuralNetClassifier/NeuralNetClassifier-L4x500-Adam-0.001-\(0.9-0.999\)-5000-10000/threshold.csv
