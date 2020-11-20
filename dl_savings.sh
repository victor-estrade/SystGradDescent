# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/savings/ ./savings
# scp -r titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/Playground.ipynb ./
# scp titanic:/home/tao/vestrade/workspace/SystML/INFERNO/paper-inferno/code/*.ipynb ./
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/savings/EASYHIGGS-marginal ./savings/EASYHIGGS-marginal
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/ ./OUTPUT



rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/MODELS/ ./OUTPUT/MODELS
rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/COMPARE/ ./OUTPUT/COMPARE

rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG/learning_curve/ ./OUTPUT/GG/learning_curve

mkdir -p "OUTPUT/GG-prior/DataAugmentation/DataAugmentation-L4x200-Adam-0.001-(0.9-0.999)-2000-20/cv_0/"
rsync -rtv "titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/DataAugmentation/DataAugmentation-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_summaries.png" OUTPUT/GG-prior/DataAugmentation/DataAugmentation-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_summaries.png
rsync -rtv "titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/DataAugmentation/DataAugmentation-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_distrib.png" OUTPUT/GG-prior/DataAugmentation/DataAugmentation-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_distrib.png
mkdir -p "./OUTPUT/GG-prior/NeuralNetClassifier/NeuralNetClassifier-L4x200-Adam-0.001-(0.9-0.999)-2000-20/cv_0"
rsync -rtv "titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/NeuralNetClassifier/NeuralNetClassifier-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_summaries.png" ./OUTPUT/GG-prior/NeuralNetClassifier/NeuralNetClassifier-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_summaries.png
rsync -rtv "titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/NeuralNetClassifier/NeuralNetClassifier-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_distrib.png" ./OUTPUT/GG-prior/NeuralNetClassifier/NeuralNetClassifier-L4x200-Adam-0.001-\(0.9-0.999\)-2000-20/cv_0/valid_distrib.png




# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/PivotClassifier2/ ./OUTPUT/GG-prior/PivotClassifier2
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-calib/Calib_lam/ ./OUTPUT/S3D2-calib/Calib_lam/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-calib/Calib_r/ ./OUTPUT/S3D2-calib/Calib_r/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/S3D2-prior/PivotClassifier/ ./OUTPUT/S3D2-prior/PivotClassifier/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/MODELS/S3D2/PivotClassifier/ ./OUTPUT/MODELS/S3D2/PivotClassifier/
# rsync -rtv titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/MODELS/S3D2/Calib_lam ./OUTPUT/MODELS/S3D2/Calib_lam/

# rsync -rtv --include='valid_roc_concat.png' --include='*/' --exclude='*' titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/GG-prior/PivotClassifier/ ./OUTPUT/GG-prior/PivotClassifier


# rsync -rtv ./hessian titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/hessian/


# rsync -rtv "titanic:/home/tao/vestrade/workspace/SystML/SystGradDescent/OUTPUT/VAR-HIGGSTES/NeuralNetClassifier/NeuralNetClassifier-L4x500-Adam-0.001-\(0.9-0.999\)-5000-10000/threshold.csv" OUTPUT/VAR-HIGGSTES/NeuralNetClassifier/NeuralNetClassifier-L4x500-Adam-0.001-\(0.9-0.999\)-5000-10000/threshold.csv
