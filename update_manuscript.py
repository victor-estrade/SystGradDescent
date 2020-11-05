# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import shutil
import stat


OUT_DIR = "/home/estrade/Bureau/PhD/SystML/SystGradDescent/OUTPUT"
THESIS_DIR = "/home/estrade/Bureau/PhD/Manuscrit"

def cp(src, dst):
    """Copy file if dst is less recent than src."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_stat = os.stat(src)
    try:
        dst_stat = os.stat(dst)
    except FileNotFoundError:
        dst_stat = (0,)*10
    src_modif_time = src_stat[stat.ST_MTIME]
    dst_modif_time = dst_stat[stat.ST_MTIME]
    if src_modif_time > dst_modif_time:
        shutil.copyfile(src, dst)
        print("  ++", dst[len(THESIS_DIR):])
    else:
        print("  --", dst[len(THESIS_DIR):])




def src(arg, *args):
    return os.path.join(arg, *args)

dst = src  # Alias to ease reading

# Models
DA = "DataAugmentation"
GB = "GradientBoostingModel"
INF = "Inferno"
NN = "NeuralNetClassifier"
PIVOT = "PivotClassifier"
REG = "Regressor"
TP = "TangentPropClassifier"

# Benchmarks
GG = "GG"
GGPrior = "GG-prior"
GGPriorPlus = "GG-prior-plus"
GGCalib = "GG-calib"
GGMarginal = "GG-marginal"
GGCheat = "GG-cheat"

S3D2 = "S3D2"
S3D2Prior = "S3D2-prior"
S3D2Calib = "S3D2-calib"
S3D2Marginal = "S3D2-marginal"
S3D2Cheat = "S3D2-cheat"

# Others
COMPARE = "COMPARE"
PROFUSION = "PROFUSION"
BEST_MSE = "BEST_MSE"
BEST_MEDIAN = "BEST_MEDIAN"



# Chapter
Chap1 = os.path.join("Chapter1", "Figs", "Raster")
Chap2 = os.path.join("Chapter2", "Figs", "Raster")
Chap3 = os.path.join("Chapter3", "Figs", "Raster")
Chap4 = os.path.join("Chapter4", "Figs", "Raster")
Chap5 = os.path.join("Chapter5", "Figs", "Raster")
Chap6 = os.path.join("Chapter6", "Figs", "Raster")

App1 = os.path.join("Appendix1", "Figs", "Raster")
App2 = os.path.join("Appendix2", "Figs", "Raster")
App3 = os.path.join("Appendix3", "Figs", "Raster")
App4 = os.path.join("Appendix4", "Figs", "Raster")





def main():

    # =================================================================
    # CHAPTER 4
    # =================================================================
    print('CHAPTER 4')
    cp(src(OUT_DIR, GG, "explore", "x_distrib.png")
            , dst(THESIS_DIR, Chap4, "minitoy", "gg_distrib.png") )
    cp(src(OUT_DIR, S3D2, "explore", "pairgrid.png")
            , dst(THESIS_DIR, Chap4, "s3d2", "pairgrid.png") )

    # =================================================================
    # CHAPTER 5
    # =================================================================
    print('CHAPTER 5')
    # fig:gg_baseline_nominal_n_samples_mse
    print('fig:gg_baseline_nominal_n_samples_mse')
    cp(src(OUT_DIR, COMPARE, GGPrior, GB, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, GB,  "profusion_nominal_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, NN, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, NN,  "profusion_nominal_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, DA, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, DA,  "profusion_nominal_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, TP, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, TP,  "profusion_nominal_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, INF, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, INF,  "profusion_nominal_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, PIVOT, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, PIVOT,  "profusion_nominal_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, REG, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, REG,  "profusion_nominal_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGMarginal, REG, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGMarginal, REG,  "profusion_nominal_n_samples_mse.png") )

    cp(src(OUT_DIR, COMPARE, GGPriorPlus, REG, PROFUSION,  "profusion_nominal_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPriorPlus, REG,  "profusion_nominal_n_samples_mse.png") )


    # fig:compare_gg_best_mse_n_samples
    print('fig:compare_gg_best_mse_n_samples')
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=50-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=50-boxplot_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=100-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=100-boxplot_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=500-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=500-boxplot_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-boxplot_mse.png") )

    # fig:gg_baseline_n_samples_mse
    print("fig:gg_baseline_n_samples_mse")
    cp(src(OUT_DIR, COMPARE, GGPrior, GB, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, GB,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, NN, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, NN,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, DA, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, DA,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, TP, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, TP,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, INF, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, INF,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, PIVOT, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, PIVOT,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, REG, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, REG,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGMarginal, REG, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGMarginal, REG,  "profusion_n_samples_mse.png") )

    cp(src(OUT_DIR, COMPARE, GGPriorPlus, REG, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPriorPlus, REG,  "profusion_n_samples_mse.png") )


    # fig:gg_baseline_compare_calib_estimator
    print('fig:gg_baseline_compare_calib_estimator')
    cp(src(OUT_DIR, COMPARE, GGPrior, GB, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, GB,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, GB, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, GB,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, NN, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, NN,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, NN, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, NN,  "profusion_true_mu_target_mean.png") )

    # fig:gg_syst_aware_compare_calib_estimator
    print('fig:gg_syst_aware_compare_calib_estimator')
    cp(src(OUT_DIR, COMPARE, GGPrior, DA, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, DA,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, DA, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, DA,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, TP, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, TP,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, TP, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, TP,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, PIVOT, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, PIVOT,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, PIVOT, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, PIVOT,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, INF, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, INF,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, INF, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, INF,  "profusion_true_mu_target_mean.png") )

    # fig:gg_regressor_compare_calib_estimator
    print('fig:gg_regressor_compare_calib_estimator')
    cp(src(OUT_DIR, COMPARE, GGPrior, REG, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, REG,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, REG, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, REG,  "profusion_true_mu_target_mean.png") )
    cp(src(OUT_DIR, COMPARE, GGMarginal, REG, PROFUSION,  "profusion_true_mu_target_mean.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGMarginal, REG,  "profusion_true_mu_target_mean.png") )

    # fig:gg_baseline_compare_calib_n_samples_mse
    print('fig:gg_baseline_compare_calib_n_samples_mse')
    cp(src(OUT_DIR, COMPARE, GGPrior, NN, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, NN,  "profusion_n_samples_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, NN, PROFUSION,  "profusion_n_samples_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, NN,  "profusion_n_samples_mse.png") )

    # fig:compare_gg_best_mse
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-errplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-errplot_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-boxplot_mse.png") )

    cp(src(OUT_DIR, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=2000-errplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=2000-errplot_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=2000-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=2000-boxplot_mse.png") )


    # fig:gg_mean_link
    print('fig:gg_mean_link')
    cp(src(OUT_DIR, GG, "explore", "mean_link.png")
        , dst(THESIS_DIR, Chap5, GG,  "mean_link.png") )

    # fig:gg_prior_distrib_summaries
    print('fig:gg_prior_distrib_summaries')
    cp(src(OUT_DIR, GGPrior, DA, "DataAugmentation-L4x200-Adam-0.001-(0.9-0.999)-2000-20", "cv_0", "valid_summaries.png")
            , dst(THESIS_DIR, Chap5, GGPrior, DA, "valid_summaries.png") )
    cp(src(OUT_DIR, GGPrior, DA, "DataAugmentation-L4x200-Adam-0.001-(0.9-0.999)-2000-20", "cv_0", "valid_distrib.png")
            , dst(THESIS_DIR, Chap5, GGPrior, DA, "valid_distrib.png") )
    cp(src(OUT_DIR, GGPrior, NN, "NeuralNetClassifier-L4x200-Adam-0.001-(0.9-0.999)-2000-20", "cv_0", "valid_summaries.png")
            , dst(THESIS_DIR, Chap5, GGPrior, NN, "valid_summaries.png") )
    cp(src(OUT_DIR, GGPrior, NN, "NeuralNetClassifier-L4x200-Adam-0.001-(0.9-0.999)-2000-20", "cv_0", "valid_distrib.png")
            , dst(THESIS_DIR, Chap5, GGPrior, NN, "valid_distrib.png") )


    # fig:compare_gg_best_mse50_samples
    print('fig:compare_gg_best_mse50_samples')
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=50-errplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=50-errplot_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=50-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=50-boxplot_mse.png") )

    cp(src(OUT_DIR, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=50-errplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=50-errplot_mse.png") )
    cp(src(OUT_DIR, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=50-boxplot_mse.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGCalib, BEST_MSE, "GG-calib_best_average_N=50-boxplot_mse.png") )


    # fig:compare_gg_prior_best_mse_v_stat_syst
    print('fig:compare_gg_prior_best_mse_v_stat_syst')
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-errplot_v_stat.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-errplot_v_stat.png") )
    cp(src(OUT_DIR, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-errplot_v_syst.png")
            , dst(THESIS_DIR, Chap5, COMPARE, GGPrior, BEST_MSE, "GG-prior_best_average_N=2000-errplot_v_syst.png") )



if __name__ == '__main__':
    main()
