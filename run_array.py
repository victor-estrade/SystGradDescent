import os
import sys
import argparse
import datetime
import itertools
from uuid import uuid4

import pandas as pd

from subprocess import call

def parse_args():
    parser = argparse.ArgumentParser(description="Job launcher")

    parser.add_argument('benchmark', help='benchmark main',)
    parser.add_argument('--logdir', help='path to the log directory',
                        default='logs')
    parser.add_argument('--xp-name', help='name of the experiment',
                        default='debug')
    parser.add_argument('--docker-image', help='the docker image',
                        default='estradevictorantoine/systml:1.4', type=str)

    # max runtime days-hours:min:sec
    ressources = parser.add_argument_group('ressources', 'ressources taken by every job')
    ressources.add_argument('--max-time', help='maximum job time',
                        default='12-00:00:00')
    ressources.add_argument('--cpu', help='number of allocated CPUs',
                        default=6, type=int)
    ressources.add_argument('--mem', help='allocated RAM',
                        default='64g')
    ressources.add_argument('--partition', help='chosen partition on the cluster',
                        default='titanic')
    ressources.add_argument('--gpu', help='number of allocated GPUs',
                        default=0, type=int)

    # main arguments
    main_args = parser.add_argument_group('main_args', 'arguments passed to the all subjobs')
    main_args.add_argument('--no-cuda', '--no-gpu', help='flag to use or not the gpu',
                        action='store_false', dest='cuda')
    main_args.add_argument('--retrain', help='flag to force retraining',
                        action='store_true')
    main_args.add_argument('--skip-minuit', help='flag to skip minuit NLL minization',
                        action='store_true')

    main_args.add_argument("--start-cv", type=int,
                        default=0, help="start of i_cv for range(start, end)")
    main_args.add_argument("--end-cv", type=int,
                        default=30, help="end of i_cv for range(start, end)")
    main_args.add_argument('--load-run', help='load saved runs. Do not run the models',
                        action='store_true')
    main_args.add_argument('--estimate-only', help='Turns off conditional estimation for V_stat and V_syst',
                        action='store_true')
    main_args.add_argument('--conditional-only', help='Turns off common estimation',
                        action='store_true')

    grid_args = parser.add_argument_group('grid_args', 'arguments passed to the subjobs for grid search')
    grid_args.add_argument('--feature-id',
                        nargs='+',
                        help='feature index for Feature filter model',
                        type=int)

    grid_args.add_argument('--n-estimators',
                        nargs='+',
                        help='number of estimators',
                        type=int)

    grid_args.add_argument('--max-depth',
                        nargs='+',
                        help='maximum depth of trees',
                        type=int)

    grid_args.add_argument('--learning-rate', '--lr',
                        nargs='+',
                        help='learning rate',
                        type=float)

    grid_args.add_argument('--beta1',
                        nargs='+',
                        help='beta 1 for Adam',
                        type=float)

    grid_args.add_argument('--beta2',
                        nargs='+',
                        help='beta 2 for Adam',
                        type=float)

    grid_args.add_argument('--weight-decay',
                        nargs='+',
                        help='weight decay for SGD',
                        type=float)

    grid_args.add_argument('--trade-off',
                        nargs='+',
                        help='trade-off for multi-objective models',
                        type=float)

    grid_args.add_argument('-w', '--width',
                        nargs='+',
                        help='width for the data augmentation sampling',
                        type=float)

    grid_args.add_argument('--n-unit',
                        nargs='+',
                        help='Number of units in layers. Controls NN width.',
                        type=int)

    grid_args.add_argument('--sample-size',
                        nargs='+',
                        help='mini-batch size',
                        type=int)

    grid_args.add_argument('--batch-size',
                        nargs='+',
                        help='mini-batch size',
                        type=int)

    grid_args.add_argument('--n-steps',
                        nargs='+',
                        help='number of update steps',
                        type=int)

    grid_args.add_argument('--n-augment',
                        nargs='+',
                        help='number of times the dataset is augmented',
                        type=int)

    grid_args.add_argument('--n-adv-pre-training-steps',
                        nargs='+',
                        help='number of update steps for the pre-training',
                        type=int)

    grid_args.add_argument('--n-clf-pre-training-steps',
                        nargs='+',
                        help='number of update steps for the pre-training',
                        type=int)

    grid_args.add_argument('--n-recovery-steps',
                        nargs='+',
                        help='number of update steps for the catch training of auxiliary models',
                        type=int)

    args = parser.parse_args()
    grid_args_dict = extract_group_args(parser, args, 'grid_args')
    main_args_dict = extract_group_args(parser, args, 'main_args')

    return args, grid_args_dict, main_args_dict


def extract_group_args(parser, args, group_title):
    is_group_title = lambda group : group.title == group_title
    group = list(filter(is_group_title, parser._action_groups))[0]
    group_args = {action.option_strings[0]: getattr(args, action.dest, None)
                                    for action in group._group_actions}
    return group_args


SBATCH_TEMPLATE = \
"""#!/bin/bash
#SBATCH --account=tau
#SBATCH --array={array}%10
#SBATCH --job-name={xp_name}
#SBATCH --output={log_stdout}
#SBATCH --error={log_stderr}
#SBATCH -t {max_time}             # max runtime days-hours:min:sec
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={memory}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu}
#SBATCH --exclude=baltic-1,republic-3,republic-1,titanic-4

hostname

function dockerkill
{{
    echo "Killing docker {container_name}_${{SLURM_ARRAY_TASK_ID}}"
    docker kill {container_name}_${{SLURM_ARRAY_TASK_ID}}
    echo "Cancelling job ${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
    scancel "${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
}}

trap dockerkill TERM
trap dockerkill INT
trap dockerkill CONT

GRID_PARAMS=$(cat {parameters_file} | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)
WORKDIR="/home/tao/vestrade/workspace/SystML/SystGradDescent"

echo "SLURM_ARRAY_TASK_ID"
echo $SLURM_ARRAY_TASK_ID

echo "GRID_PARAMS"
echo "${{GRID_PARAMS}}"

sdocker -i  -v /home/tao/vestrade/datawarehouse:/datawarehouse \
            -v $WORKDIR:$WORKDIR --name "{container_name}_${{SLURM_ARRAY_TASK_ID}}" \
            {docker_image} \
            bash -c "cd ${{WORKDIR}}; python -m {benchmark} {main_args} ${{GRID_PARAMS}}"
"""

#
# python -m {benchmark} {main_args} ${{GRID_PARAMS}}
#


def param_to_grid(parameter_dict):
    params = itertools.product(*parameter_dict.values())
    param_names = parameter_dict.keys()
    grid = [{k:v for k, v in zip(param_names, v_list)} for v_list in params]
    return grid


def grid_to_str(grid):
    return "\n".join([' '.join(['{} {}'.format(k, v) for k, v in d.items()]) for d in grid])


def to_list(l):
    try:
        iter(l)
        return l
    except TypeError:
        return [l]


def register(logdir, now, benchmark, xp_logdir, main_args, parameter_dict):
    info = format_register(now, benchmark, xp_logdir, main_args, parameter_dict)
    print(info)
    logfile = os.path.join(logdir, "run_log.txt")
    with open(logfile, "a") as file:
        print(info, file=file)


def format_register(now, benchmark, xp_logdir, main_args, parameter_dict):
    tabulation = " "*4
    head = "="*len(now)
    param_grid = f"\n{tabulation}".join([f"{k} : {v}" for k, v in parameter_dict.items()])
    info = \
    f"""
{now}
{tabulation} {benchmark:25s} {xp_logdir}
{tabulation}{main_args}
{tabulation}{param_grid}
"""
    return info

def main():
    # Extract arguments :
    args, grid_args, main_args = parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    xp_name = args.xp_name
    logdir = args.logdir
    xp_logdir = os.path.join(args.logdir, xp_name, now)
    max_time = args.max_time
    cpu = args.cpu
    memory = args.mem
    partition = args.partition
    gpu = args.gpu
    docker_image = args.docker_image
    benchmark = args.benchmark

    container_name = str(uuid4())[:8]

    # Main parameters for grid search
    parameter_dict = {k: to_list(v) for k, v in grid_args.items() if v is not None}
    grid = param_to_grid(parameter_dict)
    array = "1-{}".format(len(grid))

    # Handle flag in main_args
    if main_args['--retrain'] :
        main_args['--retrain'] = ' '
    else:
        main_args.pop('--retrain')

    if main_args['--load-run'] :
        main_args['--load-run'] = ' '
    else:
        main_args.pop('--load-run')

    if main_args['--estimate-only'] :
        main_args['--estimate-only'] = ' '
    else:
        main_args.pop('--estimate-only')

    if main_args['--conditional-only'] :
        main_args['--conditional-only'] = ' '
    else:
        main_args.pop('--conditional-only')

    if not main_args['--no-cuda'] :
        main_args['--no-cuda'] = ' '
    else:
        main_args.pop('--no-cuda')

    if main_args['--skip-minuit'] :
        main_args['--skip-minuit'] = ' '
    else:
        main_args.pop('--skip-minuit')

    main_args = " ".join(["{} {}".format(k, v) for k, v in main_args.items()])

    # Extra arguments
    log_stdout = os.path.join(xp_logdir, '%A_%a.stdout')
    log_stderr = os.path.join(xp_logdir, '%A_%a.stderr')
    script_slurm = os.path.join(xp_logdir, 'script.slurm')
    parameters_file = os.path.join(xp_logdir, 'parameters.txt')
    parameters_file_csv = os.path.join(xp_logdir, 'parameters.csv')

    # Final formating
    script = SBATCH_TEMPLATE.format(**locals())

    # Prepare logs files, slurm script, parameters, etc
    os.makedirs(xp_logdir, exist_ok=True)
    with open(script_slurm, "w") as file:
        print(script, file=file)
    # Prepare parameter grid file
    with open(parameters_file, "w") as file:
        print(grid_to_str(grid), file=file)
    pd.DataFrame(grid).to_csv(parameters_file_csv, index=False)

    # Start job
    cmd = ['sbatch', script_slurm]
    print(" ".join(cmd))
    call(cmd)
    register(logdir, now, benchmark, xp_logdir, main_args, parameter_dict)

if __name__ == '__main__':
    main()
