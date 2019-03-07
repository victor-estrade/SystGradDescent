import os
import sys
import argparse
import datetime
import itertools

import pandas as pd

from subprocess import call


def parse_args():
    parser = argparse.ArgumentParser(description="Job launcher")

    parser.add_argument('--logdir', help='path to the log directory',
                        default='logs')
    parser.add_argument('--xp-name', help='name of the experiment',
                        default='debug')
    parser.add_argument('--docker-image', help='the docker image',
                        default='estradevictorantoine/systml:1.1', type=str)

    # max runtime days-hours:min:sec
    ressources = parser.add_argument_group('ressources', 'ressources taken by every job')
    ressources.add_argument('--max-time', help='maximum job time',
                        default='2-00:00:00')
    ressources.add_argument('--cpu', help='number of allocated CPUs',
                        default=6, type=int)
    ressources.add_argument('--mem', help='allocated RAM',
                        default='64g')
    ressources.add_argument('--partition', help='chosen partition on the cluster',
                        default='titanic')
    ressources.add_argument('--gpu', help='number of allocated GPUs',
                        default=0, type=int)

    # main arguments
    main_args = parser.add_argument_group('main_args', 'arguments passed to the subjobs for grid search')
    main_args.add_argument('--n-estimators',
                        nargs='+',
                        help='number of estimators',
                        default=1000, type=int)

    main_args.add_argument('--max-depth',
                        nargs='+',
                        help='maximum depth of trees',
                        default=3, type=int)

    main_args.add_argument('--learning-rate', '--lr',
                        nargs='+',
                        help='learning rate',
                        default=1e-3, type=float)

    main_args.add_argument('--trade-off',
                        nargs='+',
                        help='trade-off for multi-objective models',
                        default=1.0, type=float)

    main_args.add_argument('-w', '--width',
                        nargs='+',
                        help='width for the data augmentation sampling',
                        default=5, type=float)

    main_args.add_argument('--batch-size',
                        nargs='+',
                        help='mini-batch size',
                        default=128, type=int)

    main_args.add_argument('--n-steps',
                        nargs='+',
                        help='number of update steps',
                        default=10000, type=int)

    main_args.add_argument('--n-augment',
                        nargs='+',
                        help='number of times the dataset is augmented',
                        default=2, type=int)

    main_args.add_argument('--n-adv-pre-training-steps',
                        nargs='+',
                        help='number of update steps for the pre-training',
                        default=3000, type=int)

    main_args.add_argument('--n-clf-pre-training-steps',
                        nargs='+',
                        help='number of update steps for the pre-training',
                        default=3000, type=int)

    main_args.add_argument('--n-recovery-steps',
                        nargs='+',
                        help='number of update steps for the catch training of auxiliary models',
                        default=5, type=int)

    args = parser.parse_args()
    main_args_dict = extract_group_args(parser, args, 'main_args')

    return args, main_args_dict


def extract_group_args(parser, args, group_title):
    parser_groups = {group.title:{action.option_strings[0]:getattr(args,action.dest,None) 
                        for action in group._group_actions} 
                    for group in parser._action_groups}
    group_args = parser_groups[group_title]
    return group_args


SBATCH_TEMPLATE = \
"""#!/bin/sh
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

GRID_PARAMS=$(cat {parameters_file} | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)
WORKDIR="/home/tao/vestrade/workspace/SystML/"

sdocker -i  -v /data/titanic_3/users/vestrade/datawarehouse:/datawarehouse \
            -v /data/titanic_3/users/vestrade/savings:/data/titanic_3/users/vestrade/savings \
            -v $WORKDIR:$WORKDIR \
            {docker_image} \
            /bin/sh -c "cd ${{WORKDIR}}; python main.py ${{GRID_PARAMS}}" 
"""


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


def main():
    # Extract arguments :
    args, main_args = parse_args()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    xp_name = args.xp_name
    logdir = os.path.join(args.logdir, xp_name, now)
    max_time = args.max_time
    cpu = args.cpu
    memory = args.mem
    partition = args.partition
    gpu = args.gpu
    docker_image = args.docker_image

    # Main parameters for grid search
    parameter_dict = {k:to_list(v) for k, v in main_args.items()}
    grid = param_to_grid(parameter_dict)
    array = "1-{}".format(len(grid))

    # Extra arguments
    log_stdout = os.path.join(logdir, '%A_%a.stdout')
    log_stderr = os.path.join(logdir, '%A_%a.stderr')
    script_slurm = os.path.join(logdir, 'script.slurm')
    parameters_file = os.path.join(logdir, 'parameters.txt')
    parameters_file_csv = os.path.join(logdir, 'parameters.csv')

    # Final formating
    script = SBATCH_TEMPLATE.format(**locals())

    # Prepare logs files, slurm script, parameters, etc
    os.makedirs(logdir, exist_ok=True)
    with open(script_slurm, "w") as file:
        print(script, file=file)
    # Prepare parameter grid file
    with open(parameters_file, "w") as file:
        print(grid_to_str(grid), file=file)
    pd.DataFrame(grid).to_csv(parameters_file_csv, index=False)

    # Start job
    call(['sbatch', script_slurm])

if __name__ == '__main__':
    main()
