import os
import sys
import argparse
import datetime

import pandas as pd

from subprocess import call


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
#SBATCH --exclude=baltic-1,titanic-1,republic-1,republic-2,republic-3,titanic-4,republic-5,republic-6

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


def parse_args():
    parser = argparse.ArgumentParser(description="Job launcher")

    parser.add_argument('benchmark', help='benchmark main',)
    parser.add_argument('--logdir', help='path to the log directory',
                        default='logs')
    parser.add_argument('--xp-name', help='name of the experiment',
                        default='debug')
    parser.add_argument('--docker-image', help='the docker image',
                        default='estradevictorantoine/systml:1.3', type=str)

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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)


if __name__ == '__main__':
    main()
