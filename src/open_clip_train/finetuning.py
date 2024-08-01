import yaml

from .main import main
from .utils import hyperparameters_path, hyperparameters_eval_path


def convert_yaml_to_argv(yaml_dict):
    """
    Convert a dictionary to a list of arguments to be passed to the main function.
    This prevents the need to modify the main function to accept a dictionary as input.
    """
    argv = []
    for key, value in yaml_dict.items():
        argv.append(f'--{key}')
        argv.append(f'{value}')
    return argv


def finetuning():
    # Load the configuration file
    eval = hyperparameters_eval_path
    train = hyperparameters_path

    with open(train) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    args = convert_yaml_to_argv(config)
    main(args)

