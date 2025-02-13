import os


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
