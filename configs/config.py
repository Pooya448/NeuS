import yaml


def load_configuration(config_path):
    """
    Loads a configuration file.

    Parameters:
        config_path (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config
