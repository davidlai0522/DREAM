import yaml

class ConfigLoader:
    """
    A class to load configuration from a YAML file.
    """

    def __init__(self, config_file: str):
        """
        Initialize the ConfigLoader with a YAML file.

        Args:
            config_file: The path to the YAML file.
        """
        self.config_file = config_file
        self.config = None

    def load(self):
        """
        Load the configuration from the YAML file.
        """
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        """
        Get a value from the configuration.

        Args:
            key: The key to retrieve.
            default: The default value to return if the key is not found.

        Returns:
            The value associated with the key, or the default value if not found.
        """
        if self.config is None:
            self.load()
        return self.config.get(key, default)

    def get_pddl_config(self):
        """
        Get the PDDL configuration from the YAML file.

        Returns:
            The PDDL configuration.
        """
        return self.get("pddl", {})

    def get_reinforcement_learning_config(self):
        """
        Get the RL configuration from the YAML file.

        Returns:
            The RL configuration.
        """
        return self.get("reinforcement_learning", {})
    
    def get_deterministic_config(self):
        """
        Get the deterministic configuration from the YAML file.

        Returns:
            The deterministic configuration.
        """
        return self.get("deterministic", {})
