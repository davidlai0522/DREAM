from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO

class RLModelInference:
    def __init__(self, env: GymWrapper, model_path, algorithm="ppo"):
        self.env = env
        self.model_path = model_path
        self.algorithm = algorithm
        self.model = self._load_model()
        self._print_model_info()

    def _load_model(self):
        if self.algorithm == "ppo":
            model = PPO.load(f"{self.model_path}")
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        return model

    def _print_model_info(self):
        # print("Model Architecture:")
        # print(f"\t{self.model.policy}")
        print("\nModel Hyperparameters:")
        print(
            f"learning_rate: {self.model.learning_rate}"
        )  # Example of a hyperparameter
        print(f"n_steps: {self.model.n_steps}")  # Number of steps per update
        print(
            f"batch_size: {self.model.batch_size}"
        )  # Size of the batch for each training update
        print(f"gamma: {self.model.gamma}")  # Discount factor for rewards
        print(
            f"gae_lambda: {self.model.gae_lambda}"
        )  # GAE (Generalized Advantage Estimation) lambda

    def test(self, num_steps=1000):
        obs, _ = self.env.reset()
        for _ in range(num_steps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, _, info = self.env.step(action)
            self.env.render()
            if dones:
                print(f"Task completed! Total rewards: {rewards}")
                break
        self.env.close()
        print("Finish testing...")
