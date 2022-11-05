from gymnasium.envs.registration import register

register(
     id="working_memory_env/DMTSGridEnv-4x4-v0",
     entry_point="working_memory_env.envs:DMTSGridEnv",
     kwargs={
          "grid_size": 4,
          "max_steps": 300,
     }
)