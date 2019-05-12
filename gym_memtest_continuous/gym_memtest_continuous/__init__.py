from gym.envs.registration import register
 
register(id='MemTestContinuous-v0', 
    entry_point='gym_memtest_continuous.envs:MemTestContinuousEnv', 
)