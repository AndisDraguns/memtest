from gym.envs.registration import register
 
register(id='MemTest-v0', 
    entry_point='gym_memtest.envs:MemTestEnv', 
)