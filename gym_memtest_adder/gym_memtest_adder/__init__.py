from gym.envs.registration import register
 
register(id='MemTestAdder-v0', 
    entry_point='gym_memtest_adder.envs:MemTestAdderEnv', 
)