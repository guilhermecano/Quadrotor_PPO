from tensorforce.agents import Agent
import json
from drone_vrep_api import DroneVrepEnv
from env_drone_tf import EnvArDrone
from tensorforce.execution import Runner
import logging
import numpy as np
from hyperdash import Experiment
import tensorflow as tf
import os

def main():

    env = EnvArDrone()
    restore_model_path= "./models"

    #Optional GPU usage configuration
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

    #Network configuration
    network_spec=[
            dict(type='dense', size=64, activation='tanh'),
            dict(type='dense', size=64, activation='tanh')]
    
    agent_file = "../configs/ppo.json"

    #Agent configuration file
    with open(agent_file, 'r') as fp:
        agent_config = json.load(fp=fp)
    
    #agent_config['execution']['session_config'] = tf.ConfigProto(gpu_options=gpu_options)

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=env.states,
            actions=env.actions,
            network=network_spec,
        )
    )

    if os.path.exists(restore_model_path):
        print("Restoring saved model....")
        agent.restore_model(directory=restore_model_path)

    print("Running model trained on {agent} for Environment '{env}'".format(agent=agent, env=env))    
    
    #Running policy
    state = env.reset()
    try:
        while(True):
            actions = agent.act(state, deterministic=True)
            state, _, _ = env.execute(actions)
    except KeyboardInterrupt:
        print("Run finished")

if __name__ == '__main__':
    main()
