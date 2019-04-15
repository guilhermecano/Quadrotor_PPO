from tensorforce.agents import Agent
from drone_vrep_api import DroneVrepEnv
from env_drone_tf import EnvArDrone
from tensorforce.execution import Runner
import numpy as np
import tensorflow as tf
import json
import os
import csv

def main():
    env = EnvArDrone()
    max_timesteps = 250
    max_episodes = 5000000 #None for keep on training
    save_model_path = "./models/model"
    backup_path = "./stamps/model"
    restore_model_path= "./models"
    #Optional GPU usage configuration
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

    #Network configuration
    network_spec=[
            dict(type='dense', size=64, activation='tanh'),
            dict(type='dense', size=64, activation='tanh')]

    agent_file = "../configs/ppo.json"

    #Agent configuration file
    with open(agent_file, 'r') as fp:
        agent_config = json.load(fp=fp)
    agent_config['execution']['session_config'] = tf.ConfigProto(gpu_options=gpu_options)
#    agent_config['device']='/cpu:0'

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

    runner = Runner(agent, env)

    def episode_finished(r):
        if r.episode % 10 == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))

            print("Episode reward: {}".format(r.episode_rewards[-1]))

            print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
        if r.episode % 100 == 0:
            with open('reward.csv', 'a') as csvfile:
                rew_writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                rew_writer.writerow([np.mean(r.episode_rewards[-100:]),np.var(r.episode_rewards[-100:])])

        if r.timestep >= 8000000 and r.timestep <8000253:
            print("Saving TimeStamp...")
            agent.save_model(directory=backup_path, append_timestep=True)
        if r.timestep >= 10000000 and r.timestep <10000253:
            print("Saving TimeStamp...")
            agent.save_model(directory=backup_path, append_timestep=True)
        if r.timestep >= 15000000 and r.timestep <15000253:
            print("Saving TimeStamp...")
            agent.save_model(directory=backup_path, append_timestep=True)
        if r.timestep >= 20000000 and r.timestep <20000253:
            print("Saving TimeStamp...")
            agent.save_model(directory=backup_path, append_timestep=True)
        if r.episode % 100 == 0:
            print("Saving Model...")
            agent.save_model(directory=save_model_path, append_timestep=False)

        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run( num_timesteps=7000000, num_episodes=max_episodes, max_episode_timesteps=max_timesteps, episode_finished=episode_finished)
    runner.close()

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    main()
