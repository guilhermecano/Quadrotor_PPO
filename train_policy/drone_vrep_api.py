import vrep_env
import vrep # vrep.sim_handle_parent

import gym
import math
from gym import spaces

import numpy as np
import random as rnd
import os

class DroneVrepEnv(vrep_env.VrepEnv):
    metadata = {
        'render.modes': [],
    }
    def __init__(
        self,
        server_addr='127.0.0.1',
        server_port= -15000,
        #Editar o path da cena do drone:
        scene_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../scenes/ardrone_modeled_headless.ttt'),
    ):
        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        # All Joints
        joint_names = [
             'joint1',
             'joint2',
             'joint3',
             'joint4',
        ]
        # Some shapes
        shape_names = [
            'Quadricopter',
            'Quadricopter_target'
        ]

        # Getting object handles

        # Meta
        #self.webcam = self.get_object_handle('webcam')

        # Actuators
        self.oh_joint = joint_names
        # Shapes
        self.oh_shape = list(map(self.get_object_handle, shape_names))

        # One action per joint
        num_act = len(self.oh_joint)

        # Multiple dimensions per shape
        #num_obs = ((len(self.oh_shape)*3*3)+1);
        num_obs = 18

        self.joints_max_velocity = 100.0
        act = np.array( [self.joints_max_velocity] * num_act )
        obs = np.array(          [np.inf]          * num_obs )

        self.action_space = spaces.Box( np.array([0]*num_act) ,act)
        self.observation_space = spaces.Box(-obs,obs)

        self.obj_get_string_signal_stream('signalDataAccelerometer')

        print('DroneVrepEnv: initialized')

    def _make_observation(self):

        lst_o = []
        rel_pos = []
        rel_orient = []
        rel_lin_vel = []
        rel_ang_vel = []
        # Include shapes relative position in observation
        for i_oh in self.oh_shape:
            if i_oh == self.oh_shape[0]:
                drone_pos = self.obj_get_position(i_oh,relative_to=None)
                rel_orient = self.obj_get_orientation(i_oh, relative_to=self.oh_shape[1])
                global_orient = self.obj_get_orientation(i_oh, relative_to=None)
                lin_vel , ang_vel = self.obj_get_velocity(i_oh)
            if i_oh == self.oh_shape[1]:
                target_pos = self.obj_get_position(i_oh,relative_to=None)
                goal_lin_vel , goal_ang_vel = self.obj_get_velocity(i_oh)

        #Scaling:
        sc_p = 0.5
        sc_lv = 0.5
        sc_av = 0.15

        #relative measurements
        rel_pos = [sc_p*(drone_pos[0] - target_pos[0]),sc_p*(drone_pos[1] - target_pos[1]), sc_p*(drone_pos[2] - target_pos[2])]
        rel_ang_vel = ang_vel

        # Rotation matrix calculation (drone -> goal)
        r11 = math.cos(rel_orient[2])*math.cos(rel_orient[1])
        r12 = math.cos(rel_orient[2])*math.sin(rel_orient[1])*math.sin(rel_orient[0]) - math.sin(rel_orient[2])*math.cos(rel_orient[0])
        r13 = math.cos(rel_orient[2])*math.sin(rel_orient[1])*math.cos(rel_orient[0]) + math.sin(rel_orient[2])*math.sin(rel_orient[0])
        r21 = math.sin(rel_orient[2])*math.cos(rel_orient[1])
        r22 = math.sin(rel_orient[2])*math.sin(rel_orient[1])*math.sin(rel_orient[0])+ math.cos(rel_orient[2])*math.cos(rel_orient[0])
        r23 = math.sin(rel_orient[2])*math.sin(rel_orient[1])*math.cos(rel_orient[0]) - math.cos(rel_orient[2])*math.sin(rel_orient[0])
        r31 = -math.sin(rel_orient[1])
        r32 = math.cos(rel_orient[1])*math.sin(rel_orient[0])
        r33 = math.cos(rel_orient[1])*math.cos(rel_orient[0])

        #Relative linear velocity (robot's frame) calculation
        # Rotation matrix calculation (drone -> world)
        g11 = math.cos(global_orient[2])*math.cos(global_orient[1])
        g12 = math.cos(global_orient[2])*math.sin(global_orient[1])*math.sin(global_orient[0]) - math.sin(global_orient[2])*math.cos(global_orient[0])
        g13 = math.cos(global_orient[2])*math.sin(global_orient[1])*math.cos(global_orient[0]) + math.sin(global_orient[2])*math.sin(global_orient[0])
        g21 = math.sin(global_orient[2])*math.cos(global_orient[1])
        g22 = math.sin(global_orient[2])*math.sin(global_orient[1])*math.sin(global_orient[0])+ math.cos(global_orient[2])*math.cos(global_orient[0])
        g23 = math.sin(global_orient[2])*math.sin(global_orient[1])*math.cos(global_orient[0]) - math.cos(global_orient[2])*math.sin(global_orient[0])
        g31 = -math.sin(global_orient[1])
        g32 = math.cos(global_orient[1])*math.sin(global_orient[0])
        g33 = math.cos(global_orient[1])*math.cos(global_orient[0])

        R = np.array([[g11,g12,g13],[g21,g22,g23],[g31,g32,g33]])
        Rinv = np.transpose(R)

        #linear vel to observation
        rel_lin_vel = np.matmul(Rinv,np.asarray(lin_vel)).tolist()

        #Accelerometer
        accel = self.obj_get_string_signal_buff('signalDataAccelerometer')
        ac_out= [i * sc_acc for i in accel]

        if not ac_out:
            ac_out = [0.0, 0.0, -0.981]
        #Add  into observation list
        lst_o += rel_pos
        lst_o += [g11,g12,g13,g21,g22,g23,g31,g32,g33]
        lst_o += rel_ang_vel
        lst_o += rel_lin_vel

        self.observation = np.array(lst_o).astype('float32')


    def _make_action(self, a):
        for i_oh, i_a in zip(self.oh_joint, a):
            #self.obj_set_velocity(i_oh, i_a)
            self.obj_set_float_signal(i_oh, i_a)

    #get additional information
    def _get_reward_data(self):
        drone_pos = self.obj_get_position(self.oh_shape[0], relative_to=self.oh_shape[1])
        drone_orient = self.obj_get_orientation(self.oh_shape[0], relative_to=self.oh_shape[1])
        return drone_pos, drone_orient

    def _step(self, action):
        if isinstance(action,dict):
            ac = np.zeros(4)
            #print(action)
            #print(action['action1'])
            for i in range(4):
                ac[i] = action['action{}'.format(i)]
            action = ac

        action = np.clip(action, 0 , self.joints_max_velocity)

        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Actuate
        self._make_action(action)
        # Step
        self.step_simulation()
        # Observe
        self._make_observation()

        # Reward
        rel_pos, _ = self._get_reward_data()

        raio = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
        #norm_ang = math.sqrt(x_roll**2 + y_pitch**2 + z_yaw**2)
        norm_a_vel = math.sqrt(self.observation[12]**2 + self.observation[13]**2 + self.observation[14]**2)
        norm_vel = math.sqrt(self.observation[15]**2 + self.observation[16]**2 + self.observation[17]**2)

        r_alive = 4.0

        #Reward function
        reward = r_alive - 1.25*raio

        # Early stop
        #done = False
        stand_threshold = 3.2
        done = (raio > stand_threshold)

        #return self.observation, done, reward
        return self.observation, reward, done, {}

    def _reset(self, random = 0):
        if self.sim_running:
            self.stop_simulation()
        self.start_simulation()
        #self.step_simulation()

        #Uniform distribution reset
        if random == 1:
            #position
            pos = np.zeros(3)
            raio_init = 0.5
            for i in range(3):
                if i <=1:
                    z = rnd.uniform(-raio_init,raio_init)
                    pos[i] = z
                else:
                    z = rnd.uniform(1.7 - raio_init,1.7 + raio_init)
                    pos[i] = z
            self.obj_set_object_position(self.oh_shape[0],pos.tolist())

            #angular pos
            ang_pos = np.zeros(3)
            ang_max = 1.57 #90 graus
            for i in range(3):
                z = rnd.uniform(-ang_max,ang_max)
                pos[i] = z
            self.obj_set_object_orientation(self.oh_shape[0], ang_pos.tolist())

        #Gaussian reset
        if random == 2:
            #position
            pos = np.zeros(3)
            for i in range(3):
                z = rnd.gauss(0,0.3)
                if i <=1:
                    pos[i] = z
                else:
                    pos[i] = z + 1.7
            self.obj_set_object_position(self.oh_shape[0],pos.tolist())

            #angular pos
            ang_pos = np.zeros(3)
            for i in range(3):
                z = rnd.gauss(0,0.6)
                ang_pos[i] = z
            self.obj_set_object_orientation(self.oh_shape[0], ang_pos.tolist())

        self._make_observation()

        return self.observation

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        return []

def main(args):
    env = DroneVrepEnv()
    for i_episode in range(30):
        observation = env._reset()
        total_reward = 0
        for t in range(16):
            # (do not use this on a real robot)
            action = env.action_space.sample()
            print(action)
            observation, reward, done, _ = env._step(action)
            total_reward += reward
            if done:
                break
        print("Episode finished after {} timesteps".format(t+1))
        print("Total reward: {}".format(total_reward))
    env.close()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
