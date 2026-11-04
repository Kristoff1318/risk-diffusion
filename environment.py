"""Native Maze2D Large v1, optionally with additional physical obstacles."""
import xml.etree.ElementTree as xml

import numpy as np
import runtime
from diffuser.datasets import load_environment

# Fixed prior obstacle locations, in the same qpos coordinates as the agent.
ADDED_OBSTACLES = ((3.0, 4.0, 0.35), (5.0, 8.0, 0.35), (7.0, 5.0, 0.35))


def make_environment(obstacles=()):
    env = load_environment('maze2d-large-v1')
    if not obstacles:
        return env
    import mujoco_py
    tree = xml.fromstring(env.model.get_xml())
    world = tree.find('worldbody')
    offset = env.model.body_pos[1, :2]
    for index, (x, y, radius) in enumerate(obstacles):
        world_position = np.array([x, y]) + offset
        xml.SubElement(world, 'geom', name=f'added_obstacle_{index}', type='cylinder',
                       pos=f'{world_position[0]} {world_position[1]} 0',
                       size=f'{radius} 0.2', contype='2', conaffinity='1', condim='1',
                       margin='0.002', rgba='0.8 0.15 0.15 1')
    env.model = mujoco_py.load_model_from_xml(xml.tostring(tree, encoding='unicode'))
    env.sim = mujoco_py.MjSim(env.model)
    env.data = env.sim.data
    env.init_qpos = env.sim.data.qpos.copy()
    env.init_qvel = env.sim.data.qvel.copy()
    return env


def active_collision(env):
    """Only active MuJoCo contacts; ground contact is normal for this agent."""
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        names = [env.model.geom_id2name(int(g)) or ''
                 for g in (contact.geom1, contact.geom2)]
        if contact.dist <= .002 and any(
                name.startswith(('wall_', 'added_obstacle_')) for name in names):
            return True
    return False
