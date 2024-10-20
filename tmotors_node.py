from collections import namedtuple

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState


TorqueCmd = namedtuple("TorqueCmd", "tau")
PositionCmd = namedtuple("PositionCmd", "q")
EnableCmd = namedtuple("EnableCmd", "")
Cmd = TorqueCmd | PositionCmd | EnableCmd


class TMotorsNode(Node):

    def __init__(self):
        super().__init__("tmotors_node")

        self.joint_sensor = self.create_subscription(JointState, "/joints_sensor", self._read_state, 1)
        self.joint_cmd = self.create_publisher(JointState, "/joints_cmd", 1)

        self.joint_state = np.zeros(3)


    def _read_state(self, msg):
        self.joint_state[0] = msg.position[1]
        self.joint_state[1] = msg.velocity[1]
        self.joint_state[2] = msg.effort[1]

    def send_cmd(self, cmd):
        msg = JointState()

        match cmd:
            case TorqueCmd():
                msg.name = "torque", "torque"
                msg.position = [0., 0.]
                msg.velocity = [0., 0.]
                msg.effort = [0., cmd.tau]
            case PositionCmd():
                msg.name = "position", "position"
                msg.position = [0., cmd.q]
                msg.velocity = [0., 0.]
                msg.effort = [0., 0.]
            case EnableCmd():
                msg.name = "enable", "enable"
                msg.position = [0., 0.]
                msg.velocity = [0., 0.]
                msg.effort = [0., 0.]
            case _:
                raise ValueError("cmd must be of type Cmd")

        self.joint_cmd.publish(msg)
