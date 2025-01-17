import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "protos", "generated")) # Path to compiled proto files
import time_step_pb2
from google.protobuf.internal.decoder import _DecodeVarint32

import pandas as pd

class SimData:
    """Wrapper class for parsing protobuf experiment data."""

    def __init__(self, log_data_path, commands_path=''):
        """Constructor"""

        # Data dict
        self.data = {}

        self.read_log_data(log_data_path)

        # Add user comamnds if csv file exists
        if commands_path:
            self.read_commands(commands_path)


    def read_log_data(self, log_data_path):
        """Parses a protobuf file with simulation logs."""

        with open(log_data_path, 'rb') as f:
            buf = f.read()
            n = 0

            # Read each timestep
            while n < len(buf):
                msg_len, new_pos = _DecodeVarint32(buf, n)
                n = new_pos
                msg_buf = buf[n:n+msg_len]
                n += msg_len
                read_data = time_step_pb2.TimeStep()
                read_data.ParseFromString(msg_buf)

                # Create an entry for the timestep
                self.data[read_data.time] = {}
                self.data[read_data.time]['log'] = read_data
                self.data[read_data.time]['commands'] = [] # Create empty list of commands
                
            # Store experiment summary
            if self.data:

                # Total timesteps
                self.totalTime = len(self.data)

                # Points scored
                self.totalPoints = self.data[self.totalTime]['log'].points

                # Number of leaders and robots
                leaderNames = []
                teamNames = []
                for robot in self.data[1]['log'].robots:
                    if robot.state == time_step_pb2.Robot.LEADER:
                        leaderNames.append(robot.name)
                        teamNames.append(robot.teamID)

                self.leaders = leaderNames
                self.teams = teamNames
                self.numLeaders = len(self.leaders)
                self.numWorkers = len(self.data[1]['log'].robots) - self.numLeaders

                # Number of tasks that appeared in the experiment
                maxId = 0
                for task in self.data[self.totalTime]['log'].tasks:
                    id = int(task.name[5:])
                    if id > maxId:
                        maxId = id

                self.numTasks = maxId


    def read_commands(self, commands_path):
        """Parses a csv file with user commands."""

        df = pd.read_csv(commands_path)
        # print(df.to_string())

        self.users = set()

        # iterate row
            # append command (type, value, user, robot) to time

        for ind in df.index:
            command = {}

            time = df['TIME'][ind]
            command['time'] = time
            command['type'] = df['COMMAND'][ind]
            command['user'] = df['USER'][ind]
            command['value'] = df['VALUE'][ind]

            # Assume valid username is less than 8 characters
            if len(str(command['user'])) < 8:
                self.data[time]['commands'].append(command)
                self.users.add(int(command['user']))

            # print(self.data[time]['commands'])


    def __getitem__(self, key):
        return self.data[key]
    