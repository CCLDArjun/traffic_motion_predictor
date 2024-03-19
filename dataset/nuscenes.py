import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

_size_to_version = {
    "mini": "v1.0-mini",
    "full": "v1.0-trainval"
}

_size_to_split = {
    "mini": "mini_train",
    "full": "train",
}

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, size="mini", seconds_in_future=6, seconds_in_past=2):
        self.nusc = NuScenes(version=_size_to_version[size], dataroot=dataroot)
        self.instances = get_prediction_challenge_split(_size_to_split[size], dataroot=dataroot)
        self.dataroot = dataroot
        self.helper = PredictHelper(self.nusc)
        self.seconds_in_future = seconds_in_future
        self.static_later_rast = StaticLayerRasterizer(self.helper)
        self.agent_rast = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=seconds_in_past)
        self.input_creator = InputRepresentation(self.static_later_rast, self.agent_rast, Rasterizer())
        self.seconds_in_past = seconds_in_past
        self.hertz = 2

    def __len__(self):
        return len(self.instances)

    def _pad(self, array, length):
        if len(array) == length:
            return array
        first = array[0]
        padding = np.array([first] * (length - len(array)))
        return np.concatenate([array, padding], axis=0)

    def __getitem__(self, idx):
        # instance_token  is the token of the agent we want to track
        # sample_token is the token of the sample (video scenario) we want to track the agent in
        instance_token, sample_token = self.instances[idx].split("_")
        sample = self.helper.get_sample_annotation(instance_token, sample_token)
        kwargs = {
            "instance_token": instance_token,
            "sample_token": sample_token,
        }

        agent_future_xy_local = self.helper.get_future_for_agent(
            **kwargs,
            seconds=self.seconds_in_future,
            in_agent_frame=True,
        )

        agent_future_xy_global = self.helper.get_future_for_agent(
            **kwargs,
            seconds=self.seconds_in_future,
            in_agent_frame=False,
        )

        agent_past_xy_local = self.helper.get_past_for_agent(
            **kwargs,
            seconds=self.seconds_in_past,
            in_agent_frame=True,
        )

        agent_past_xy_global = self.helper.get_past_for_agent(
            **kwargs,
            seconds=self.seconds_in_past,
            in_agent_frame=False,
        )

        global_sample = self.helper.get_annotations_for_sample(sample_token)
        sample_thing = self.nusc.get("sample", sample_token)
        tuple_containing_img = self.nusc.get_sample_data(sample_thing["data"]["CAM_FRONT"])
        top_down_repr = self.static_later_rast.make_representation(instance_token, sample_token)
        agent_rast = self.agent_rast.make_representation(instance_token, sample_token)

        input_image = self.input_creator.make_input_representation(instance_token, sample_token) # this is combined agent and static raster

        velocity = self.helper.get_velocity_for_agent(instance_token, sample_token)
        acceleration = self.helper.get_acceleration_for_agent(instance_token, sample_token)
        heading_change_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)

        return {
            "global_sample": global_sample,
            "sample": sample,
            "instance_token": instance_token,
            "sample_token": sample_token,
            "future": {
                "agent_xy_local": agent_future_xy_local,
                "agent_xy_global": agent_future_xy_global,
            },
            "past": {
                "agent_xy_local": self._pad(agent_past_xy_local, self.seconds_in_past * self.hertz),
                "agent_xy_global": self._pad(agent_past_xy_global, self.seconds_in_past * self.hertz),
            },
            "top_down_repr": top_down_repr,
            "input_image": input_image,
            "agent_rast": agent_rast,
            "velocity": velocity,
            "acceleration": acceleration,
            "heading_change_rate": heading_change_rate,
        }

