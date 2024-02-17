from torch.utils.data import Dataset
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper

_size_to_version = {
    "mini": "v1.0-mini",
    "full": "v1.0-trainval"
}

_size_to_split = {
    "mini": "mini_train",
    "full": "trainval"
}


class NuScenesDataset(Dataset):
    def __init__(self, dataroot, size="mini", seconds_in_future=3):
        self.nusc = NuScenes(version=_size_to_version[size], dataroot=dataroot)
        self.instances = get_prediction_challenge_split(_size_to_split[size], dataroot=dataroot)
        self.dataroot = dataroot
        self.helper = PredictHelper(self.nusc)
        self.seconds_in_future = seconds_in_future

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # instance_token  is the token of the agent we want to track
        # sample_token is the token of the sample (video scenario) we want to track the agent in
        instance_token, sample_token = self.instances[idx].split("_")
        sample = self.helper.get_sample_annotation(instance_token, sample_token)
        kwargs = {
            "instance_token": instance_token,
            "sample_token": sample_token,
            "seconds": self.seconds_in_future,
        }

        agent_future_xy_local = self.helper.get_future_for_agent(
            **kwargs,
            in_agent_frame=True,
        )

        agent_future_xy_global = self.helper.get_future_for_agent(
            **kwargs,
            in_agent_frame=False,
        )

        agent_past_xy_local = self.helper.get_past_for_agent(
            **kwargs,
            in_agent_frame=True,
        )

        agent_past_xy_global = self.helper.get_past_for_agent(
            **kwargs,
            in_agent_frame=False,
        )

        global_sample = self.helper.get_annotations_for_sample(sample_token)

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
                "agent_xy_local": agent_past_xy_local,
                "agent_xy_global": agent_past_xy_global,
            },
        }
