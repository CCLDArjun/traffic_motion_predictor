import random
import string
from functools import wraps
from datetime import datetime
import os

import torch
from torch.profiler import profile, record_function, ProfilerActivity

NSIGHT_STARTED = False

def get_time_str():
    return datetime.now().strftime("%m-%d-%y_%H-%M-%S")

def random_id(n=10):
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=n))

def profiler(F, filename): 
    def new_func(*args, **kwargs):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=False,
            profile_memory=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{filename}"),
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
        ) as prof:
            ret = F(*args, **kwargs)

        print("CPUUUU")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print("CUDAAA")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


        return ret
    return new_func

def startNsight():
    global NSIGHT_STARTED
    torch.cuda.cudart().cudaProfilerStart()
    print("NSIGHT_STARTED")
    NSIGHT_STARTED = True

def stopNsight():
    global NSIGHT_STARTED
    torch.cuda.cudart().cudaProfilerStop()
    NSIGHT_STARTED = False

def nsight_profiler(F, name):
    def new_func(*args, **kwargs):
        if NSIGHT_STARTED:
            print("pushing", name)
            torch.cuda.nvtx.range_push(name)
        ret = F(*args, **kwargs)
        if NSIGHT_STARTED:
            torch.cuda.nvtx.range_pop()
        return ret

    return new_func

