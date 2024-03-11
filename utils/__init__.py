import random
import string
from functools import wraps
from datetime import datetime
import os


def get_time_str():
    return datetime.now().strftime("%m-%d-%y_%H-%M-%S")


def random_id(n=10):
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=n))
