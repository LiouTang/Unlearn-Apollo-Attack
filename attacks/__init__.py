from .attack_framework import Attack_Framework
from .Apollo import Apollo
from .ULiRA import ULiRA

def get_attack(name, **kwargs) -> Attack_Framework:
    return eval(name)(**kwargs)