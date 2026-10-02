"""Shared problem specification; evaluator implements dynamics independently."""
from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class Problem:
    horizon:int=20
    dt:float=.2
    strength:float=.12
    process_sigma:float=.025
    initial_sigma:float=.015
    goal_half:float=.30
    control_limit:float=2.5
    obstacles:tuple=((0.,0.,.43),)
    name:str='circle'
    def as_dict(self):return asdict(self)
