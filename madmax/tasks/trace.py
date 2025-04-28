# An individual computation or communication trace
class Trace:
    def __init__(
        self,
        name,
        duration,
        t_start,
        in_dep,
    ):
        self.trace = {
            "name" : name,
            "duration" : duration,
            "t_start" : t_start,
            "t_end" : t_start + duration,
            "in_dep" : in_dep
        }