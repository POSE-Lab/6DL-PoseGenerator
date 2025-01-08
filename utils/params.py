class RenderParams:
    def __init__(self, params_dict):
        # dynamically create class attributes from given params_dict
        for key, value in params_dict.items():
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)
        