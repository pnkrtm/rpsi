class BadCalcBaseException(Exception):
    def __init__(self):
        super().__init__()


class BadRPModelException(BadCalcBaseException):
    def __init__(self):
        super().__init__()
