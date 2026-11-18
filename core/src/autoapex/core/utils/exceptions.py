
class AutoApexException(Exception):
    """
    Generic AutoApex exception.
    Can be used to identify AutoApex specific exception classes.
    """
    pass


class InsufficientTime(AutoApexException):
    """
    Similar to TimeLimitExceeded, raised when the expected outcome of an operation
    would exceed the time limit, prior to exceeding the time limit.
    """
    pass


class TimeLimitExceeded(InsufficientTime):
    """
    Exception raised when the time limit has been exceeded (over budget)
    """
    pass


class NotEnoughMemoryError(AutoApexException):
    pass


class NoGPUError(AutoApexException):
    pass


class NotEnoughCudaMemoryError(AutoApexException):
    pass


class NoValidFeatures(AutoApexException):
    pass


class NoStackFeatures(NoValidFeatures):
    pass


class NotValidStacker(AutoApexException):
    pass
