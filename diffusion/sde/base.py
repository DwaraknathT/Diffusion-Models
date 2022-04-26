import abc
import torch
import torch.nn as nn


class AbstractSDE(abc.ABC):
    """Base class for diffusions"""

    @abc.abstractmethod
    def drift(
        self,
        t: float,
    ) -> None:
        """Computes the drift of the SDE at time t

        Args:
            t (float): Time index
        """
        pass

    @abc.abstractmethod
    def diffusion(
        self,
        t: float,
    ) -> None:
        """Computes the diffusion of the SDE at time index t

        Args:
            t (float): Time index
        """
        pass

    @abc.abstractmethod
    def sample(
        self,
        score_function: nn.Module,
        shape: tuple,
    ) -> tuple:
        pass
