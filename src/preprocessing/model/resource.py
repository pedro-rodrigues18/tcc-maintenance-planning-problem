from pydantic import BaseModel


class Resource(BaseModel):
    """
    Represents a resource in the maintenance planning problem.

    Attributes:
        name (str): The name of the resource.
        max (list[int]): The maximum capacity of the resource for each time period.
        min (list[int]): The minimum capacity of the resource for each time period.
    """

    name: str
    max: list[int]
    min: list[int]
    workload: list[dict[int, float]] = []
