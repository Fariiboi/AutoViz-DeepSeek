"""Message bus for inter-agent communication in the visualization pipeline.

Provides async publish/consume queues for VizTask (main → sub-agent)
and VizResult (sub-agent → consumer).
"""

import asyncio
from dataclasses import dataclass, field


@dataclass
class VizTask:
    """A visualization task assigned by the main agent to a sub-agent."""

    task_id: str
    chart_type: str
    features: list[str]
    title: str
    description: str
    replan_generation: int = 0  # 0 = initial plan; incremented for critic-driven replacement tasks


@dataclass
class VizResult:
    """Result produced by a sub-agent after completing a visualization task."""

    task_id: str
    chart_type: str
    features: list[str]
    status: str  # "done" | "skipped" | "error"
    image_path: str | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class MessageBus:
    """Dual asyncio.Queue bus for visualization pipeline.

    - task_queue:   main agent → sub-agents  (VizTask)
    - result_queue: sub-agents → consumer    (VizResult)
    """

    def __init__(self) -> None:
        self._task_queue: asyncio.Queue[VizTask] = asyncio.Queue()
        self._result_queue: asyncio.Queue[VizResult] = asyncio.Queue()

    async def publish_task(self, task: VizTask) -> None:
        await self._task_queue.put(task)

    async def consume_task(self) -> VizTask:
        return await self._task_queue.get()

    async def publish_result(self, result: VizResult) -> None:
        await self._result_queue.put(result)

    async def consume_result(self) -> VizResult:
        return await self._result_queue.get()

    def result_queue_size(self) -> int:
        return self._result_queue.qsize()
