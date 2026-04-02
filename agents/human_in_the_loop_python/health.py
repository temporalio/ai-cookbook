import logging
import threading
import time

from temporalio.worker import (
    ActivityInboundInterceptor,
    CustomSlotSupplier,
    ExecuteActivityInput,
    FixedSizeSlotSupplier,
    Interceptor,
    SlotMarkUsedContext,
    SlotPermit,
    SlotReleaseContext,
    SlotReserveContext,
    WorkerTuner,
    WorkflowSlotInfo,
)

logger = logging.getLogger(__name__)


class HealthTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self._last_update = time.time()

    def is_busy(self) -> bool:
        return self._active > 0

    def last_update_time(self) -> int:
        return int(self._last_update)

    def _increment(self):
        with self._lock:
            self._active += 1
            self._last_update = time.time()

    def _decrement(self):
        with self._lock:
            self._active = max(0, self._active - 1)
            self._last_update = time.time()


class _ActivityInterceptor(ActivityInboundInterceptor):
    def __init__(self, next, tracker: HealthTracker):
        super().__init__(next)
        self._tracker = tracker

    async def execute_activity(self, input: ExecuteActivityInput):
        logger.debug("Activity started: %s (active: %d)", input.fn.__name__, self._tracker._active + 1)
        self._tracker._increment()
        try:
            return await self.next.execute_activity(input)
        finally:
            self._tracker._decrement()
            logger.debug("Activity finished: %s (active: %d)", input.fn.__name__, self._tracker._active)


class HealthInterceptor(Interceptor):
    def __init__(self, tracker: HealthTracker):
        self._tracker = tracker

    def intercept_activity(self, next):
        return _ActivityInterceptor(next, self._tracker)


class _WorkflowSlotSupplier(CustomSlotSupplier):
    def __init__(self, tracker: HealthTracker):
        self._tracker = tracker

    async def reserve_slot(self, ctx: SlotReserveContext) -> SlotPermit:
        return SlotPermit()

    def try_reserve_slot(self, ctx: SlotReserveContext) -> SlotPermit | None:
        return SlotPermit()

    def mark_slot_used(self, ctx: SlotMarkUsedContext) -> None:
        workflow_type = (
            ctx.slot_info.workflow_type
            if isinstance(ctx.slot_info, WorkflowSlotInfo)
            else "unknown"
        )
        logger.debug("Workflow task started: %s (active: %d)", workflow_type, self._tracker._active + 1)
        self._tracker._increment()

    def release_slot(self, ctx: SlotReleaseContext) -> None:
        if ctx.slot_info is not None:
            self._tracker._decrement()
            logger.debug("Workflow task finished (active: %d)", self._tracker._active)


def create_tuner(tracker: HealthTracker) -> WorkerTuner:
    return WorkerTuner.create_composite(
        workflow_supplier=_WorkflowSlotSupplier(tracker),
        activity_supplier=FixedSizeSlotSupplier(100),
        local_activity_supplier=FixedSizeSlotSupplier(100),
        nexus_supplier=FixedSizeSlotSupplier(100),
    )
