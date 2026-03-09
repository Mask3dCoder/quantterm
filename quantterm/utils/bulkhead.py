"""
Bulkhead pattern for resource isolation.
Limits concurrent operations to prevent resource exhaustion.
"""
import asyncio
from typing import Dict, Any
import contextlib


class Bulkhead:
    """
    Limits concurrent operations to prevent resource exhaustion.
    """
    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queue: int = 100,
        timeout: float = 30.0
    ):
        self.name = name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = asyncio.Semaphore(max_queue)
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.active_count = 0
        self.queue_count = 0
        self.total_rejected = 0
        self.total_executed = 0
    
    @contextlib.asynccontextmanager
    async def acquire(self):
        """Acquire slot with timeout."""
        acquired_queue = False
        try:
            # Try to enter queue
            try:
                acquired_queue = await asyncio.wait_for(
                    self.queue_size.acquire(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                self.total_rejected += 1
                raise asyncio.TimeoutError(
                    f"Bulkhead {self.name}: Queue full, rejected request"
                )
            
            self.queue_count += 1
            
            # Wait for execution slot
            try:
                async with asyncio.timeout(self.timeout):
                    async with self.semaphore:
                        self.queue_count -= 1
                        self.active_count += 1
                        self.total_executed += 1
                        yield self
                        self.active_count -= 1
            except asyncio.TimeoutError:
                self.total_rejected += 1
                raise
                
        finally:
            if acquired_queue:
                self.queue_size.release()
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'active': self.active_count,
            'queued': self.queue_count,
            'max_concurrent': self.max_concurrent,
            'max_queue': self.max_queue,
            'utilization': self.active_count / self.max_concurrent if self.max_concurrent > 0 else 0,
            'total_executed': self.total_executed,
            'total_rejected': self.total_rejected,
            'rejection_rate': self.total_rejected / max(self.total_executed + self.total_rejected, 1)
        }


# Global bulkhead instances
YAHOO_BULKHEAD = Bulkhead("yahoo_api", max_concurrent=5, max_queue=20)
OPTIONS_BULKHEAD = Bulkhead("options_api", max_concurrent=3, max_queue=10)
