"""GPU memory and utilization monitoring via pynvml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.gateway.schemas import GPUStats
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUPoller:
    """GPU statistics poller using pynvml."""
    
    device_id: int = 0
    _initialized: bool = False
    _handle: Optional[object] = None
    
    def __post_init__(self) -> None:
        """Initialize NVML on first use."""
        self._initialized = False
        self._handle = None
    
    def initialize(self) -> bool:
        """Initialize NVML library. Returns True if successful."""
        if self._initialized:
            return True
        
        try:
            try:
                import nvidia_ml_py as pynvml
            except ImportError:
                import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if self.device_id >= device_count:
                logger.warning(
                    "GPU device_id out of range",
                    device_id=self.device_id,
                    available=device_count,
                )
                return False
            
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self._initialized = True
            logger.info(
                "NVML initialized",
                device_id=self.device_id,
                name=self._get_device_name(),
            )
            return True
            
        except ImportError:
            logger.warning("pynvml not installed, GPU monitoring disabled")
            return False
        except Exception as e:
            logger.error("Failed to initialize NVML", error=str(e))
            return False
    
    def _get_device_name(self) -> str:
        """Get device name."""
        if not self._initialized or self._handle is None:
            return "Unknown"
        try:
            import pynvml
            return pynvml.nvmlDeviceGetName(self._handle).decode("utf-8")
        except Exception:
            return "Unknown"
    
    def get_stats(self) -> Optional[GPUStats]:
        """Get current GPU statistics."""
        if not self.initialize():
            return None
        
        try:
            import pynvml
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            total_mb = mem_info.total / (1024 * 1024)
            used_mb = mem_info.used / (1024 * 1024)
            free_mb = mem_info.free / (1024 * 1024)
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            utilization = util.gpu
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = None
            
            # Power
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
            except Exception:
                power_draw = None
                power_limit = None
            
            return GPUStats(
                device_id=self.device_id,
                name=self._get_device_name(),
                total_memory_mb=total_mb,
                used_memory_mb=used_mb,
                free_memory_mb=free_mb,
                utilization_percent=float(utilization),
                temperature_c=float(temp) if temp is not None else None,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                timestamp=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.error("Failed to get GPU stats", error=str(e))
            return None
    
    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
                self._initialized = False
                self._handle = None
            except Exception as e:
                logger.error("Error shutting down NVML", error=str(e))


class GPUMonitor:
    """Multi-GPU monitoring manager."""
    
    def __init__(self, device_ids: list[int] | None = None) -> None:
        """Initialize GPU monitor.
        
        Args:
            device_ids: List of GPU device IDs to monitor. If None, monitors all available.
        """
        self.pollers: list[GPUPoller] = []
        self._initialized = False
        
        if device_ids is None:
            # Auto-detect GPUs
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                device_ids = list(range(device_count))
                pynvml.nvmlShutdown()
            except Exception:
                device_ids = []
        
        for device_id in device_ids:
            poller = GPUPoller(device_id=device_id)
            if poller.initialize():
                self.pollers.append(poller)
        
        self._initialized = len(self.pollers) > 0
        logger.info(
            "GPU monitor initialized",
            num_gpus=len(self.pollers),
            device_ids=[p.device_id for p in self.pollers],
        )
    
    def get_all_stats(self) -> list[GPUStats]:
        """Get statistics for all monitored GPUs."""
        stats = []
        for poller in self.pollers:
            stat = poller.get_stats()
            if stat is not None:
                stats.append(stat)
        return stats
    
    def get_total_memory_stats(self) -> dict[str, float]:
        """Get aggregate memory statistics across all GPUs."""
        total_mb = 0.0
        used_mb = 0.0
        free_mb = 0.0
        
        for poller in self.pollers:
            stat = poller.get_stats()
            if stat is not None:
                total_mb += stat.total_memory_mb
                used_mb += stat.used_memory_mb
                free_mb += stat.free_memory_mb
        
        return {
            "total_memory_mb": total_mb,
            "used_memory_mb": used_mb,
            "free_memory_mb": free_mb,
            "utilization_percent": (used_mb / total_mb * 100) if total_mb > 0 else 0.0,
        }
    
    def shutdown(self) -> None:
        """Shutdown all pollers."""
        for poller in self.pollers:
            poller.shutdown()
        self.pollers = []
        self._initialized = False
