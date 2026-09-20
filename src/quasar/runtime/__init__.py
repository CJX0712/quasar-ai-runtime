"""L3 · 编排层：配置、装配、门面、可观测性、自诊断。"""

from .container import Services, build_services, close_services
from .doctor import CheckResult, DoctorReport, environment_snapshot, run_doctor
from .pipeline import HealthReport, Pipeline
from .settings import Settings
from .telemetry import CountingSink, ManagedTracer

__all__ = [
    "Settings",
    "Services",
    "build_services",
    "close_services",
    "Pipeline",
    "HealthReport",
    "ManagedTracer",
    "CountingSink",
    "run_doctor",
    "DoctorReport",
    "CheckResult",
    "environment_snapshot",
]
