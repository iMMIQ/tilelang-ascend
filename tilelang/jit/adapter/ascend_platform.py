# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AscendPlatformSpec:
    canonical: str
    soc_version: str
    ccec_arch: str
    npu_arch: str
    pto_arch_macro: str
    memory_model: str
    l0a_size: int
    l0b_size: int
    l1_size: int
    l0c_size: int
    ub_size: int


_PLATFORM_ALIASES = {
    "A2": "A2",
    "A3": "A3",
    "A5": "A5",
    "310P": "310P",
    "310P1": "310P",
    "310P3": "310P",
    "ASCEND310P": "310P",
    "ASCEND310P1": "310P",
    "ASCEND310P3": "310P",
}


_PLATFORM_SPECS = {
    "A2": AscendPlatformSpec(
        canonical="A2",
        soc_version="Ascend910B1",
        ccec_arch="dav-c220",
        npu_arch="dav-2201",
        pto_arch_macro="C220",
        memory_model="MEMORY_BASE",
        l0a_size=64 * 1024,
        l0b_size=64 * 1024,
        l1_size=512 * 1024,
        l0c_size=128 * 1024,
        ub_size=192 * 1024 - 256,
    ),
    "A3": AscendPlatformSpec(
        canonical="A3",
        soc_version="Ascend910_93",
        ccec_arch="dav-c220",
        npu_arch="dav-2201",
        pto_arch_macro="C220",
        memory_model="MEMORY_BASE",
        l0a_size=64 * 1024,
        l0b_size=64 * 1024,
        l1_size=512 * 1024,
        l0c_size=128 * 1024,
        ub_size=192 * 1024 - 256,
    ),
    "A5": AscendPlatformSpec(
        canonical="A5",
        soc_version="Ascend950",
        ccec_arch="dav-c310",
        npu_arch="dav-310",
        pto_arch_macro="C310",
        memory_model="REGISTER_BASE",
        l0a_size=64 * 1024,
        l0b_size=64 * 1024,
        l1_size=512 * 1024,
        l0c_size=256 * 1024,
        ub_size=256 * 1024,
    ),
    "310P": AscendPlatformSpec(
        canonical="310P",
        soc_version="Ascend310P1",
        ccec_arch="dav-m200",
        npu_arch="dav-m200",
        pto_arch_macro="M200",
        memory_model="MEMORY_BASE",
        l0a_size=64 * 1024,
        l0b_size=64 * 1024,
        l1_size=1024 * 1024,
        l0c_size=256 * 1024,
        ub_size=256 * 1024,
    ),
}


def normalize_ascend_platform(platform: str) -> str:
    if platform is None:
        return "A3"
    normalized = str(platform).upper().replace("_", "").replace("-", "")
    return _PLATFORM_ALIASES.get(normalized, platform)


def get_ascend_platform_spec(platform: str) -> AscendPlatformSpec:
    canonical = normalize_ascend_platform(platform)
    if canonical not in _PLATFORM_SPECS:
        supported = ", ".join(sorted(_PLATFORM_SPECS))
        raise ValueError(f"Unsupported Ascend platform {platform!r}. Supported platforms: {supported}")
    return _PLATFORM_SPECS[canonical]
