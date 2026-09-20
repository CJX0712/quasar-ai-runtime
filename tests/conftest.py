"""pytest 公共夹具。

三条纪律：

1. 每个测试都往 tmp_path 里写数据，绝不碰仓库内的 data/。
   一个会污染生产索引的测试套件，跑第二次就会给出不同的结果。
2. 默认走离线档（configs/default.toml）：无模型、无网络。
   需要真模型/真网络的测试必须显式 skip，不能悄悄依赖外部服务。
3. 只通过 build_services 装配，不手工 new provider——
   否则测试覆盖的是"我以为的装配方式"，不是"真实的装配方式"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for extra in (str(ROOT / "src"), str(ROOT / "tools")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from quasar.runtime.container import Services, build_services, close_services  # noqa: E402
from quasar.runtime.pipeline import Pipeline  # noqa: E402
from quasar.runtime.settings import Settings  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture
def settings() -> Settings:
    return Settings.load(ROOT / "configs" / "default.toml", root=ROOT)


@pytest.fixture
async def services(settings: Settings, tmp_path: Path):
    built = build_services(settings, data_dir=tmp_path)
    try:
        yield built
    finally:
        await close_services(built)


@pytest.fixture
async def pipeline(services: Services) -> Pipeline:
    return Pipeline(services)


@pytest.fixture
async def indexed(pipeline: Pipeline, root: Path) -> Pipeline:
    """灌入仓库自带的语料（5 篇文档 / 9 个块）。"""
    await pipeline.ingest_corpus(root / "assets" / "corpus")
    return pipeline
