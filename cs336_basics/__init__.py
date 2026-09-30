# cs336_basics 包初始化：读取并暴露当前项目版本信息。
import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
