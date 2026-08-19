from pathlib import Path

import pandas as pd


def read_csv(file_path, encoding="utf-8", **kwargs):
    """读取 CSV 文件并返回 DataFrame。

    参数:
        file_path: CSV 文件路径。
        encoding: 文件编码，默认 utf-8。
        **kwargs: 传递给 pandas.read_csv 的其他参数。
    """
    path = _validate_file(file_path, {".csv"})
    try:
        return pd.read_csv(path, encoding=encoding, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"CSV 文件读取失败: {path}") from exc


def read_excel(file_path, sheet_name=0, **kwargs):
    """读取 Excel 文件并返回 DataFrame 或 DataFrame 字典。

    参数:
        file_path: Excel 文件路径，支持 .xlsx 和 .xls。
        sheet_name: 工作表名称、索引或列表，默认读取第一个工作表。
        **kwargs: 传递给 pandas.read_excel 的其他参数。
    """
    path = _validate_file(file_path, {".xlsx", ".xls"})
    try:
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Excel 文件读取失败: {path}") from exc


def read_data(file_path, encoding="utf-8", sheet_name=0, **kwargs):
    """根据扩展名自动读取 CSV 或 Excel 文件。

    CSV 使用 encoding；Excel 使用 sheet_name。其余参数通过 kwargs
    传给对应的 pandas 读取函数。
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return read_csv(file_path, encoding=encoding, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return read_excel(file_path, sheet_name=sheet_name, **kwargs)
    raise ValueError(f"不支持的文件格式: {suffix or '无扩展名'}")


def _validate_file(file_path, supported_suffixes):
    """检查文件是否存在且扩展名是否符合要求。"""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.suffix.lower() not in supported_suffixes:
        formats = ", ".join(sorted(supported_suffixes))
        raise ValueError(f"文件格式不支持，应为: {formats}")
    return path


