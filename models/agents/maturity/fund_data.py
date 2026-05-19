"""
真实基金数据加载模块。
从外部数据源（CSV文件等）加载基金数据，包含收益、标准差等指标。
"""
import pandas as pd
from pathlib import Path

# 四类基金的定义参数（用于 Agent1 聚类后的类别命名）
FUND_CATEGORIES = {
    '优质型': {
        'return_range': (0.12, 0.25),
        'risk_range': (0.02, 0.10),
        'label': '高收益低风险',
        'desc': '收益高且波动小，表现优异的基金'
    },
    '增长型': {
        'return_range': (0.18, 0.35),
        'risk_range': (0.15, 0.30),
        'label': '高收益高风险',
        'desc': '收益高但波动大，适合进取型投资者'
    },
    '稳健型': {
        'return_range': (0.03, 0.10),
        'risk_range': (0.02, 0.10),
        'label': '低收益低风险',
        'desc': '收益稳定波动小，适合保守型投资者'
    },
    '风险型': {
        'return_range': (0.03, 0.10),
        'risk_range': (0.15, 0.30),
        'label': '低收益高风险',
        'desc': '收益低但波动大，通常不建议长期持有'
    },
}

# 真实数据 CSV 必须包含的列
REQUIRED_COLUMNS = ['id', 'name', 'annual_return', 'annual_std']


def load_funds(source):
    """从外部 CSV 数据源加载真实基金数据。

    Parameters
    ----------
    source : str or list of str
        数据源路径，支持以下形式：
        - 单个 CSV 文件路径，如 'data/funds.csv'
        - 多个 CSV 文件路径的列表，如 ['data/a.csv', 'data/b.csv']
        - 目录路径，自动加载该目录下所有 .csv 文件

    Returns
    -------
    pd.DataFrame
        包含以下字段的基金数据：
        - id            : str   基金唯一标识
        - name          : str   基金名称
        - annual_return : float 年化收益率（如 0.15 表示 15%）
        - annual_std    : float 年化标准差（如 0.08 表示 8%）
        - sharpe_ratio  : float 夏普比率（若 CSV 未提供则自动计算）

    Raises
    ------
    FileNotFoundError
        指定的文件或目录不存在，或目录中无 CSV 文件。
    ValueError
        CSV 数据缺少必要列时抛出。

    Examples
    --------
    >>> df = load_funds('data/funds.csv')
    >>> df = load_funds(['data/stock_funds.csv', 'data/bond_funds.csv'])
    >>> df = load_funds('data/funds/')
    """
    # ---- 解析 source → DataFrame ----
    if isinstance(source, list):
        dfs = [pd.read_csv(f) for f in source]
        df = pd.concat(dfs, ignore_index=True)
    else:
        source_path = Path(source)
        if source_path.is_dir():
            csv_files = sorted(source_path.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError(f'目录 {source} 中没有找到 CSV 文件')
            dfs = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(source)

    # ---- 校验必要列 ----
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f'数据缺少必要列: {missing}。'
            f'CSV 文件需包含: {REQUIRED_COLUMNS}'
        )

    # ---- 自动计算夏普比率（若未提供） ----
    if 'sharpe_ratio' not in df.columns:
        df['sharpe_ratio'] = (
            df['annual_return'] / df['annual_std'].replace(0, float('nan'))
        ).round(4)
        df['sharpe_ratio'] = df['sharpe_ratio'].fillna(0.0)

    return df
