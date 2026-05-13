"""
虚拟基金数据生成模块。
生成带有收益、标准差等指标的虚拟基金数据，用于原型系统测试。
"""
import numpy as np
import pandas as pd

# 四类基金的定义参数
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


def generate_funds(funds_per_category=10, seed=42):
    """生成虚拟基金数据。

    Parameters
    ----------
    funds_per_category : int
        每类基金的数量，默认10只，总共40只。
    seed : int
        随机种子，保证结果可复现。

    Returns
    -------
    pd.DataFrame
        包含基金ID、名称、类别、年化收益、年化标准差、夏普比率等字段。
    """
    np.random.seed(seed)
    funds = []

    for cat_name, params in FUND_CATEGORIES.items():
        r_lo, r_hi = params['return_range']
        s_lo, s_hi = params['risk_range']
        for i in range(funds_per_category):
            annual_return = np.random.uniform(r_lo, r_hi)
            annual_std = np.random.uniform(s_lo, s_hi)
            funds.append({
                'id': f'F{len(funds) + 1:03d}',
                'name': f'{cat_name}_{i + 1}',
                'category': cat_name,
                'category_label': params['label'],
                'category_desc': params['desc'],
                'annual_return': round(annual_return, 4),
                'annual_std': round(annual_std, 4),
                'sharpe_ratio': round(annual_return / annual_std, 4) if annual_std > 0 else 0.0,
            })

    return pd.DataFrame(funds)
