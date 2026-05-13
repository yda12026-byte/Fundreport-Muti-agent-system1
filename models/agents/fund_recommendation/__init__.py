"""
智能基金推荐多智能体系统

包含三个协作智能体：
  - agent1_classifier : 基金分类智能体（聚类分类）
  - agent2_pair_selector : 基金对生成智能体（期望信息增益）
  - agent3_preference_learner : 偏好学习智能体（贝叶斯更新）
"""
from .fund_data import generate_funds, FUND_CATEGORIES
from .agent1_classifier import FundClassifier
from .agent2_pair_selector import PairSelector
from .agent3_preference_learner import PreferenceLearner

__all__ = [
    'generate_funds',
    'FUND_CATEGORIES',
    'FundClassifier',
    'PairSelector',
    'PreferenceLearner',
]
