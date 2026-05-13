"""
智能体3：偏好学习智能体。
基于离散选择模型（Logit）和贝叶斯更新理论，
通过用户对基金对的二选一结果，推断用户的收益偏好和风险厌恶程度。
"""
import numpy as np
import pandas as pd


class PreferenceLearner:
    """偏好学习器，维护关于用户偏好参数 (α, β) 的后验分布。

    用户对基金 i 的效用: U_i = α * return_i - β * risk_i
    选择概率（Logit模型）: P(选 i 而非 j) = σ(U_i - U_j)  (σ为sigmoid函数)

    使用网格近似进行贝叶斯更新。
    """

    def __init__(self, alpha_range=(0.5, 12.0), beta_range=(0.5, 12.0),
                 n_alpha=30, n_beta=30):
        # 构建参数网格
        self.alpha_grid = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        self.beta_grid = np.linspace(beta_range[0], beta_range[1], n_beta)
        # 二维网格 (n_beta, n_alpha)，用于广播计算
        self.alpha_grid_2d, self.beta_grid_2d = np.meshgrid(
            self.alpha_grid, self.beta_grid
        )

        # 均匀先验
        n_points = n_beta * n_alpha
        self.posterior = np.ones((n_beta, n_alpha)) / n_points

        self.n_iterations = 0
        self.choice_history = []

    def _sigmoid(self, x):
        """数值稳定的sigmoid函数。"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def update(self, fund_i, fund_j, choice):
        """根据用户的一次选择更新后验分布。

        Parameters
        ----------
        fund_i : Series/ dict
            基金i的信息，需包含annual_return和annual_std。
        fund_j : Series/ dict
            基金j的信息。
        choice : int
            0 表示选基金i，1 表示选基金j。
        """
        diff_return = fund_i['annual_return'] - fund_j['annual_return']
        diff_risk = fund_i['annual_std'] - fund_j['annual_std']

        # 计算每个参数点下选基金i的概率
        utility_diff = (
            self.alpha_grid_2d * diff_return
            - self.beta_grid_2d * diff_risk
        )
        prob_choose_i = self._sigmoid(utility_diff)

        # 根据实际选择确定似然
        if choice == 0:
            likelihood = prob_choose_i
        else:
            likelihood = 1.0 - prob_choose_i

        # 贝叶斯更新: 后验 ∝ 似然 × 先验
        self.posterior *= likelihood
        # 归一化
        posterior_sum = np.sum(self.posterior)
        if posterior_sum > 0:
            self.posterior /= posterior_sum
        else:
            # 数值保护：如果似然在所有参数点上都接近0，回退到均匀分布
            self.posterior = np.ones_like(self.posterior) / self.posterior.size

        self.n_iterations += 1
        self.choice_history.append({
            'fund_i': fund_i['id'],
            'fund_j': fund_j['id'],
            'choice': choice,
            'diff_return': diff_return,
            'diff_risk': diff_risk,
        })

    def get_choice_probability(self, fund_i, fund_j):
        """预测用户在当前偏好下选择基金i而非基金j的概率。"""
        diff_return = fund_i['annual_return'] - fund_j['annual_return']
        diff_risk = fund_i['annual_std'] - fund_j['annual_std']
        utility_diff = (
            self.alpha_grid_2d * diff_return
            - self.beta_grid_2d * diff_risk
        )
        prob = self._sigmoid(utility_diff)
        return float(np.sum(prob * self.posterior))

    def get_parameter_estimates(self):
        """获取参数的后验均值和标准差。"""
        alpha_mean = float(np.sum(self.alpha_grid_2d * self.posterior))
        beta_mean = float(np.sum(self.beta_grid_2d * self.posterior))

        alpha_var = float(np.sum(
            (self.alpha_grid_2d - alpha_mean) ** 2 * self.posterior
        ))
        beta_var = float(np.sum(
            (self.beta_grid_2d - beta_mean) ** 2 * self.posterior
        ))

        return {
            'alpha_mean': round(alpha_mean, 3),
            'alpha_std': round(np.sqrt(alpha_var), 3),
            'beta_mean': round(beta_mean, 3),
            'beta_std': round(np.sqrt(beta_var), 3),
        }

    def get_category_expected_utility(self, funds_df):
        """计算每类基金的期望效用（后验平均）。"""
        # 用DataFrame中已有的类别
        categories = funds_df[['category', 'category_label']].drop_duplicates()
        results = []

        for _, row in categories.iterrows():
            cat = row['category']
            cat_funds = funds_df[funds_df['category'] == cat]
            if len(cat_funds) == 0:
                continue

            exp_utils = []
            for _, fund in cat_funds.iterrows():
                utility = (
                    self.alpha_grid_2d * fund['annual_return']
                    - self.beta_grid_2d * fund['annual_std']
                )
                exp_utils.append(float(np.sum(utility * self.posterior)))

            results.append({
                'category': cat,
                'category_label': row['category_label'],
                'avg_expected_utility': round(np.mean(exp_utils), 4),
                'std_expected_utility': round(np.std(exp_utils), 4),
            })

        results.sort(key=lambda x: x['avg_expected_utility'], reverse=True)
        return results

    def is_converged(self):
        """判断后验是否已经足够集中，可以做推荐了。

        判断条件：
        1. 最少4轮交互
        2. 参数估计的变异系数(标准差/均值) < 25%
        3. 最多12轮交互后强制收敛
        """
        if self.n_iterations < 4:
            return False
        if self.n_iterations >= 12:
            return True

        params = self.get_parameter_estimates()
        alpha_cv = params['alpha_std'] / params['alpha_mean']
        beta_cv = params['beta_std'] / params['beta_mean']
        avg_cv = (alpha_cv + beta_cv) / 2

        return avg_cv < 0.25

    def recommend(self, funds_df, top_n=3):
        """生成最终的基金推荐。

        基于后验期望效用对基金排序，推荐得分最高的基金及最匹配的类别。

        Parameters
        ----------
        funds_df : pd.DataFrame
            包含所有基金信息的DataFrame。
        top_n : int
            推荐基金的数量。

        Returns
        -------
        dict
            包含推荐结果的字典。
        """
        # 计算每只基金的后验期望效用
        fund_utilities = []
        for _, fund in funds_df.iterrows():
            utility = (
                self.alpha_grid_2d * fund['annual_return']
                - self.beta_grid_2d * fund['annual_std']
            )
            exp_utility = float(np.sum(utility * self.posterior))
            fund_utilities.append((fund, exp_utility))

        fund_utilities.sort(key=lambda x: x[1], reverse=True)

        # 类别偏好排名
        category_ranking = self.get_category_expected_utility(funds_df)
        best_category = category_ranking[0] if category_ranking else None

        return {
            'best_category': best_category,
            'category_ranking': category_ranking,
            'top_funds': [f[0] for f in fund_utilities[:top_n]],
            'sorted_funds': fund_utilities,
            'n_iterations': self.n_iterations,
        }
