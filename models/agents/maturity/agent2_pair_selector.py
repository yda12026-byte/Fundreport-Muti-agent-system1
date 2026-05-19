"""
智能体2：基金对生成智能体。
基于当前后验分布，计算每对基金的期望信息增益（Expected Information Gain, EIG），
选择能最大程度减少不确定性的基金对，供用户进行二选一比较。
"""
import itertools
import numpy as np


class PairSelector:
    """最优基金对选择器，基于信息论方法驱动自适应比较。"""

    def __init__(self, n_candidates=12, random_seed=42):
        self.n_candidates = n_candidates
        self.rng = np.random.RandomState(random_seed)
        self.last_eig = 0.0
        # 缓存基金对列表，避免每轮重建
        self._all_pairs = None
        self._fund_data_cache = {}  # index -> (return, risk) 元组缓存

    @staticmethod
    def _is_dominated(ret_i, risk_i, ret_j, risk_j):
        """检查一对基金是否存在严格占优关系。

        若一只基金收益更高且风险更低，则它全面占优另一只，
        无论用户偏好如何都会选它，这种比较无信息量。
        """
        if ret_i >= ret_j and risk_i <= risk_j and (ret_i > ret_j or risk_i < risk_j):
            return True
        if ret_j >= ret_i and risk_j <= risk_i and (ret_j > ret_i or risk_j < risk_i):
            return True
        return False

    def _ensure_pairs_cache(self, funds_df):
        """缓存基金索引对和基础数据，过滤掉全面占优对。"""
        if self._all_pairs is not None and len(self._all_pairs) > 0:
            return
        indices = list(funds_df.index)
        # 缓存每个基金的数值数据（避免loc开销）
        for idx in indices:
            f = funds_df.loc[idx]
            self._fund_data_cache[idx] = (
                f['annual_return'], f['annual_std']
            )

        # 构建基金对，过滤掉严格占优对
        self._all_pairs = []
        for i, j in itertools.combinations(indices, 2):
            ret_i, risk_i = self._fund_data_cache[i]
            ret_j, risk_j = self._fund_data_cache[j]
            if not self._is_dominated(ret_i, risk_i, ret_j, risk_j):
                self._all_pairs.append((i, j))

    def select_pair(self, funds_df, preference_learner):
        """选择信息量最大的基金对。

        Parameters
        ----------
        funds_df : pd.DataFrame
            基金数据，需包含annual_return和annual_std。
        preference_learner : PreferenceLearner
            偏好学习器实例，包含当前后验分布。

        Returns
        -------
        tuple
            (fund_i_index, fund_j_index), eig_value
        """
        # 缓存基金对
        self._ensure_pairs_cache(funds_df)

        # 首轮（均匀先验）：直接选跨类别对，跳过EIG计算
        if preference_learner.n_iterations == 0:
            pair = self._first_round_pair(funds_df)
            self.last_eig = 0.5
            return pair, 0.5

        # 从缓存中随机采样候选对
        total_pairs = len(self._all_pairs)
        n_sample = min(self.n_candidates, total_pairs)
        selected = self.rng.choice(total_pairs, n_sample, replace=False)

        # 计算每个候选对的EIG，选最优
        best_pair = None
        best_eig = -1.0

        for idx in selected:
            i, j = self._all_pairs[idx]
            try:
                eig = self._expected_information_gain_fast(
                    i, j, preference_learner
                )
                if eig > best_eig:
                    best_eig = eig
                    best_pair = (i, j)
                    # EIG接近上限时提前停止
                    if best_eig > 0.5:
                        break
            except Exception:
                continue

        if best_pair is None:
            i, j = self._all_pairs[selected[0]]
            best_pair = (i, j)
            best_eig = 0.0

        self.last_eig = best_eig
        return best_pair, best_eig

    def _first_round_pair(self, funds_df):
        """首轮从已过滤的非占优对缓存中随机选择。"""
        # _ensure_pairs_cache 已在 select_pair 中调用，_all_pairs 已过滤
        if self._all_pairs:
            idx = self.rng.choice(len(self._all_pairs))
            return self._all_pairs[idx]
        # 兜底
        indices = list(funds_df.index)
        return tuple(self.rng.choice(indices, 2, replace=False))

    def _expected_information_gain_fast(self, i, j, pl):
        """基于缓存数据的快速EIG计算。

        直接使用缓存的数值元组，避免 DataFrame.loc 调用。
        """
        ret_i, risk_i = self._fund_data_cache[i]
        ret_j, risk_j = self._fund_data_cache[j]
        diff_return = ret_i - ret_j
        diff_risk = risk_i - risk_j

        # 如果两只基金完全一样，EIG为0
        if diff_return == 0.0 and diff_risk == 0.0:
            return 0.0

        posterior = pl.posterior

        # 当前后验的熵
        current_entropy = -np.sum(posterior * np.log(posterior + 1e-16))

        # 选基金i的概率（每个参数点）
        utility_diff = pl.alpha_grid_2d * diff_return - pl.beta_grid_2d * diff_risk
        prob_i_given_theta = 1.0 / (1.0 + np.exp(-np.clip(utility_diff, -50, 50)))

        # 预测概率
        pred_prob_i = float(np.sum(prob_i_given_theta * posterior))
        if pred_prob_i < 1e-10 or pred_prob_i > 0.9999999999:
            return 0.0

        # 假设用户选i → 后验 → 熵
        posterior_i = prob_i_given_theta * posterior
        posterior_i /= np.sum(posterior_i)
        entropy_i = -np.sum(posterior_i * np.log(posterior_i + 1e-16))

        # 假设用户选j → 后验 → 熵
        prob_j_given_theta = 1.0 - prob_i_given_theta
        posterior_j = prob_j_given_theta * posterior
        posterior_j /= np.sum(posterior_j)
        entropy_j = -np.sum(posterior_j * np.log(posterior_j + 1e-16))

        # 期望信息增益
        expected_entropy = pred_prob_i * entropy_i + (1.0 - pred_prob_i) * entropy_j
        return max(0.0, current_entropy - expected_entropy)

    def _expected_information_gain(self, f_i, f_j, pl):
        """计算一对基金的期望信息增益，衡量用户选择后对后验不确定性的减少量。

        信息增益 = H(当前后验) - E[ H(更新后后验) ]
        其中H为熵，期望E对用户可能的选择取平均。

        Parameters
        ----------
        f_i : Series
            基金i的信息。
        f_j : Series
            基金j的信息。
        pl : PreferenceLearner
            偏好学习器实例。

        Returns
        -------
        float
            期望信息增益值。
        """
        posterior = pl.posterior

        # 1) 当前后验的熵
        current_entropy = -np.sum(
            posterior * np.log(posterior + 1e-16)
        )

        # 2) 对每个参数点，计算选基金i的概率
        diff_return = f_i['annual_return'] - f_j['annual_return']
        diff_risk = f_i['annual_std'] - f_j['annual_std']
        utility_diff = (
            pl.alpha_grid_2d * diff_return
            - pl.beta_grid_2d * diff_risk
        )
        prob_i_given_theta = 1.0 / (1.0 + np.exp(-np.clip(utility_diff, -50, 50)))

        # 3) 预测概率（边缘化参数）
        pred_prob_i = float(np.sum(prob_i_given_theta * posterior))
        pred_prob_j = 1.0 - pred_prob_i

        if pred_prob_i < 1e-10 or pred_prob_j < 1e-10:
            return 0.0  # 结果几乎确定，没有信息量

        # 4) 假设用户选i → 更新后验 → 计算熵
        posterior_i = prob_i_given_theta * posterior
        posterior_i /= np.sum(posterior_i)
        entropy_i = -np.sum(posterior_i * np.log(posterior_i + 1e-16))

        # 5) 假设用户选j → 更新后验 → 计算熵
        prob_j_given_theta = 1.0 - prob_i_given_theta
        posterior_j = prob_j_given_theta * posterior
        posterior_j /= np.sum(posterior_j)
        entropy_j = -np.sum(posterior_j * np.log(posterior_j + 1e-16))

        # 6) 期望信息增益
        expected_entropy = pred_prob_i * entropy_i + pred_prob_j * entropy_j
        eig = current_entropy - expected_entropy

        return max(0.0, eig)

    def select_pairs_mixed(self, funds_df, preference_learner, n_pairs=3):
        """生成多个候选基金对（含EIG最高的和跨类别的），增加多样性。

        用于更丰富的用户交互场景。返回多个候选对及其EIG值。

        Returns
        -------
        list
            [((i_idx, j_idx), eig), ...]
        """
        fund_indices = list(funds_df.index)

        # 获取EIG最高的对
        best_pair, best_eig = self.select_pair(funds_df, preference_learner)

        pairs_with_eig = [(best_pair, best_eig)]

        # 添加一些跨类别的对，增加多样性
        categories = funds_df['category'].unique()
        for cat_a, cat_b in itertools.combinations(categories, 2):
            cat_a_funds = funds_df[funds_df['category'] == cat_a].index.tolist()
            cat_b_funds = funds_df[funds_df['category'] == cat_b].index.tolist()

            if len(cat_a_funds) == 0 or len(cat_b_funds) == 0:
                continue

            # 尝试找到非占优对
            for _ in range(10):
                i = self.rng.choice(cat_a_funds)
                j = self.rng.choice(cat_b_funds)

                if (i, j) == best_pair or (j, i) == best_pair:
                    continue

                ret_i, risk_i = self._fund_data_cache[i]
                ret_j, risk_j = self._fund_data_cache[j]
                if self._is_dominated(ret_i, risk_i, ret_j, risk_j):
                    continue

                f_i = funds_df.loc[i]
                f_j = funds_df.loc[j]
                try:
                    eig = self._expected_information_gain(f_i, f_j, preference_learner)
                    pairs_with_eig.append(((i, j), eig))
                except Exception:
                    continue
                break

        pairs_with_eig.sort(key=lambda x: x[1], reverse=True)
        return pairs_with_eig[:n_pairs]
