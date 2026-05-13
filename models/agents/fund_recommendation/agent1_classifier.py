"""
智能体1：基金分类智能体。
根据基金的收益和风险指标，将基金分为四类：
- 优质型（高收益低风险）
- 增长型（高收益高风险）
- 稳健型（低收益低风险）
- 风险型（低收益高风险）
"""
import numpy as np
import pandas as pd


class FundClassifier:


    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.funds_df = None

    def fit(self, funds_df):
        """对基金进行聚类分类。

        使用年化收益和年化标准差两个维度，通过KMeans算法聚类，
        然后根据聚类中心的收益和风险特征自动命名类别。
        """
        self.funds_df = funds_df.copy()
        X = funds_df[['annual_return', 'annual_std']].values

        # 简单实现KMeans（避免依赖sklearn，纯numpy实现）
        clusters = self._kmeans(X, self.n_clusters, seed=42)

        # 计算聚类中心
        centroids = np.array([
            X[clusters == k].mean(axis=0) if np.any(clusters == k) else np.zeros(2)
            for k in range(self.n_clusters)
        ])

        # 根据聚类中心的收益和风险确定类别名称
        # 计算所有中心的均值作为分割阈值
        mean_return = centroids[:, 0].mean()
        mean_risk = centroids[:, 1].mean()

        cluster_to_category = {}
        for k, (ret, risk) in enumerate(centroids):
            high_return = ret >= mean_return
            high_risk = risk >= mean_risk
            if high_return and not high_risk:
                cluster_to_category[k] = '优质型'
            elif high_return and high_risk:
                cluster_to_category[k] = '增长型'
            elif not high_return and not high_risk:
                cluster_to_category[k] = '稳健型'
            else:
                cluster_to_category[k] = '风险型'

        self.funds_df['cluster'] = clusters
        self.funds_df['category'] = [cluster_to_category[c] for c in clusters]

        # 补充类别标签
        label_map = {
            '优质型': '高收益低风险',
            '增长型': '高收益高风险',
            '稳健型': '低收益低风险',
            '风险型': '低收益高风险',
        }
        desc_map = {
            '优质型': '收益高且波动小，表现优异的基金',
            '增长型': '收益高但波动大，适合进取型投资者',
            '稳健型': '收益稳定波动小，适合保守型投资者',
            '风险型': '收益低但波动大，通常不建议长期持有',
        }
        self.funds_df['category_label'] = self.funds_df['category'].map(label_map)
        self.funds_df['category_desc'] = self.funds_df['category'].map(desc_map)

        return self

    def _kmeans(self, X, k, seed=42, max_iters=100):
        """简易KMeans实现。"""
        rng = np.random.RandomState(seed)
        n_samples = X.shape[0]

        # 随机初始化中心
        indices = rng.choice(n_samples, k, replace=False)
        centers = X[indices].copy()

        for _ in range(max_iters):
            # 分配每个点到最近的中心
            distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            labels = np.argmin(distances, axis=1)

            # 更新中心
            new_centers = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                for i in range(k)
            ])

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        return labels

    def get_category_summary(self):
        """返回各类基金的统计摘要。"""
        summary = self.funds_df.groupby(['category', 'category_label']).agg(
            基金数量=('id', 'count'),
            平均年化收益=('annual_return', 'mean'),
            平均年化标准差=('annual_std', 'mean'),
            平均夏普比率=('sharpe_ratio', 'mean'),
        ).round(4).reset_index()
        summary.columns = ['类别', '特征', '基金数量', '平均年化收益', '平均年化标准差', '平均夏普比率']
        return summary

    def get_funds_by_category(self, category=None):
        """按类别获取基金列表。"""
        if category:
            return self.funds_df[self.funds_df['category'] == category]
        return {cat: group for cat, group in self.funds_df.groupby('category')}
