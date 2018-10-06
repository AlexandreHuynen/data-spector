import numpy as np
import pandas as pd
import plotly.figure_factory as ff

from itertools import combinations
from pandas.api import types
from scipy import stats as sp_stats


from data_spector.inspector import DataInspector
from data_spector.utils import create_freqplot


class DataDifferentiator:

    def __init__(self, **kwargs):
        """
        Inspects provided data-sets, comparing features distributions and providing convenient
        data summary. Typically used to differentiate features distribution between training and
        testing data-sets.

        Note that the order in which the data-sets are provided matters. The first provided data-set
        is considered as the 'master data-set' or reference, all the others will be compared to it.

            - Look at the 'features_stats' property to get a summary of the basic statistics of the
            each data set.
            - ...
        """

        assert len(kwargs) > 0, 'must provide at least one pandas.DataFrame'

        self.dfs = [DataInspector(df) for df in kwargs.values()]
        self.names = [name for name in kwargs.keys()]

        self.master_features = set(self.dfs[0].features)
        self.others_features = set(*[df.features for df in self.dfs[1:]])
        self.shared_features = self.master_features.intersection(self.others_features)
        self.all_features = self.master_features.union(self.others_features)

        self.features_stats = self._dfs_concat(*[df.features_stats for df in self.dfs])

    def inspect(self, which='shared'):

        lst_features = ['master', 'shared', 'all']

        if which in lst_features:
            features = getattr(self, which + '_features')
        else:
            raise ValueError('features should be in {}'.format(lst_features))

        raise NotImplemented()

    def inspect_type(self, type, plot=False):
        """
        Correlation heatmap for int and float; count of values for bool/categorical

        Args:
            type (str):
            plot (bool):

        Returns:

        """
        raise NotImplemented

    def inspect_feature(self, feature, plot=False):
        """
        Print a detailed summary of the feature and comparison among the provided data-sets

        Args:
            feature (str): the feature to inspect
            plot (bool): whether the feature distribution it to be plotted (default=False)
        """

        assert feature in self.all_features, 'requested feature is not in the provided data-sets'

        summaries, traces = [], []
        for inspector in self.dfs:
            _summary, _ = inspector._get_feature_summary(feature=feature, plot=False)
            summaries.append(_summary)

        summary = self._dfs_concat(*summaries)
        diff = self._get_feature_diff(feature=feature)

        fig = None
        if plot:
            fig = self._get_feature_plot(feature=feature)

        return summary, diff, fig

    def _get_feature_diff(self, feature):

        assert isinstance(feature, str)

        feature_type = self.features_stats.loc[feature].loc[:, 'types'].value_counts(dropna=False)
        feature_type = feature_type.index[0]  # Choose the predominant type

        try:
            diff = getattr(self, '_' + feature_type + '_feature_diff')(feature)
        except AttributeError:
            diff = None

        return diff

    def _get_feature_plot(self, feature):

        assert isinstance(feature, str)

        feature_type = self.features_stats.loc[feature].loc[:, 'types'].value_counts(dropna=False)
        feature_type = feature_type.index[0]  # Choose the predominant type

        series = {
            self.names[i]: insp.data[feature].dropna() for i, insp in enumerate(self.dfs)
            if (feature in insp.features)
        }

        if feature_type == 'numerical':
            fig = ff.create_distplot(list(series.values()), list(series.keys()))
        elif feature_type in ['categorical', 'boolean']:
            fig = create_freqplot(list(series.values()), list(series.keys()))
        else:
            fig = None

        return fig

    def _numerical_feature_diff(self, feature):
        """
        Inspects the differences among the datasets for the specified feature.

        Perform the following tests:
        -   The two-sided T-test for the null hypothesis that 2 independent samples have identical
            average (expected) values.
        -   The one-way ANOVA tests the null hypothesis that two or more groups have the same
            population mean.
        -   The Kruskal-Wallis H-test tests the null hypothesis that the population median of all
            of the groups are equal.

        Notes:
        -   The larger the t-score, the more difference there is between groups. The smaller
            the t-score, the more similarity there is between groups.
        -   The p-value is the probability that the results from your sample data occurred by
            chance.

        """

        series = {
            self.names[i]: insp.data[feature].dropna() for i, insp in enumerate(self.dfs)
            if (feature in insp.features)
        }

        tests = {
            't-test': {'func': sp_stats.ttest_ind, 'kwargs': None},
            'anova': {'func': sp_stats.f_oneway, 'kwargs': None},
            'h-test': {'func': sp_stats.kruskal, 'kwargs': None}
        }

        diff = pd.DataFrame()
        for n1, n2 in combinations(series.keys(), 2):
            for tst_name, tst in tests.items():
                if tst['kwargs']:
                    stat = tst['func'](series[n1], series[n2], **tst['kwargs'])
                else:
                    stat = tst['func'](series[n1], series[n2])
                diff = diff.append(pd.DataFrame(
                    data=[[n1, n2, tst_name, stat.statistic, stat.pvalue]]
                ))

        diff.columns = ['set_1', 'set_2', 'test', 'statistic', 'pvalue']

        return diff.reset_index(drop=True)

    def _dfs_concat(self, *args):
        return pd.concat([*args], axis=1, join='outer', sort=False, keys=[*self.names])

    def _traces_concat(self, *args):
        return None


if __name__ == '__main__':
    from data_spector.utils import load_data
    from plotly.offline import plot

    train, test = load_data('Titanic')
    data_dif = DataDifferentiator(train=train, test=test)

    _summary, _diff, _fig = data_dif.inspect_feature('Age', plot=True)
    print(_summary)
    print(_diff)
    plot(_fig)

    # _summary, _fig = data_dif.dfs[0]._get_feature_summary('Pclass', plot=True)
    # plot(_fig)



