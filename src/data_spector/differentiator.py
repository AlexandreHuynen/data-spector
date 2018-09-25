import numpy as np
from scipy import stats as sp_stats
import pandas as pd
from pandas.api import types


from .inspector import DataInspector


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
            _summary, _trace = inspector._get_feature_summary(feature=feature, plot=plot)
            summaries.append(_summary)
            traces.append(_trace)

        summary = self._dfs_concat(*summaries)
        diff = self._get_feature_diff(feature=feature)

        if plot:
            traces = self._traces_concat(*traces)

        return summary, diff, traces

    def _get_feature_diff(self, feature):

        assert isinstance(feature, str)

        feature_type = self.features_stats.loc[feature, 'types']
        diff = getattr(self, '_' + feature_type + '_feature_diff')(feature)

        return diff

    def _numerical_feature_diff(self, feature):

        diff = None
        series = {
            self.names[i]: insp.data[feature] for i, insp in enumerate(self.dfs)
            if (feature in insp.features)
        }

        ttest = sp_stats.ttest_ind(*series.values())
        # The larger the t-score, the more difference there is between groups. The smaller
        # the t-score, the more similarity there is between groups. The p-value is the
        # probability that the results from your sample data occurred by chance.

        c = {'t-statistic': ttest[0], 'two-tailed p-value': ttest[1]}

        return diff

    def _dfs_concat(self, *args):
        return pd.concat([*args], axis=1, join='outer', sort=False, keys=[*self.names])

    def _traces_concat(self, *args):
        return None
