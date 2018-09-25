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
        self.others_features = set([df.features for df in self.dfs[1:]])
        self.shared_features = self.master_features.union(self.others_features)

        self.features_stats = self._dfs_concat([df.features_stats for df in self.dfs])

    def inspect_feature(self, features, plot=False):
        """
        Print a detailed summary of the feature and comparison among the provided data-sets

        Args:
            features (str or list of str): the feature(s) to inspect
            plot (bool): whether the feature distribution it to be plotted (default=False)

        """

        if isinstance(features, list):
            for feature in features:
                self.inspect_feature(features=feature, plot=plot)

        elif not isinstance(features, str):
            raise ValueError('features should either be a string or list of string, '
                             'provided {}'.format(features))

    def _get_feature_summary(self, feature, common=True, plot=False):
        feature_type = self.features_stats[self.train_name].loc[feature, 'types']

        summary = getattr(self, '_get_' + feature_type + '_summary')(feature, common=common)

        if plot:
            getattr(self, '_plot_' + feature_type + '_summary')(feature, common=common)

        return summary

    def _numeric_feature_comparison(self, feature):

        summary = self._train_test_concat(
            self._numeric_feature_summary('train'),
            self._numeric_feature_summary('test'),
        )

        ttest = sp_stats.ttest_ind(self.train[feature], self.test[feature])
        # The larger the t-score, the more difference there is between groups. The smaller
        # the t-score, the more similarity there is between groups. The p-value is the
        # probability that the results from your sample data occurred by chance.

        c = {'t-statistic': ttest[0], 'two-tailed p-value': ttest[1]}

        return summary

    def _dfs_concat(self, *args):
        return pd.concat([*args], axis=1, join='outer', sort=False, keys=[*self.names])