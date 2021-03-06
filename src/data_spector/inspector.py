import numpy as np
import pandas as pd

from pandas.api import types
import plotly.figure_factory as ff


from data_spector.utils import create_freqplot


class DataInspector:

    def __init__(self, data, exclude=None):
        """"

        Args:
            data (pandas.DataFrame): data-set to inspect
            exclude (str or list of str): features to exclude from inspector
        """

        columns = data.columns

        if exclude:
            if isinstance(exclude, list):
                to_drop = [col for col in exclude if col in columns]
            elif isinstance(exclude, str):
                to_drop = exclude if exclude in columns else None
            else:
                raise 'exclude must either be a string or a list of string, ' \
                      'provided {}'.format(exclude)

            data = data.drop(columns=to_drop)

        self.data = data
        self.data_corr = data.corr()
        self.features = list(data.columns)

        self.features_stats = self._get_features_stats()

    def _get_features_stats(self):
        """Compute features statistic of the pandas.DataFrame df"""
        counts = self.data.count()
        counts.name = 'counts'

        uniques = pd.Series(
            data={col: self.data[col].nunique() for col in self.features}, name='uniques'
        )

        length = len(self.data)
        missing = ((length - counts)/length).apply(lambda x: '{0:,.2f}%'.format(100 * x))
        missing.name = 'missing'

        stats = pd.concat([counts, uniques, missing], axis=1)
        stats['dtypes'] = [types.infer_dtype(self.data[col]) for col in self.features]

        stats['types'] = 'other'
        for tps, features in self._get_implied_type(stats).items():
            stats.loc[features, 'types'] = tps

        return stats

    @staticmethod
    def _get_implied_type(stats, cat_threshold=0.01):
        """
        Deduce 'type' based on features distribution, supported implied types are
            ['constant', 'boolean', 'unique', 'datetime', 'numeric', 'categorical', 'other']
        """

        def get_remaining_features(table, types_dic):
            remaining_features = set(table.index).difference(
                set([val for vals in types_dic.values() for val in vals])
            )
            return table.loc[list(remaining_features)]

        features_type = dict()
        features_type['constant'] = stats['uniques'][stats['uniques'] == 1].index
        features_type['boolean'] = stats['uniques'][stats['uniques'] == 2].index

        _stats = get_remaining_features(stats, features_type)
        features_type['datetime'] = _stats['uniques'][
            _stats['dtypes'].isin(['datetime64', 'datetime', 'date', 'timedelta64',
                                  'timedelta', 'time', 'period'])
        ].index

        _stats = get_remaining_features(stats, features_type)
        features_type['unique'] = _stats['uniques'][_stats['uniques'] == _stats['counts']].index

        _stats = get_remaining_features(stats, features_type)
        features_type['categorical'] = _stats['uniques'][
            (
                    (_stats['dtypes'] == 'integer') |
                    (_stats['dtypes'] == 'string') |
                    (_stats['dtypes'] == 'mixed')
            )
            & (_stats['uniques'] / _stats['counts'] <= cat_threshold)
            ].index

        _stats = get_remaining_features(stats, features_type)
        features_type['numerical'] = _stats['uniques'][(
            (_stats['dtypes'] == 'floating') | (_stats['dtypes'] == 'integer')
        )].index

        return features_type

    def _get_feature_summary(self, feature, plot=False):

        assert isinstance(feature, str)
        assert isinstance(plot, bool)

        summary, fig = None, None
        if feature in self.features:
            feature_type = self.features_stats.loc[feature, 'types']

            summary, fig = getattr(self, '_' + feature_type + '_feature_summary')(feature, plot)

        return summary, fig

    def _numerical_feature_summary(self, feature, plot=False, top_corr=3):
        """Return a simple summary of a numerical feature"""
        series = self.data[feature]
        summary = pd.Series(data={
            'type': 'numeric',
        })

        summary = summary.append(
            self.features_stats.loc[feature, ['counts', 'uniques', 'missing']]
        ).append(
            series.describe()[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        )

        corr = self.data_corr[feature].drop(feature)
        corr = np.fabs(corr).sort_values(ascending=False, inplace=False)
        summary = summary.append(pd.Series(data={
            'top_correlations': self._percentage_format(corr[:top_corr]).to_dict()
        }))

        fig = None
        if plot:
            fig = ff.create_distplot([series.dropna()], [feature])

        return summary, fig

    def _categorical_feature_summary(self, feature, plot=False):
        """Return a simple summary of a categorical feature"""
        series = self.data[feature]
        summary = pd.Series(data={
            'type': 'categorical',
            'values': list(series.unique()),
        })

        summary = summary.append(
            self.features_stats.loc[feature, ['counts', 'uniques', 'missing']]
        )

        value_counts = (series.value_counts(dropna=False) / len(series)).reset_index()
        value_counts = value_counts.rename(columns={'index': feature, feature: 'freq'})

        value_counts['freq'] = self._percentage_format(value_counts['freq'])
        if len(value_counts) >= 4:
            top = value_counts[:2].set_index(feature)['freq'].to_dict()
            flop = value_counts[-2:].set_index(feature)['freq'].to_dict()

            summary = summary.append(pd.Series(data={'top': top, 'flop': flop}))

        else:
            top = value_counts.set_index(feature)['freq'].to_dict()
            summary = summary.append(pd.Series(data={'top': top}))

        fig = None
        if plot:
            fig = create_freqplot([series], None)

        return summary, fig

    def _datetime_feature_summary(self, feature, plot=False):
        """Return a simple summary of a datetime feature"""
        series = self.data[feature]
        summary = pd.Series(data={
            'type': 'datetime',
        }).append(series.describe())

        fig = None
        if plot:
            fig = dict()

        return summary, fig

    def _boolean_feature_summary(self, feature, plot=False):
        """Return a simple summary of a boolean feature"""
        series = self.data[feature]
        summary = pd.Series(data={
            'type': 'categorical',
            'values': list(series.unique()),
        })

        summary = summary.append(
            self.features_stats.loc[feature, ['counts', 'missing']]
        )

        value_counts = (series.value_counts(dropna=False) / len(series)).reset_index()
        value_counts = value_counts.rename(columns={'index': feature, feature: 'freq'})
        value_counts['freq'] = self._percentage_format(value_counts['freq'])

        summary = summary.append(
            pd.Series(data=value_counts.set_index(feature)['freq'].to_dict())
        )

        fig = None
        if plot:
            fig = dict()

        return summary, fig

    def _unique_feature_summary(self, feature, plot=False):
        """Return a simple summary of a unique feature"""
        # series = self.data[feature]
        summary = pd.Series(data={
            'type': 'unique',
            'dtype': self.features_stats[feature]['dtype']
        }).append(
            self.features_stats.loc[feature, ['counts', 'missing']]
        )

        fig = None
        if plot:
            fig = dict()

        return summary, fig

    def _constant_feature_summary(self, feature, plot=False):
        """Return a simple summary of a constant feature"""
        series = self.data[feature]

        summary = pd.Series(data={
            'type': 'constant', 'value': list(series.unique())
        })

        summary = summary.append(
            self.features_stats.loc[feature, ['counts', 'missing']]
        )

        fig = None
        if plot:
            fig = dict()

        return summary, fig

    def _other_feature_summary(self, feature, plot=False):
        """Return a simple summary of a constant feature"""
        series = self.data[feature]

        summary = pd.Series(data={
            'type': 'other'
        }).append(series.describe())

        fig = None
        if plot:
            fig = dict()

        return summary, fig

    @staticmethod
    def _percentage_format(series, digits=2):
        """Format series to percentage"""
        template = '{0:,.' + str(digits) + 'f}%'
        return series.apply(lambda x: template.format(100 * x))
