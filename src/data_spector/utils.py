import pandas as pd
import plotly.graph_objs as go

from pathlib import Path

PKG_DIR = Path(__file__).parent.parent.parent


def load_data(name):
    data = []
    for set in ['train', 'test']:
        data.append(pd.read_csv(PKG_DIR / 'data' / name / '{}.csv'.format(set)))

    return data


def create_freqplot(data, labels=None):

    assert (labels is None) or (len(data) == len(labels))
    feature_name = data[0].name

    x = labels or [feature_name]

    freqs = []
    for i, series in enumerate(data):
        _freqs = (series.value_counts(dropna=False) / len(series))
        _freqs.name = labels[i] if labels else feature_name
        freqs.append(_freqs)

    freqs = pd.concat(freqs, axis=1, join='outer', sort=False)

    traces = []
    for cat in list(freqs.index):
        traces.append(
            go.Bar(
                x=x,
                y=freqs.loc[cat, :],
                name=cat
            )
        )

    return go.Figure(data=traces, layout=go.Layout(barmode='stack'))

