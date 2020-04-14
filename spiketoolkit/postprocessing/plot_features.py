from plotly import graph_objs as go
from plotly.subplots import make_subplots
import numpy as np


def plot_metrics(sorting, unit_id, invert_template=False, savedir=None):
    idx = [i for i, u in enumerate(sorting.get_unit_ids()) if u == unit_id][0]

    unit_props = sorting.get_unit_property_names(unit_id)
    assert 'template' in unit_props, 'template should be in unit properties'
    if 'trough_to_peak_duration' in unit_props:
        ttp = sorting.get_unit_property(unit_id, 'trough_to_peak_duration')
    else:
        ttp = None
    if 'peak_index_in_template' in unit_props:
        pkidx = sorting.get_unit_property(unit_id, 'peak_index_in_template')
    else:
        pkidx = None

    if 'through_index_in_template' in unit_props:
        thidx = sorting.get_unit_property(unit_id, 'through_index_in_template')
    else:
        thidx = None

    template = sorting.get_unit_property(unit_id, 'template')
    time = (np.arange(0, template.shape[1]) / sorting.get_sampling_frequency()) * 1000

    row_idx = 1
    col_idx = 1
    fig = make_subplots(rows=2,
                        cols=2,
                        )

    sl = True
    sub_title = []
    for template_idx in range(template.shape[0]):

        if ttp is not None:
            temp_ttp = ttp[template_idx]

        if pkidx is not None:
            pk_time = time[pkidx[template_idx]]

        if thidx is not None:
            th_time = time[thidx[template_idx]]

        if invert_template:
            y = -template[template_idx, :]
        else:
            y = template[template_idx, :]

        ymax = np.max(y) + 20
        ymin = np.min(y) - 20

        fig.add_scatter(
            x=time,
            y=y,
            row=row_idx,
            col=col_idx,
            line=dict(
                color='blue'
            ),
            showlegend=False,
        )

        if pkidx is not None:
            fig.add_scatter(
                x=np.ones(2) * pk_time,
                y=[ymin, ymax],
                mode='lines',
                row=row_idx,
                col=col_idx,
                line=dict(
                    color='green',
                ),
                name='peak',
                showlegend=sl,
            )

        if thidx is not None:
            fig.add_scatter(
                x=np.ones(2) * th_time,
                y=[ymin, ymax],
                row=row_idx,
                col=col_idx,
                line=dict(
                    color='red'
                ),
                mode='lines',
                name='trough',
                showlegend=sl,
            )

        if ttp is not None and thidx is not None and pkidx is not None:
            fig.add_annotation(
                dict(
                    x=pk_time,
                    y=ymax,
                    ax=-((pk_time + th_time) / 2) - 20,
                    ay=-20,
                    text=f'{temp_ttp * 1000:.2f}',
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.add_annotation(
                dict(
                    x=th_time,
                    y=ymax,
                    text='',
                    ax=((pk_time + th_time) / 2),
                    ay=-10,
                ),
                row=row_idx,
                col=col_idx,
            )

        fig.update_yaxes(
            range=[ymin, ymax],
            row=row_idx,
            col=col_idx,
            tickvals=[],
        )

        if row_idx == 2:
            fig.update_xaxes(
                title_text='time (ms)',
                row=row_idx,
                col=col_idx,
            )
        else:
            fig.update_xaxes(
                tickvals=[],
                row=row_idx,
                col=col_idx,
            )

        row_idx += 1
        if row_idx > 2:
            row_idx = 1
            col_idx += 1

        if sl:
            sl = False

    fig.update_layout(
        width=600,
        height=300,
        margin=dict(
            t=50,
            l=0,
            r=0,
            b=0,
        ),
    )

    if savedir is None:
        fig.show()
    else:
        savename = f'{unit_id}_metrics.png'
        print(f'saving: {savename}')
        fig.write_image(savename)

