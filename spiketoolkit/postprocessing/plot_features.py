from plotly import graph_objs as go
from plotly.subplots import make_subplots
import spiketoolkit as st
from spiketoolkit.postprocessing import waveform_features_utils as wfl
import numpy as np
import plotly
from scipy.signal import resample
from plotly.colors import sequential as seq

wave_main = seq.Greens[-3]
wave_2nd = seq.gray[2]

peak_col = seq.GnBu[-3]
thr_col = seq.GnBu[-3]

wh_col = seq.Peach[-1]

slope_col = seq.Plasma[-3]



def plot_metrics(sorting, recording, unit_id, invert_template=False, savedir=None,
                 upsampling_factor=None):
    unit_props = sorting.get_unit_property_names(unit_id)
    max_chan = st.postprocessing.get_unit_max_channels(
        sorting=sorting,
        recording=recording,
        unit_ids=unit_id
    )[0]
    template = sorting.get_unit_property(unit_id, 'template')
    n_up = template.shape[1] * upsampling_factor
    fs = sorting.get_sampling_frequency() * upsampling_factor
    template = resample(template, n_up, axis=1)

    time = np.arange(0, template.shape[1]) * (1/fs) * 1000  # in ms
    time = resample(time, n_up)

    row_idx = 1
    col_idx = 1
    subtitles = []
    for i in range(template.shape[0]):
        if i == 2:
            subtitles.append('')
        if i == max_chan:
            subtitles.append(f'main (ch {i})')
        else:
            subtitles.append(f'ch {i}')
    fig = make_subplots(rows=2,
                        cols=3,
                        column_widths=[0.35, 0.35, 0.3],
                        subplot_titles=subtitles,
                        specs=[
                            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'table'}],
                            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'table'}]
                        ]
                        )

    show_legend = True

    if invert_template:
        through_idx, peak_idx = wfl._get_trough_and_peak_idx(
            -template, fs, 1
        )

    else:
        through_idx, peak_idx = wfl._get_trough_and_peak_idx(
            template, fs, 1
        )

    # ------------- loop over template -----------------------
    for template_idx in range(template.shape[0]):
        tr_idx = through_idx[template_idx]
        pk_idx = peak_idx[template_idx]

        if invert_template:
            y = -template[template_idx, :]
            cr_pre, cr_post = wfl._get_halfwidth_crossing(
                -template[template_idx, :], pk_idx
            )
        else:
            y = template[template_idx, :]

        ymax = np.max(y) + 20
        ymin = np.min(y) - 20

        # plot waveforms
        if template_idx == max_chan:
            wave_clr = wave_main
        else:
            wave_clr = wave_2nd
        fig.add_scatter(
            x=time,
            y=y,
            row=row_idx,
            col=col_idx,
            line=dict(
                color=wave_clr
            ),
            showlegend=False,
            hoverinfo='skip',
        )

        # plot peak to valley
        if 'peak_to_valley' in unit_props:
            fig.add_scatter(
                x=[time[tr_idx], time[tr_idx]], y=[y[tr_idx], ymax],
                row=row_idx, col=col_idx,
                mode='lines',
                line=dict(color=peak_col, dash='dash'),
                showlegend=show_legend,
                legendgroup='valley',
                name='valley',
                hoverinfo='text',
                hovertext='valley',
            )
            fig.add_scatter(
                x=[time[pk_idx], time[pk_idx]], y=[y[pk_idx], ymax],
                row=row_idx, col=col_idx,
                mode='lines',
                line=dict(color=peak_col, dash='dash'),
                showlegend=show_legend,
                legendgroup='peak',
                name='peak',
                hoverinfo='text',
                hovertext=f'peak {time[pk_idx]}',
            )

        # plot halfwidth
        if 'halfwidth' in unit_props:
            fig.add_scatter(
                x=[time[cr_pre], time[cr_pre]], y=[ymin, y[cr_pre]],
                row=row_idx, col=col_idx,
                mode='lines',
                line=dict(color=wh_col, dash='dash'),
                showlegend=show_legend,
                legendgroup='hw-pre',
                name='hw-pre',
                hoverinfo='text',
                hovertext='half-width',
            )
            fig.add_scatter(
                x=[time[cr_post], time[cr_post]], y=[ymin, y[cr_post]],
                row=row_idx, col=col_idx,
                mode='lines',
                line=dict(color=wh_col, dash='dash'),
                showlegend=show_legend,
                legendgroup='hw-post',
                name='hw-post',
                hoverinfo='text',
                hovertext='half-width',
            )

        # plot repol slope
        if 'repolarization_slope' in unit_props:
            xdata = time[tr_idx:pk_idx]
            ydata = y[tr_idx:pk_idx]
            slope = wfl._get_slope(xdata, ydata)

            yplot = slope[1] + xdata * slope[0]
            yplot = [yplot[0], yplot[-1]]
            xdata = [xdata[0], xdata[-1]]

            fig.add_scatter(
                x=xdata,
                y=yplot,
                mode='lines+markers',
                line=dict(color=slope_col),
                showlegend=show_legend,
                legendgroup='rep_slope',
                name='rep-slope',
                hoverinfo='text',
                hovertext='rep-slope',
                row=row_idx, col=col_idx,
            )

        # plot recovery slope
        if 'recovery_slope' in unit_props:
            window = 0.7 # TODO dont hardcode
            max_idx = int(pk_idx + ((window/1000)*sorting.get_sampling_frequency()))
            max_idx = np.min([max_idx, template.shape[1]])

            xdata = time[pk_idx:max_idx]
            ydata = y[pk_idx:max_idx]
            slope = wfl._get_slope(xdata, ydata)

            yplot = slope[1] + xdata * slope[0]
            yplot = [yplot[0], yplot[-1]]
            xdata = [xdata[0], xdata[-1]]

            fig.add_scatter(
                x=xdata,
                y=yplot,
                mode='lines+markers',
                line=dict(color=slope_col),
                showlegend=show_legend,
                legendgroup='rec_slope',
                name='rec-slope',
                hoverinfo='text',
                hovertext='rec-slope',
                row=row_idx, col=col_idx,
            )



        # ------------- style axes ----------------------
        if col_idx == 1:
            fig.update_yaxes(
                range=[ymin, ymax],
                row=row_idx,
                col=col_idx,
            )
        elif col_idx == 2:
            fig.update_yaxes(
                range=[ymin, ymax],
                row=row_idx,
                col=col_idx,
            )

        if row_idx == 2:
            fig.update_xaxes(
                title_text='time (ms)',
                row=row_idx,
                col=col_idx,
            )
        else:
            fig.update_xaxes(
                row=row_idx,
                col=col_idx,
            )

        row_idx += 1
        if row_idx > 2:
            row_idx = 1
            col_idx += 1

        if show_legend:
            show_legend = False

    chnames = []
    for i in range(template.shape[0]):
        if i == max_chan:
            chnames.append(f'main ({i})')
        else:
            chnames.append(f'{i}')

    values = [chnames]
    header = ['channel']
    namemap = dict(
        peak_to_valley='ptv',
        halfwidth='halfwidth',
        peak_trough_ratio='pt-ratio',
        recovery_slope='rec-slope',
        repolarization_slope='rep-slope',
    )
    for name in wfl.all_1D_features:
        if name in unit_props:
            header.append(namemap[name])
        chan_vals = []
        for i in range(template.shape[0]):

            val = sorting.get_unit_property(unit_id, name)[i]
            if name == 'peak_to_valley' or name == 'halfwidth':
                val = f'{val * 1000:.3f}'
            elif name == 'peak_trough_ratio':
                val = f'{val:.2f}'
            elif 'slope' in name:
                val = f'{val / 1000 :.2f}'

            chan_vals.append(val)
        values.append(chan_vals)

    fig.add_trace(
        go.Table(
            columnwidth=[2, 2, 2, 2, 2],
            header=dict(values=header),
            cells=dict(values=values),
        ),
        row=1,
        col=3,
    )
    fig.update_layout(
        title=f'unit: {unit_id}',
        width=1200,
        height=500,
        margin=dict(
            t=50,
            l=0,
            r=0,
            b=0,
        ),
        legend=dict(x=0.7, y=0.4)
    )

    if savedir is None:
        # fig.show()
        plotly.offline.plot(fig)
    else:
        savename = f'{savedir}/{unit_id}_metrics.png'
        print(f'saving: {savename}')
        fig.write_image(savename)

