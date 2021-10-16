import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_correlation_matrix(df, annotate_cells=True, file_name=None):
    '''
    Uses matplot lib to plot a correlation matrix between the columns of the dataframe.

            Parameters:
                    df (DataFrame): Data used to plot the correlation matrix
                    annotate_cells (bool): When true annotate the cell with the correlation
                    file_name (String): Path and filename to save the plot

            Returns:
                    correlation (DataFrame): Correlation DataFrame
    '''
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, ax_lab) = plt.subplots(2, gridspec_kw=grid_kws,
                                   figsize=(10, 12))
    corr = df.corr()
    df_columns = np.asarray(list(corr.columns))
    ticks = list(np.linspace(0, len(df_columns)-1, 24, dtype=np.int))
    #ticks = [0, 153, 306, 459, 612, 765, 917]
    sns.heatmap(corr,
                mask=np.zeros_like(corr, dtype=np.bool),
                # sns.color_palette("coolwarm", as_cmap=True),#sns.color_palette("icefire", as_cmap=True),#sns.color_palette("vlag", as_cmap=True),#
                cmap=sns.diverging_palette(220, 10, sep=48, as_cmap=True),
                square=True,
                ax=ax,
                annot=annotate_cells,
                cbar_kws={"orientation": "horizontal"},
                cbar_ax=ax_lab,
                vmin=-1,
                vmax=1)
    if len(df_columns) > 100:
        tick_labels = list(df_columns[ticks])
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=15)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=15)
    ax_lab.set_xticklabels(['-1.00', '-0.75', '-0.50', '-0.25',
                           '0.00', '0.25', '0.50', '0.75', '1.00'], fontsize=13)

    if file_name is not None:
        plt.savefig(file_name, dpi=450)

    return corr
