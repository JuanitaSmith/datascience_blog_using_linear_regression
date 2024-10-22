import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, \
    PredictionErrorDisplay, median_absolute_error, \
    mean_absolute_error
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from statsmodels.stats.diagnostic import normal_ad

from project1_data_science_blog.src.utils import inverse_log_transform, \
    calculate_residuals


def hex_to_rgb(hex_value):
    """ Convert color hex code to rgb for sns mapping """

    h = hex_value.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def set_plot_defaults():
    """Set defaults formatting for consistency across all plots """

    # set plot style
    sns.set_style("whitegrid")

    # my custom color palette - I call is ocean spray
    hex_colors = [
        '7C9E9E',  # base
        'E2E9E9',  # light gray
        '578686',  # group 1
        '667595',  # group 2
        'FFEDC8',  # highlight
        '366F6F',  # highlight intense
        'DFC591',  # base_complementary
    ]

    rgb_colors = list(map(hex_to_rgb, hex_colors))
    # sns.palplot(rgb_colors)

    base_color = rgb_colors[0]
    base_grey = rgb_colors[1]
    base_color_group1 = rgb_colors[2]
    base_color_group2 = rgb_colors[3]
    base_highlight = rgb_colors[4]
    base_highlight_intense = rgb_colors[5]
    base_complimentary = rgb_colors[6]

    # up and down arrows for growth indicators
    symbols = [u'\u25BC', u'\u25B2']

    small_size = 8
    medium_size = 10
    bigger_size = 12

    plt.rc('font', size=small_size, weight='ultralight', family='sans-serif')  # controls default text sizes and font
    plt.rc('axes', titlesize=bigger_size, titlecolor='black', titleweight='bold', labelsize=medium_size,
           labelcolor='black', labelweight='ultralight')  # axes settings
    # fontsize of the ytick labels
    plt.rc('xtick', labelsize=small_size)
    # fontsize of the xtick labels
    plt.rc('ytick', labelsize=small_size)
    # legend fontsize
    plt.rc('legend', fontsize=small_size)
    # fontsize of the figure title
    plt.rc('figure', titlesize=bigger_size, titleweight="bold", figsize=[8, 4])

    return (base_color, base_highlight_intense, base_highlight, base_complimentary, base_grey,
            base_color_group1, base_color_group2, symbols)


# set default plot formatting
(BASE_COLOR, BASE_HIGHLIGHT_INTENSE, BASE_HIGHLIGHT, BASE_COMPLIMENTARY, BASE_GREY, BASE_COLOR_ARR, BASE_COLOR_DEP,
 SYMBOLS) = set_plot_defaults()


def improve_yticks(maxvalue,
                   bins=10):
    """Dynamically set the binsize to control space between annotation and bars"""

    binsize = maxvalue / bins
    if maxvalue > 4000000:
        ind = 'mil'
        div = 1000000
        binsize = 500000
        yticks = np.arange(0, maxvalue + binsize, binsize)
        ylabels = ['{:1.1f}'.format(tick / div) + ind for tick in yticks]
    elif maxvalue > 1000000:
        ind = 'mil'
        div = 1000000
        binsize = 200000
        yticks = np.arange(0, maxvalue + binsize, binsize)
        ylabels = ['{:1.1f}'.format(tick / div) + ind for tick in yticks]
    elif maxvalue > 10000:
        ind = 'k'
        div = 1000
        yticks = np.arange(0, maxvalue + binsize, binsize)
        ylabels = ['{:1.0f}'.format(tick / div) + ind for tick in yticks]
    else:
        ind = ''
        div = 1
        yticks = np.arange(0, maxvalue + binsize, binsize)
        ylabels = ['{:1.0f}'.format(tick / div) + ind for tick in yticks]

    return yticks, ylabels, binsize


def plot_period_side_by_side(df, col, annotate=True, title='month', rotate=False, sharey=False, base_grey=BASE_GREY,
                             base_color=BASE_COLOR, order=None):
    """ 2 bar plots side by side for a certain category: plot 1 showing only delays, plot 2 all flights"""

    flight_total = df.groupby(col)['total_flights'].mean().sum()
    delay_total = df.groupby(col)['delayed'].mean().sum()

    # Customize annotation
    if rotate:
        plt.figure(figsize=(14, 4))
        weight = 'ultralight'
        rotation = 90
        color = 'black'
        xytext = (0, 8)
        size = 6
    else:
        plt.figure(figsize=(14, 4))
        weight = 'ultralight'
        rotation = None
        color = 'black'
        xytext = (0, 3)
        size = 8

    # PLOT 2: visualize distribution of ALL flights per selected period
    ax1 = plt.subplot(1, 2, 2)
    sns.barplot(data=df, x=col, y='total_flights', color=base_grey, label='All flights', errorbar=None, errwidth=1,
                order=order)

    # improve yticks
    maxvalue = df.groupby(col)['total_flights'].mean().max()
    yticks, ylabels, binsize = improve_yticks(maxvalue)
    plt.yticks(yticks, ylabels)

    # annotate bars with % of total flights
    if annotate:
        for p in ax1.patches:
            ax1.annotate("{:.1%}".format(p.get_height() / flight_total),
                         (p.get_x() + p.get_width() / 2.,
                          p.get_height()),
                         ha='center', va='center',
                         xytext=xytext,
                         textcoords='offset points',
                         size=7,
                         weight=weight,
                         color='black',
                         rotation=rotation)

    plt.title('Average total flights {}'.format(title))
    plt.xlabel(title)
    plt.ylabel('')

    # PLOT 1: visualize distribution of DELAYED flights per selected period

    # control stacking
    if sharey:
        ax2 = plt.subplot(1, 2, 1, sharey=ax1, sharex=ax1)
    else:
        ax2 = plt.subplot(1, 2, 1, sharex=ax1)

    sns.barplot(data=df, x=col, y='delayed', color=base_color, label='Delayed flights', errorbar=None, errwidth=1,
                order=order)

    if not sharey:
        # improve yticks
        maxvalue = df.groupby(col)['delayed'].mean().max()
        yticks, ylabels, binsize = improve_yticks(maxvalue)
        plt.yticks(yticks, ylabels)

    # annotate bars with % of total flights
    if annotate:
        for p in ax2.patches:
            ax2.annotate("{:.1%}".format(p.get_height() / delay_total),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=xytext,
                         textcoords='offset points',
                         size=7,
                         weight=weight,
                         color=color,
                         rotation=rotation)

    plt.title('Average delayed flights per {}'.format(title))
    plt.xlabel(title)
    plt.ylabel('Number of flights')

    plt.suptitle('Distribution of flights per {}'.format(title))
    plt.tight_layout()
    plt.show()


def plot_period_stacked(df, col, figsize=None, base_grey=BASE_GREY, base_color=BASE_COLOR, annotate=True, title='month',
                        rotate=False):
    """Plot proportion total flights vs delayed flights per category

       Print % of total flights at the top of each bar, example

       % of all flights in January / all flights
       % of delayed flights in January / all flights
    """

    flight_total = df.groupby(col)['total_flights'].mean().sum()

    if figsize:
        plt.figure(figsize=figsize)

        # Customize annotation
    if rotate:
        weight = 'ultralight'
        rotation = 90
        color = 'black'
        xytext = (0, 8)
        size = 6
    else:
        weight = 'ultralight'
        rotation = None
        color = 'black'
        xytext = (0, 3)
        size = 8

    ax1 = sns.barplot(data=df, x=col, y='total_flights', color=base_grey, label='All flights', errorbar=None,
                      errwidth=1)
    ax2 = sns.barplot(data=df, x=col, y='delayed', color=base_color, label='Delayed flights', errorbar=None, errwidth=1,
                      width=0.6, edgecolor=base_color)

    #   improve yticks
    maxvalue = df.groupby(col)['total_flights'].mean().max()
    yticks, ylabels, binsize = improve_yticks(maxvalue)
    plt.yticks(yticks, ylabels)

    #  annotate bars with % of total flights
    if annotate:
        for p in ax2.patches:
            ax2.annotate("{:.1%}".format(p.get_height() / flight_total),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=xytext,
                         textcoords='offset points',
                         size=7,
                         weight=weight,
                         color=color,
                         rotation=rotation)

    plt.title('Average total flights for {}'.format(title))
    plt.xlabel(title)
    plt.ylabel('Number of flights')

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def flights_by_cat(df, col, title='Origin airports with most delayed flights', topn=20, base_color=BASE_COLOR,
                   base_highlight=BASE_HIGHLIGHT_INTENSE,
                   lookup=False, df_lookup=None, df_lookup_field=None):
    """ Count plot - first 3 highest bars are highlighted """

    # calculate top category and order of bar charts
    top = df[col].value_counts(ascending=False)
    top_order = top.index[:topn]

    # replace index key with key and description
    if lookup:
        top.index = top.index + ':' + df_lookup.loc[top.index][df_lookup_field]

    clrs = [base_color if i >= 3 else base_highlight for i in np.arange(0, topn + 1, 1)]

    ax = sns.countplot(data=df, y=col, order=top_order, palette=clrs, orient='h', width=0.6)

    plt.title('{} '.format(title), weight='bold')
    plt.xlabel(title)
    plt.ylabel(col)

    # improve xticks and labels
    ticks, xlabels, binsize = improve_yticks(top[0])
    plt.xticks(ticks, xlabels)

    #   calculate and print % on the top of each bar
    ticks = ax.get_yticks()
    new_labels = []
    locs, labels = plt.yticks()
    for loc, label in zip(locs, labels):
        count = top[loc]
        perc = '{:0.1f}%'.format((count / top.sum()) * 100)
        # print only the first characters of xlabel descriptions
        text = top.index[loc][:40]
        new_labels.append(text)
        plt.text(count + (0.2 * binsize), loc, perc, ha='center', va='center', color='black', fontsize=6,
                 weight='ultralight')
    plt.yticks(ticks, new_labels, fontsize=6, weight='ultralight')

    plt.tight_layout()
    plt.show()


def cat_heatmap(df,
                reason,
                center=0):
    """ heat map with standard formatting """

    g = sns.heatmap(df, center=center, cmap='Spectral', linewidths=0.003, linecolor='lightgrey', square=True,
                    mask=df < 1, annot=True, fmt=".0f",
                    cbar_kws={"orientation": "vertical", "pad": 0.03, "shrink": 0.5})

    # put xlabels on top
    plt.title('{}'.format(reason.upper()))
    plt.xlabel('Origin Airport')
    plt.ylabel('Carrier')


def compare_features(df,
                     cols_of_interest,
                     conda,
                     condb,
                     cata_description,
                     catb_description,
                     title_extension,
                     feature_focus):
    if feature_focus:
        cols_of_interest.remove(feature_focus)

    groupa = df.query(conda)
    groupb = df.query(condb)

    groupa = groupa[cols_of_interest].describe().T['mean'].to_frame()
    groupa['category'] = cata_description

    groupb = groupb[cols_of_interest].describe().T['mean'].to_frame()
    groupb['category'] = catb_description

    df = pd.concat([groupa, groupb], axis=0)
    df.sort_index(inplace=True)

    plt.figure(figsize=(12, 6))

    cat = list(df.category.unique())

    ax = sns.barplot(data=df, y=df['mean'], x=df.index, hue='category', errorbar='sd',
                     palette={cata_description: BASE_GREY, catb_description: BASE_COLOR},
                     hue_order=[cata_description, catb_description])

    # since WE are using hue, there are multiple containers
    for c in ax.containers:
        # set the bar label based on the y-axis
        ax.bar_label(c, fmt='%.1f', padding=1, fontsize=8)

    # pad the spacing between the number and the edge of the figure
    ax.margins(y=0.1)

    ax.legend(facecolor='w', bbox_to_anchor=(1, 1), loc='upper left')
    plt.xticks(rotation=90)
    plt.xlabel('features of interest')

    plt.title('Comparison of features: {}'.format(title_extension))
    plt.show()


def annotate_grouped_barplot(df_a,
                             df_b,
                             hue='is_business',
                             x='room_type',
                             y='percentage',
                             topn=10,
                             figsize=(6, 10),
                             title='Business and Individual Hosts: Price Comparison',
                             legend_title='Is Business?',
                             show_gridlines=True):
    """
    Categorical seaborn bar plot - plot proportions by category
    comparing business vs individual listings
    """

    plt.figure(figsize=figsize)

    # Prepare the data

    # calculate mean price
    meana = df_a.groupby(x)['price_mean'].mean().astype(int).to_frame()
    meanb = df_b.groupby(x)['price_mean'].mean().astype(int).to_frame()

    # calculate proportion of x for hue boolean field independently
    topa = df_a[x].value_counts(ascending=False, normalize=True).to_frame()[:topn]
    index = topa.index

    topa[hue] = 'True'
    topa = topa.merge(meana, how='inner', left_index=True, right_index=True)

    topb = df_b[x].value_counts(ascending=False, normalize=True).to_frame().loc[index]

    topb[hue] = 'False'
    topb = topb.merge(meanb, how='inner', left_index=True, right_index=True)

    top = pd.concat([topa, topb]).reset_index()
    top.columns = [x, 'percentage', hue, 'price_mean']

    cat = list(top[hue].unique())

    ax = sns.barplot(data=top, y=top[x], x=top[y], hue=hue,
                     palette={cat[0]: BASE_COLOR, cat[1]: BASE_GREY}, width=0.4)

    # iterate through each hue group of containers
    for c, col in zip(ax.containers, cat):
        # use the column and bar height to get the correct value for price_mean
        labels2 = [f'{x * 100:,.1f}%' for x in c.datavalues]

        labels1 = [f"(Avg: £{top.loc[(top[hue].eq(col) & top.percentage.eq(h)), 'price_mean'].iloc[0]} pn)"
                   if (h := v.get_width()) > 0 else '' for v in c]

        # add the name annotation to the top of the bar
        ax.bar_label(c, labels=labels2, padding=2, label_type='edge', color='black', weight='ultralight', fontsize=8)

        # add the name annotation to the top of the bar
        ax.bar_label(c, labels=labels1, padding=30, label_type='edge', color='black', fontweight='ultralight',
                     fontsize=8)

    # leave space at end of plot for extra text
    ax.margins(x=0.4)
    binsize = 0.1
    ax.grid(show_gridlines)
    xticks = np.arange(0, 1 + binsize, binsize)
    xlabels = ['{:1.0f}%'.format(tick * 100) for tick in xticks]
    plt.xticks(xticks, xlabels)

    plt.legend(facecolor='w', bbox_to_anchor=(1, 1), loc='upper left', title=legend_title,
               fontsize='small', title_fontsize='large')
    plt.title(title)
    plt.show()


def plot_categories(df,
                    title=None,
                    topn=20,
                    figsize=(18, 10),
                    base_color=BASE_COLOR,
                    base_grey=BASE_GREY):
    """
    Categorical seaborn bar plot - plt differences in features
    of business vs individual listings
    """

    business_total = df['business'].sum()
    individual_total = df['individual'].sum()
    grand_total = business_total + individual_total

    df = df[:topn]
    plt.figure(figsize=figsize)

    ax1 = sns.barplot(data=df, y=df.index, x='individual', color=base_grey,
                      label='Individual', errorbar=None)
    ax2 = sns.barplot(data=df, y=df.index, x='business', color=base_color,
                      label='Business', errorbar=None, width=0.3,
                      edgecolor=BASE_COLOR)

    xticks = np.arange(0, 1.1, 0.1)
    xlabels = [f'{tick:.0%}'.format(tick) for tick in xticks]
    plt.xticks(xticks, xlabels)

    labels = []
    for bars in ax1.containers:
        label = [x for x in bars.datavalues]
        labels.append(label)

    label_colors = ['red', 'black']
    for i, bars in enumerate(ax1.containers):
        labels = [f"{x:.0%}" for x in bars.datavalues]
        #         label_colors = ['black' if i ==0 else 'white' for label in labels]
        ax1.bar_label(bars, labels=labels, weight='bold', color=label_colors[i], padding=2, size=9)
    ax1.margins(x=0.1)

    plt.title('{}'.format(title))
    plt.xlabel('Proportion of business vs individual rentals')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def hist_by_cat(df,
                col,
                title='Distribution for',
                topn=20,
                base_color=BASE_COLOR,
                base_highlight=BASE_COMPLIMENTARY
                ):
    """ Categorical seaborn count plot - first 3 highest bars are highlighted """

    # set figsize
    if df[col].nunique() <= 2:
        figsize = (6, 2)
    elif df[col].nunique() <= 6:
        figsize = (6, 3)
    else:
        figsize = (12, 8)

        # calculate top category and order of bar charts
    top = df[col].value_counts(ascending=False)
    top_order = top.index[:topn]

    # update topn to max number unique values. If less than 3 unique values, disable highlighting
    topn = len(top_order) - 1

    if topn > 3:
        clrs = [base_color if i >= 3 else base_highlight for i in np.arange(0, topn + 1, 1)]
    else:
        clrs = [base_color if i >= 3 else base_color for i in np.arange(0, topn + 1, 1)]

    plt.figure(figsize=figsize)
    ax = sns.countplot(data=df,
                       y=col,
                       hue=col,
                       hue_order=top_order,
                       order=top_order,
                       palette=clrs,
                       orient='h',
                       width=0.5,
                       legend=False)

    plt.title('{} {}'.format(title, col), weight='bold')
    plt.xlabel('Frequency')
    plt.ylabel(col)

    # improve xticks and labels
    ticks, xlabels, binsize = improve_yticks(top.iloc[0])
    plt.xticks(ticks, xlabels, fontsize=8)

    #   calculate and print % on the top of each bar
    ticks = ax.get_yticks()
    new_labels = []
    locs, labels = plt.yticks()
    for loc, label in zip(locs, labels):
        count = top.iloc[loc]
        perc = '{:0.1f}%'.format((count / top.sum()) * 100)
        text = top.index[loc]
        new_labels.append(text)
        plt.text(count + (0.3 * binsize), loc, perc, ha='center', va='center',
                 color='black', fontsize=7, weight='ultralight')
    plt.yticks(ticks, new_labels, fontsize=8, weight='ultralight')

    plt.tight_layout()
    plt.show()


def annotate_bars(ax=None, fmt='.2f', **kwargs):
    """
    Add '%' to custom annotation to bar plots.

    Print the height of the bar as annotation with '%' as suffix
    Example 86.8 becomes 86.8%

    Args:
        ax (axis object) - plot to annotate
        fmt (string) - formatting of numerical numbers
    """

    ax = plt.gca() if ax is None else ax
    for p in ax.patches:
        ax.annotate('{{:{:s}}}%'.format(fmt).format(p.get_height()),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    va='center',
                    **kwargs)


def custom_mean_line(y,
                     color='r',
                     linewidth=0.8,
                     ha='left',
                     fontsize=8,
                     weight='light',
                     **kwargs):
    """ Print horizontal mean line on plots using y.mean(). """

    ym = y.mean()
    plt.axhline(ym,
                color=color,
                linestyle="dashed",
                linewidth=linewidth)

    plt.annotate(f"mean: £{y.mean():.0f}", xy=(1, ym),
                 xycoords=plt.gca().get_yaxis_transform(),
                 ha=ha,
                 fontsize=fontsize,
                 color=color,
                 weight=weight)


def plot_star_ratings(data,
                      col,
                      col_order,
                      x,
                      y,
                      x_lim,
                      y_lim,
                      suptitle=None,
                      suptitle_offset=1.05,
                      row=None,
                      row_order=None,
                      col_wrap=4,
                      height=4,
                      aspect=1):
    """
    Compare feature values with seaborn FacetGrid and regplot

    Plot looks like a point plot in the end, connecting means
    Parameters are passed to control min, max and intervals of y and x-axis

    Args:
        data (pandas dataFrame) - contain values col, x, y
        col (pandas column name) - column on FacetGrid to break down
        col_order (list) - sequence of col plots in same row
        x (pandas column name) - column on x-axis of plot
        y (pandas column name) - column on y-axis of plot
        y_lim (list with 3 entries) - controls y-axis y_lim
            e.g. [0, 300, 25] y-axis with min value 0, max value 275, interval 25
        x_lim (list with 3 entries) - controls x-axis x_lim
            e.g. [3, 5.5, 0.5] x-axis with min value 3, max value 5, interval 0.5
        suptitle (str) - overall title for all plots
        suptitle_offset (int) - space between suptitle and rest of plots
        row (pandas column name) - optional column on FacetGrid row
        row_order (list) - sequence of row plots
        col_wrap (int) - number of columns to plot in a row
        height (int) - height of plots
        aspect (int) - aspect of the plots (controlling the width)
    """
    xticks = np.arange(0, x_lim[1], x_lim[2])
    g = sns.FacetGrid(data=data,
                      col=col,
                      row=row,
                      margin_titles=True,
                      despine=False,
                      col_order=col_order,
                      row_order=row_order,
                      col_wrap=col_wrap,
                      height=height,
                      aspect=aspect)

    g.map(sns.regplot,
          x,
          y,
          x_jitter=True,
          x_bins=xticks,
          scatter_kws={'alpha': 0.3,
                       's': 20,
                       'edgecolor': 'white',
                       'linewidths': 0.5,
                       'color': BASE_HIGHLIGHT_INTENSE},
          line_kws={'color': 'orange'},
          truncate=False)

    # add mean horizontal line
    g = g.map(custom_mean_line, 'price_mean')

    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    yticks = np.arange(y_lim[0], y_lim[1], y_lim[2])
    ylabels = ['£{:1.0f}'.format(tick) for tick in yticks]
    plt.yticks(yticks, ylabels)
    plt.ylim(y_lim[0], y_lim[1])

    xticks = np.arange(x_lim[0], x_lim[1], x_lim[2])
    plt.xticks(xticks, xticks)
    g.set_axis_labels('Star rating',
                      'Average price (per night in pounds)')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.suptitle(suptitle, y=suptitle_offset)

    plt.tight_layout()


def goodness_of_fit(y,
                    y_hat,
                    show=False):
    """
    Calculates and return model performance metrics.

    Metrics: r2_score, mean_squared_error, mean_absolute_error and
             Mean Absolute Error

    Args:
        y (array-like of shape (n_samples,) - Actual values
        y_hat (array-like of shape (n_samples,) - Predicted values
        show (boolean) - If True print metrics to screen
    Returns:
        string for each calculated performance metric
    """

    # Calculate the performance scores
    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    maeEd = median_absolute_error(y, y_hat)

    if show:
        print("R-squared: {:.2f}".format(r2))
        print("Mean Squared Error (MSE): {:.2f}".format(mse))
        print("Median Absolute Error: £{:.2f} per night".format(maeEd))
        print("Mean Absolute Error: £{:.2f} per night".format(mae))

    return {
        "R-squared": f"{r2_score(y, y_hat):.3f}",
        "Median Absolute Error": f"£{median_absolute_error(y, y_hat):.2f} pn",
        "Mean Absolute Error": f"£{mean_absolute_error(y, y_hat):.2f} pn",
        "Mean Squared Error": f"{mean_squared_error(y, y_hat):.2f}",
    }


def plot_model_performance_scatter(y,
                                   y_pred1,
                                   y_pred2,
                                   title1,
                                   title2,
                                   suptitle,
                                   kind='actual_vs_predicted',
                                   lower_lim=-200,
                                   upper_lim=1500,
                                   interval=100,
                                   figsize=(12, 6)):
    """
    Compare the performance of 2 models plotting its
    predictions vs actual values

    Scatter plot with actual or residuals on y-axis and predicted on x-axis

    Args:
        y_pred1 (array-like) - target predictions from model 1
        y_pred2 (array-like) - target predictions from mdoel 2
        title1 (str) - title for plot 1
        title2 (str) - title for plot 2
        suptitle (str) - main title
        kind (str) - 2 options:
                    'actual_vs_predicted' or 'residual_vs_predicted'
        lower_lim (int) - start x-axis
        upper_lim (int) - ened x-axis
        figsize (tuple) - size of plot
    """

    bins = np.arange(lower_lim, upper_lim + interval, interval)
    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=figsize)

    plt.subplot(1, 2, 1)

    PredictionErrorDisplay.from_predictions(
        y,
        y_pred1,
        kind=kind,
        ax=ax0,
        scatter_kwargs={"alpha": 0.3,
                        'marker': 'o',
                        'color': BASE_HIGHLIGHT_INTENSE,
                        's': 50,
                        'edgecolor': 'black',
                        'linewidths': 0.3}
    )

    if kind == 'actual_vs_predicted':
        ax0.set_xticks(bins, bins, rotation=90)
        ax0.set_yticks(bins, bins)
        ax0.set_xlim(lower_lim, upper_lim)
        ax0.set_ylim(lower_lim, upper_lim)
    else:
        ax0.set_xticks(bins, bins, rotation=90)
        ax0.set_xlim(lower_lim, upper_lim)

    plt.subplot(1, 2, 2)
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred2,
        kind=kind,
        ax=ax1,
        scatter_kwargs={"alpha": 0.3,
                        'marker': 'o',
                        'color': BASE_HIGHLIGHT_INTENSE,
                        's': 50,
                        'edgecolor': 'black',
                        'linewidths': 0.3},
    )

    if kind == 'actual_vs_predicted':
        ax1.set_xticks(bins, bins, rotation=90)
        ax1.set_yticks(bins, bins)
        ax1.set_xlim(lower_lim, upper_lim)
        ax1.set_ylim(lower_lim, upper_lim)
        loc = 'upper left'
    else:
        ax1.set_xticks(bins, bins, rotation=90)
        ax1.set_xlim(lower_lim, upper_lim)
        loc = 'upper right'

    ax1.yaxis.set_tick_params(labelleft=True)

    # Add the score in the legend of each axis
    for ax, y_pred in zip([ax0, ax1], [y_pred1, y_pred2]):
        for name, score in goodness_of_fit(y, y_pred).items():
            ax.plot([], [], " ", label=f"{name}={score}")
        ax.legend(loc=loc)

    ax0.set_title(title1)
    ax1.set_title(title2)
    fig.suptitle(suptitle)
    plt.show()


def plot_model_performance_hist(y,
                                y_pred1,
                                y_pred2,
                                title1=None,
                                title2=None,
                                suptitle=None,
                                figsize=(12, 12),
                                xlim_max=1200,
                                xlim_interval=25):
    """
    Compare the performance of 2 models plotting its
    predictions vs actual values using a histogram

    Histogram with predicted values in color BASE_COLOR
    and predicted values in color BASE BASE_COMPLIMENTARY

    Args:
        y_pred1 (array-like) - target predictions from model 1
        y_pred2 (array-like) - target predictions from mdoel 2
        title1 (str) - title for plot 1
        title2 (str) - title for plot 2
        suptitle (str) - main title
        figsize (tuple) - size of plot
        xlim_max (int) - upper limit of x-axis
        xlim_interval (int) - interval of x-axis
    """

    # Actual vs predicted price comparison
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=figsize,
                                   sharex=True,
                                   sharey=True)

    # create bins
    xbins = np.arange(0, xlim_max + xlim_interval, xlim_interval)

    # plot for prediction from model 1
    ax1 = sns.histplot(y,
                       bins=xbins,
                       stat='percent',
                       kde=False,
                       color=BASE_COLOR,
                       label='actual',
                       ax=ax1)

    sns.histplot(y_pred1,
                 bins=xbins,
                 stat='percent',
                 kde=False,
                 color=BASE_COMPLIMENTARY,
                 alpha=0.5,
                 label='predicted',
                 ax=ax1)

    ax1.set_title(title1)
    ax1.legend()

    # plot for prediction from model 2
    ax2 = sns.histplot(y,
                       bins=xbins,
                       stat='percent',
                       kde=False,
                       color=BASE_COLOR,
                       label='actual',
                       ax=ax2)

    sns.histplot(y_pred2,
                 bins=xbins,
                 stat='percent',
                 kde=False,
                 color=BASE_COMPLIMENTARY,
                 alpha=0.5,
                 label='predicted',
                 ax=ax2)

    ax2.set_title(title2)
    ax2.legend()

    plt.xticks(xbins, xbins, rotation=90)
    ax1.xaxis.set_tick_params(labelbottom=True, rotation=90)
    plt.xlim(0, xlim_max)
    plt.suptitle(suptitle)

    plt.show()


def distribution_of_errors(model1,
                           model2,
                           X,
                           y,
                           title1=None,
                           title2=None,
                           suptitle=None,
                           figsize=(12, 6),
                           kind='kde',
                           ):
    """
    Normality: Assumes that the error terms are normally distributed.

    Calculates residuals and plot it's distribution on different kind of plots
    configured with kind parameter. If kind = 'kde', plot the distribution on
    a histogram and kde plots. If kind = 'hist', plot the distribution on just
    a histogram for better clarity.

    Args:
        model1 - scikit-learn fitted model
        model2 -  scikit-learn fitted model
        X (array-like or sparse matrix) - independent features
        y (array-like) - dependent feature (true / actual values)
        title1 (string) - plot 1 title
        title2 (string) - plot 2 title
        suptitle (str) - main title
        figsize (tuple) - size of plot
        kind (str) - kind of plot with possible values 'kde' or 'hist'
    Returns:
         ax - plot axis
    """
    fig = plt.figure(figsize=figsize)

    if kind == 'kde':
        ax = plt.subplot(1, 2, 1)
        normal_errors_assumption(model=model1,
                                 features=X,
                                 label=y,
                                 ax=ax,
                                 title=title1)

        ax2 = plt.subplot(1, 2, 2, sharey=ax)
        normal_errors_assumption(model=model2,
                                 features=X,
                                 label=y,
                                 ax=ax2,
                                 title=title2)

    elif kind == 'hist':
        ax1 = plt.subplot(1, 2, 1)
        x = calculate_residuals(model1,
                                X,
                                y).values.flatten()

        plt.hist(x, bins=100, color=BASE_HIGHLIGHT_INTENSE)
        plt.xlim(-1000, 1000)
        ax1.set_title(title1)

        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Residuals')

        ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        x = calculate_residuals(model2,
                                X,
                                y).values.flatten()

        sns.histplot(x, bins=100, color=BASE_HIGHLIGHT_INTENSE)
        ax2.set_title(title2)

        ax2.set_ylabel('Frequency')
        ax2.set_xlabel('Residuals')

    plt.suptitle(suptitle, y=1.05)
    plt.tight_layout()


def normal_errors_assumption(model,
                             features,
                             label,
                             ax,
                             p_value_thresh=0.05,
                             title=None):
    """
    Normality: Assumes that the error terms are normally distributed.

    Calculates residuals and plot the distribution on hist and kde plots.

    The error term (residuals) should have a mean of zero.
    If they are not, nonlinear transformations of variables may solve this.

    This assumption being violated primarily causes issues
    with the confidence intervals.

    p-value from the test - below 0.05 generally means non-normal.

    Args:
        features (array-like or sparse matrix) - independent features
        label (array-like) - dependent feature (true / actual values)
        model - scikit-learn fitted model
        ax - axis plot should be linked to
        p_value_thresh (numeric variable) - p_value threshold default 5%
        title (string) - plot title
    Returns:
         ax - plot axis
    """

    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)

    # Performing the test on the residuals
    p_value = normal_ad(df_results['Residuals'])[1]

    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        text = 'Residuals are not normally distributed'
    else:
        text = 'Residuals are normally distributed'

    # Plotting the residuals distribution
    plt.title(title)

    ax = sns.histplot(df_results['Residuals'],
                      bins=100,
                      stat='density',
                      kde=True,
                      color=BASE_COLOR)

    ax = sns.kdeplot(df_results['Residuals'],
                     color=BASE_HIGHLIGHT_INTENSE,
                     ax=ax,
                     linewidth=2)

    # print value as legend on the plot
    ax.plot([], [], " ", label=f"pvalue={p_value}: {text}")
    ax.legend(loc="upper left")


def residual_variance(model1,
                      model2,
                      X,
                      y,
                      title1=None,
                      title2=None,
                      suptitle=None,
                      figsize=(14, 4)):
    """
    Compare Homoscedasticity of 2 models.

    For linear regression we assumes that the errors/residuals
    have a constant variance, with a mean of 0.

    Visualizes residuals on a scatter plot for 2 different fitted
    models so we can compare visually which one performs better.

    Args:
        model1 - scikit-learn fitted model
        model2 -  scikit-learn fitted model
        X (array-like or sparse matrix) - independent features
        y (array-like) - dependent feature (true / actual values)
        title1 (string) - plot 1 title
        title2 (string) - plot 2 title
        suptitle (str) - main title
        figsize (tuple) - size of plot
    Returns:
         ax - plot axis
    """
    plt.figure(figsize=figsize)

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(title1)
    ax1.set_xlabel('Index')
    homoscedasticity_assumption(model1, X, y, ax=ax1)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(title2)
    ax2.set_xlabel('Index')
    homoscedasticity_assumption(model2, X, y, ax=ax2)

    plt.suptitle(suptitle, y=1.05)
    plt.show()


def homoscedasticity_assumption(model, X, y, ax):
    """
    Homoscedasticity of Error Terms

    For linear regression we assumes that the errors/residuals
    have a constant variance, with a mean of 0.

    Visualizes residuals on a scatter plot.

    Args:
        model - scikit-learn fitted model
        X (array-like or sparse matrix) - independent features
        y (array-like) - dependent feature (true / actual values)
        ax (matplotlib ax object)- axis plot should be linked to
    Returns:
         ax - plot axis
    """

    # Calculating residuals for the plot
    df_results = calculate_residuals(model, X, y)
    df_results.reset_index(inplace=True)

    sns.scatterplot(x=df_results.index,
                    y=df_results.Residuals,
                    color=BASE_COLOR,
                    marker='o',
                    edgecolor='black',
                    linewidth=0.1,
                    alpha=0.2)

    plt.plot(np.repeat(0, df_results.index.max()),
             color=BASE_COMPLIMENTARY,
             linestyle='--')

    # Removing the right spine
    ax.spines['right'].set_visible(False)

    # Removing the top spine
    ax.spines['top'].set_visible(False)


def plot_coef(model1,
              model2,
              title1,
              title2,
              suptitle,
              figsize=(12, 20),
              logscale1=False,
              logscale2=False):
    """
    Plot coefficients of 2 fitted models.

    Visually compare coefficients of 2 model models plotted on a bar chart.
    If either model was fitted on logscale, set logscale to True, coefficients
    will be converted back to original values for interpretability.

    Args:
        model1 - scikit-learn fitted model
        model2 -  scikit-learn fitted model
        title1 (string) - plot 1 title
        title2 (string) - plot 2 title
        suptitle (str) - main title
        figsize (tuple) - size of plot
        logscale1 (bool)- indicate if model1 fitted on log scale
        logscale2 (bool)- indicate if model2 is fitted on log scale
    Returns:
         ax - plot axis
    """
    # plot for model 1
    ax1 = plt.subplot(1, 2, 1)
    coef_df, intercept = coef_weights(model1, logscale1)
    coef_df['coefs'].plot.barh(figsize=figsize, color=BASE_COLOR, ax=ax1)
    plt.title(title1 + '\n Intercept: £{:.0f}'.format(intercept))
    start = 0
    plt.xlabel("Raw coefficient values")
    if logscale1:
        start = 1
    plt.axvline(x=start, color=".5")
    plt.ylabel('Independent features')
    plt.subplots_adjust(left=0.4)

    for c in ax1.containers:
        labels = [f"{x:,.3}" for x in c.datavalues]
        ax1.bar_label(c,
                      label=labels,
                      label_type='edge',
                      fontsize=6,
                      padding=3)
    ax1.margins(x=0.2)

    # plot for model 2
    ax2 = plt.subplot(1, 2, 2)
    coef_df, intercept = coef_weights(model2, logscale2)
    coef_df['coefs'].plot.barh(figsize=figsize, color=BASE_COLOR)
    plt.title(title2 + '\n Intercept: £{:.0f}'.format(intercept))
    plt.xlabel("Raw coefficient values")
    if logscale2:
        start = 1
    plt.axvline(x=start, color=".5")
    plt.ylabel('Independent features')
    plt.subplots_adjust(left=0.4)

    for c in ax2.containers:
        labels = [f"{x:,.3}" for x in c.datavalues]
        ax2.bar_label(c,
                      label=labels,
                      label_type='edge',
                      fontsize=6,
                      padding=3)
    ax2.margins(x=0.2)

    plt.suptitle(suptitle, y=1.01)
    plt.tight_layout()

    return coef_df


def coef_weights(model, logscale=False):
    """
    Retrieve coefficients of a fitted model.

    Provides a dataframe that can be used to understand the most
    influential coefficients of a linear model by providing the coefficient
    estimates along with the name of the variable attached to the coefficient.

    If model was fitted on logscale, set logscale to True. Coefficients
    will be converted back to original values for interpretability.

    Args:
        model - scikit-learn fitted model
        logscale (bool)- indicate if model fitted on log scale
    Returns:
         coefs_df (pandas dataframe) - dataframe with coeffients
         intercept (float) - model intercept converted if on logsale
    """
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = feature_names = model[:-1].get_feature_names_out()

    try:
        coefs_df['coefs'] = model[-1].coef_
        intercept = model[-1].intercept_
    except Exception as ex:
        coefs_df['coefs'] = model[-1].regressor_.coef_
        intercept = model[-1].regressor_.intercept_

    if logscale:
        intercept = inverse_log_transform(intercept)
        coefs_df['coefs'] = inverse_log_transform(coefs_df['coefs'])

    coefs_df['abs_coefs'] = abs(coefs_df['coefs'])
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    coefs_df = coefs_df.set_index('est_int')

    return coefs_df, intercept


def coefficient_variability(model,
                            X,
                            y,
                            title,
                            suptitle,
                            splits=5,
                            repeats=20,
                            jobs=4,
                            random_state=0,
                            scoring='r2',
                            figsize=(10, 12)):
    """
    Run cross validation and plot variabililty of coefficients.

    Args:
        model - scikit-learn fitted model
        X (array-like or sparse matrix) - independent features
        y (array-like) - dependent feature (true / actual values)
        title (string) - plot title
        figsize (tuple) - size of plot
        splits (int) - used for cross validation splits
        jobs (int) - used for cross validation nr of jobs
        repeats (int) - used for cross validation repeats
        scoring (int) - scoring metric to use during cross validation
    """

    feature_names = model[:-1].get_feature_names_out()

    cv = RepeatedStratifiedKFold(n_splits=splits,
                                 n_repeats=repeats,
                                 random_state=random_state)

    cv_model = cross_validate(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv,
        return_estimator=True,
        n_jobs=jobs,
        return_train_score=True,
        verbose=0
    )

    try:
        coefs = pd.DataFrame(
            [est[-1].regressor_.coef_ for est in cv_model["estimator"]],
            columns=feature_names,
        )
    except Exception as ex:
        coefs = pd.DataFrame(
            [est[-1].coef_ for est in cv_model["estimator"]],
            columns=feature_names,
        )

    plt.figure(figsize=figsize)
    sns.stripplot(data=coefs, orient="h", palette="dark:k", alpha=0.5)
    sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=10)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient importance")
    plt.title(title)
    suptitle = suptitle
    plt.suptitle(suptitle, y=1.01, x=0.65)
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()

    return cv_model


def compare_qqplot(model1,
                   model2,
                   X,
                   y,
                   title1,
                   title2,
                   suptitle,
                   figsize=(8, 4)):
    """
    Compare QQ plot for 2 models against normal distribution.

    Args:
        model1 - scikit-learn fitted model
        model2 -  scikit-learn fitted model
        X (array-like or sparse matrix) - independent features
        y (array-like) - dependent feature (true / actual values)
        title1 (string) - plot 1 title
        title2 (string) - plot 2 title
        suptitle (str) - main title
        figsize (tuple) - size of plot
    """

    fig, (ax, ax2) = plt.subplots(1, 2,
                                  figsize=figsize,
                                  sharey=True,
                                  sharex=True)

    residuals = calculate_residuals(model1, X, y)
    qq = sm.qqplot(data=residuals['Residuals'],
                   line='q',
                   ax=ax,
                   marker='.',
                   markerfacecolor=BASE_COLOR,
                   markeredgecolor=BASE_COLOR)

    residuals = calculate_residuals(model2, X, y)
    sm.qqplot(data=residuals['Residuals'],
              line='q',
              ax=ax2,
              marker='.',
              markerfacecolor=BASE_COLOR,
              markeredgecolor=BASE_COLOR)

    ax.set_title(title1)
    ax2.set_title(title2)
    plt.suptitle(suptitle, y=1.05)
    plt.show()


def model_evaluation(model1,
                     model2,
                     X,
                     y,
                     alpha1=None,
                     alpha2=None,
                     title1=None,
                     title2=None,
                     suptitle=None):
    """
    Print various model performance plots comparing 2 models

    Args:
        model1 - scikit-learn fitted model
        model2 - scikit-learn fitted model
        X (array-like or sparse matrix) - independent features
        y (array-like) - dependent feature (true / actual values)
        alpha1 = alpha model 1 was trained with
        alpha2 = alpha model 2 was trained with
        title1 (string) - plot 1 title
        title2 (string) - plot 2 title
        suptitle (str) - main title
    """
    title1 = title1 + ' (alpha = {})'.format(alpha1)
    title2 = title2 + ' (alpha = {})'.format(alpha2)

    # prediction
    y_pred_model1 = model1.predict(X)
    y_pred_model2 = model2.predict(X)

    # 1. model performance scatter plot: actual vs predicted
    text = ' Model performance - Actual vs Predicted'
    plot_model_performance_scatter(
        y,
        y_pred_model1,
        y_pred_model2,
        title1=title1,
        title2=title2,
        suptitle=suptitle + text,
        kind='actual_vs_predicted',
        lower_lim=-100,
        upper_lim=2000,
        interval=100,
        figsize=(12, 6)
    )

    # model performance scatter plot ZOOMED: actual vs predicted
    plot_model_performance_scatter(
        y,
        y_pred_model1,
        y_pred_model2,
        title1=title1,
        title2=title2,
        suptitle=suptitle + ' Model performance - Actual vs Predicted',
        kind='actual_vs_predicted',
        lower_lim=0,
        upper_lim=300,
        interval=25,
        figsize=(12, 6)
    )

    # 2. model performance histogram plot: actual vs predicted
    text = ' Model performance - Actual vs Predicted'
    plot_model_performance_hist(
        y,
        y_pred_model1,
        y_pred_model2,
        title1=title1,
        title2=title2,
        suptitle=suptitle + text,
        xlim_max=1500,
        xlim_interval=30,
        figsize=(12, 12)
    )

    # 3. Linear Assumption Test: Distribution of Residuals'
    text = ' Linear Assumption Test: Distribution of Residuals'
    distribution_of_errors(
        model1=model1,
        model2=model2,
        X=X,
        y=y,
        kind='kde',
        title1=title1,
        title2=title2,
        suptitle=suptitle + text,
        figsize=(12, 6)
    )

    text = ' Linear Assumption Test: Distribution of Residuals'
    distribution_of_errors(
        model1=model1,
        model2=model2,
        X=X,
        y=y,
        kind='hist',
        title1=title1,
        title2=title2,
        suptitle=suptitle + text,
        figsize=(12, 6)
    )

    #  4. Linear Assumption Test: Homoscedasticity
    text = ' Linear Assumption Test: Homoscedasticity of Error Terms'
    plot_model_performance_scatter(
        y,
        y_pred_model1,
        y_pred_model2,
        title1=title1,
        title2=title2,
        suptitle=suptitle + text,
        kind='residual_vs_predicted',
        lower_lim=0,
        upper_lim=1500,
        interval=100,
        figsize=(12, 6)
    )

    text = ' Linear Assumption Test: Homoscedasticity of Error Terms'
    residual_variance(
        model1=model1,
        model2=model2,
        X=X,
        y=y,
        title1=title1,
        title2=title2,
        suptitle=suptitle + text,
        figsize=(14, 4),
    )