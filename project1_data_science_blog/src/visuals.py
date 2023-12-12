import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


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
    plt.rc('xtick', labelsize=small_size)  # fontsize of the ytick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the xtick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=bigger_size, titleweight="bold", figsize=[8, 4])  # fontsize of the figure title

    return (base_color, base_highlight_intense, base_highlight, base_complimentary, base_grey,
            base_color_group1, base_color_group2, symbols)


# set default plot formatting
(BASE_COLOR, BASE_HIGHLIGHT_INTENSE, BASE_HIGHLIGHT, BASE_COMPLIMENTARY, BASE_GREY, BASE_COLOR_ARR, BASE_COLOR_DEP, SYMBOLS) = set_plot_defaults()


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
        plt.figure(figsize=[14, 4])
        weight = 'ultralight'
        rotation = 90
        color = 'black'
        xytext = (0, 8)
        size = 6
    else:
        plt.figure(figsize=[14, 4])
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

    # annote bars with % of total flights
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
                
    plt.figure(figsize=(12,6))

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
                             figsize=(6,10), 
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
        labels2=[f'{x*100:,.1f}%' for x in c.datavalues]

        labels1 = [f"(Avg: £{top.loc[(top[hue].eq(col) & top.percentage.eq(h)), 'price_mean'].iloc[0]} pn)" if (h := v.get_width()) > 0 else '' for v in c]

        # add the name annotation to the top of the bar
        ax.bar_label(c, labels=labels2, padding=2, label_type='edge', color='black', weight='ultralight', fontsize=8)  

        # add the name annotation to the top of the bar
        ax.bar_label(c, labels=labels1, padding=30, label_type='edge', color='black', fontweight='ultralight', fontsize=8) 

    # leave space at end of plot for extra text
    ax.margins(x=0.4)    
    binsize = 0.1
    ax.grid(show_gridlines)
    xticks = np.arange(0, 1 + binsize, binsize)
    xlabels = ['{:1.0f}%'.format(tick*100) for tick in xticks]       
    plt.xticks(xticks, xlabels)    

    plt.legend(facecolor='w', bbox_to_anchor=(1, 1), loc='upper left', title=legend_title, 
               fontsize='small', title_fontsize='large')    
    plt.title(title)

    plt.show()

    
def plot_categories(df, 
                    annotate=True, 
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

    ax1 = sns.barplot(data=df, y=df.index, x='individual', color=BASE_GREY,  
                      label='Individual', errorbar=None)
    ax2 = sns.barplot(data=df, y=df.index, x='business', color=BASE_COLOR,  
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
        figsize=(6,2)
    elif df[col].nunique() <= 6:
        figsize=(6,3)
    else: 
        figsize=(12,8)    
    
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
