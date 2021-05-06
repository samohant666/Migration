import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import plotly.graph_objects as go

MAPBOX_TOKEN = 'pk.eyJ1IjoidmlqYXlzaW5naGdzbGFiIiwiYSI6ImNrbDE2dzB2bjBzZm4ydWxibWIyeG5kYXcifQ.zXZCjXTH-S2UjRmrblp76g'

SEED = 100
DO_PRINT = True
MAX_MATCH = 50
MIN_VENUES_REQUIRED = 3

list_color_0 = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                '#000075', '#808080', '#ffffff', '#fff008']
list_palette_0 = ['Blues_r', 'OrRd_r', 'BuGn_r', 'PuRd_r', 'BuPu_r', 'Wistia_r', 'binary_r', 'Blues_r', 'OrRd_r', 'BuGn_r', 'PuRd_r', 'BuPu_r', 'Wistia_r', 'binary_r' ]

PROJECT_PATH = Path.cwd()
DATA_PATH = PROJECT_PATH.joinpath('dataset')
DATA_PATH_RAW = DATA_PATH.joinpath('raw_data')
DATA_PATH_ARTIFACTS = PROJECT_PATH.joinpath('artifacts')
DATA_PATH_ARTIFACTS_APPDATA = DATA_PATH_ARTIFACTS.joinpath('app_data')

col_grain = 'Neighborhood'
col_feature = 'Venue Category'  # Venue
col_feature_name = 'Venue'
col_latitude = 'Neighborhood_Latitude'
col_longitude = 'Neighborhood_Longitude'
#col_latitude = 'location_lat'
#col_longitude = 'location_lng'

colList_coordinates = [col_latitude, col_longitude]
colList_rawData = [col_grain, col_feature, col_feature_name] + colList_coordinates
#colList_rawData = [col_grain, col_feature]
colList_meta = [col_grain]


#LIST_CITY_DATA_FILE_NAME = ['toronto_venues.csv', 'new_york_venues.csv']
LIST_CITY_DATA_FILE_NAME = ['toronto_venues.csv', 'delhi_venues.csv', 'bangalore_venues.csv', 'pune_venues.csv',
                            'hyderabad_venues.csv', 'mumbai_venues.csv']


LIST_CITY = [i.split('_venues.csv')[0] for i in LIST_CITY_DATA_FILE_NAME]


DICT_CITY_NAME_MAPPER = {
    'toronto': 'Toronto',
    'delhi': 'Delhi',
    'bangalore': 'Banglore',
    'pune': 'Pune',
    'hyderabad': 'Hyderabad',
    'mumbai': 'Mumbai'
}


def get_data_path(data_type):
    path = None
    if data_type == 'raw':
        path = DATA_PATH_RAW
    elif data_type == 'artifact_app':
        path = DATA_PATH_ARTIFACTS_APPDATA
    else:
        raise Exception('invalid data_type : {}'.format(data_type))
    return path


def read_data_file(file_name=None, data_type='raw'):
    if data_type == 'artifact_app':
        file_name = '{}_{}'.format('app', file_name)

    path = get_data_path(data_type)
    read_path = path.joinpath(file_name)
    #read_path = str(read_path)
    #print(read_path)
    if data_type == 'raw':
        X = pd.read_csv(read_path, usecols=colList_rawData, engine='python')
        X = X[colList_rawData]
    else:
        X = pd.read_csv(read_path, engine='python')
    return X


def save_data_file(X=None, file_name=None, data_type='raw'):
    if data_type == 'artifact_app':
        file_name = '{}_{}'.format('app', file_name)
    path = get_data_path(data_type)
    X.to_csv(path_or_buf=path.joinpath(file_name), index=False)
    return X


def neighborhood_haiving_min_n_venues(X_raw=None, min_venues=4):
    X = X_raw.copy()
    X['count'] = 1
    X = X.groupby(by=[col_grain, col_feature], as_index=False).count()

    arr = X[col_grain].value_counts()
    return arr[arr >= min_venues].index.values


def pre_process_raw_data(X=None):
    X = X[[col_grain, col_feature]]

    ls_neighborhood = neighborhood_haiving_min_n_venues(X_raw=X, min_venues=MIN_VENUES_REQUIRED)
    X = X[X[col_grain].isin(ls_neighborhood)]

    X['count'] = 1
    X = pd.pivot_table(X, values='count', index=col_grain, columns=col_feature, aggfunc=np.sum, fill_value=0)
    X = X.drop(columns=[col_grain], errors='ignore')
    X = X.reset_index()

    return X


def prepare_app_data_from_raw_data(file_name, return_app_data=False):
    X_raw = read_data_file(file_name, data_type='raw')
    X = pre_process_raw_data(X_raw)

    save_data_file(X=X, file_name=file_name, data_type='artifact_app')

    if return_app_data:
        return X


def get_common_feature_list(X_source=None, X_dest=None):
    list_features = set(X_source.columns.values).intersection(set(X_dest.columns.values))
    list_features = list(list_features-set(colList_meta))
    return list_features


def calculate_distance(v1, v2):
    return distance.euclidean(v1, v2)


def verbose_print(msg):
    if DO_PRINT:
        print(msg)


def vector_similarity_score(v1, arr_v2, precise_match=False):
    arr_d = []
    if precise_match:
        ls = np.nonzero(v1)
        for v2 in arr_v2:
            d = calculate_distance(np.take(v1, ls), np.take(v2, ls))
            arr_d.append(d)
    else:
        for v2 in arr_v2:
            d = calculate_distance(v1, v2)
            arr_d.append(d)
    return arr_d


def mask_array(v):
    return (v > 0).astype(np.int)


def find_top_match(d_arr=None, arr_vec=None, n=1):
    ls = np.array(d_arr).argsort()[:n]
    return np.array(d_arr)[ls], arr_vec[ls]


def sort_array(vec_1, arr_vec):
    ls = vec_1.argsort()[::-1]
    # return vec_1[ls], arr_vec[ls]
    return vec_1[ls], arr_vec.T[ls].T


def find_best_match(vec_1, arr_vec_match):
    vec_1_st, arr_vec_st = sort_array(vec_1, arr_vec_match.copy())  # col wise
    arr_vec_diff = arr_vec_st - vec_1_st

    #print(vec_1_st)
    # print(vec_1_st, arr_vec_st, arr_vec_diff)

    arr_sl = []
    for v in arr_vec_diff:
        sl, *vals = stats.linregress(np.arange(0, len(v)), v)
        arr_sl.append(sl)

    ls = np.array(arr_sl).argsort()
    return np.array(arr_sl)[ls], arr_vec_diff[ls], arr_vec_st[ls], arr_vec_match[ls]
    # return np.array(arr_sl), arr_vec_diff, arr_vec_st, arr_vec_match


def matching_vector_index(arr_vec_match, arr_vec):
    arr_i = []
    for v in arr_vec_match:
        arr = np.argwhere((v == arr_vec).all(1))[0]
        arr_i.extend(list(arr))

    return arr_i


def perform_match(vec_1=None, arr_vec=None, precise_match=False, num_match=1):
    #print(vec_1)
    vec_1_mk = mask_array(vec_1)
    arr_vec_mk = mask_array(arr_vec)

    d_arr = vector_similarity_score(vec_1_mk, arr_vec_mk, precise_match=precise_match)
    d_arr_filt, arr_vec_match = find_top_match(d_arr, arr_vec, num_match)

    ls_sl, arr_vec_diff, arr_vec_st, arr_vec_match = find_best_match(vec_1, arr_vec_match)

    list_ind_match = matching_vector_index(arr_vec_match, arr_vec)

    # for d, sl, vec_diff, vec_st, vec_match in zip(d_arr_filt, ls_sl, arr_vec_diff, arr_vec_st, arr_vec_match):
    #    print('{} --> {}, {} : {}, {}'.format(vec_match, vec_st, vec_diff, np.round(sl,2), np.round(1.2,2)))

    return list_ind_match


def prepare_sorted_match_df(X_source=None, X_match=None, nv=1, colList_features=None):
    X = X_source.copy()
    X['index'] = -1
    X = X.set_index('index')

    df = pd.concat(objs=[X[colList_features], X_match[colList_features]])

    # X_source is already a selected row
    vec_source = X_source.values[0]
    ls_sort = vec_source.argsort()[::-1][:nv]
    df = df.iloc[:, ls_sort].copy()
    return df


def preapre_venue_plot_data(X_match_sorted=None, X_meta_mapper=None, colList_features=None):
    X = X_match_sorted.copy()
    ls_df_order = X.index.values  # use later for sorting

    # order of X is imp
    X = X.reset_index()
    X = pd.merge(left=X_meta_mapper, right=X, on='index', how='right')

    #X = X.drop_duplicates(subset=[col_grain], keep='first')  # new change in app

    ls_features = [i for i in colList_features if i in X.columns.values]
    plot_df = pd.melt(frame=X, id_vars=col_grain, value_vars=ls_features)
    return X, plot_df


def get_source_vector(X=None, source_name=None, return_df=False, colList_features=None):
    if return_df:
        return X.loc[X['Neighborhood'] == source_name][colList_features].copy()
    else:
        return X.loc[X['Neighborhood'] == source_name][colList_features].values[0]


def get_match_df(X=None, list_ind=None, colList_features=None):
    df = X[colList_features].copy()
    df = df.loc[list_ind]

    return df


def get_sorted_list_of_features(X_source_selected=None, colList_features=None):
    vec_source = X_source_selected.values[0]
    vec_source = vec_source.argsort()[::-1]

    arr = X_source_selected[colList_features].columns.values
    arr = arr[vec_source]
    return arr


def prepare_meta_mapper(X_dest=None, colList_meta=None, source_name=None):
    X = X_dest.copy()
    X = X[colList_meta].reset_index()
    X.loc[X.index.max()+1] = [-1, source_name]
    return X


def perform_match_wrapper(X_source=None, X_dest=None, source_name=None, num_match=None, precise_match=True,
                          colList_features=None, colList_meta=None):
    if num_match is None:
        if len(X_dest) <= 50:
            num_match = len(X_dest)-2
        else:
            num_match = MAX_MATCH

    # input data
    X_meta_mapper = prepare_meta_mapper(X_dest=X_dest, colList_meta=colList_meta, source_name=source_name)
    vec_1 = get_source_vector(X=X_source, source_name=source_name, colList_features=colList_features, return_df=False)
    #arr_vec = X_dest.drop(columns=[col_grain]).values  # ph2 change
    arr_vec = X_dest[colList_features].values

    # matching
    list_ind_match = perform_match(vec_1=vec_1, arr_vec=arr_vec, precise_match=precise_match, num_match=num_match)
    X_match = get_match_df(X=X_dest, list_ind=list_ind_match, colList_features=colList_features)

    return X_match, X_meta_mapper


def visualize_venue_match_results_wrapper(X_source=None, X_match=None, X_meta_mapper=None,
                                          source_name=None, colList_features=None, num_match=1, num_venues=1,
                                          show_plot=False):
    X_match = X_match.head(num_match)
    # prepare plot data
    X_source_selected = get_source_vector(X=X_source, source_name=source_name, return_df=True,
                                          colList_features=colList_features)
    X_match_sorted = prepare_sorted_match_df(X_source=X_source_selected, X_match=X_match,
                                             colList_features=colList_features, nv=num_venues)
    X_match_sorted_named, plot_df = preapre_venue_plot_data(X_match_sorted=X_match_sorted, X_meta_mapper=X_meta_mapper,
                                                            colList_features=colList_features)

    # plot
    ls_feature_sorted = get_sorted_list_of_features(X_source_selected=X_source_selected,
                                                    colList_features=colList_features)
    plot = plot_venue_match_data(plot_df=X_match_sorted_named, num_match=num_match, num_venues=num_venues,
                                 colList_features=colList_features, show_plot=show_plot)

    return X_match_sorted_named, plot


def plot_venue_match_data(plot_df=None, num_match=-1, num_venues=-1, colList_features=None, show_plot=False):

    # drop dup entry of nbhd
    #mac_vc = plot_df[col_grain].value_counts().values[0]
    #if mac_vc > 1:
    #    plot_df = plot_df.drop_duplicates(subset=[col_grain])
    #    num_match = num_match-1

    sns.set_style('darkgrid')
    fig, axis = plt.subplots(num_match, 1, figsize=(1.5 * num_venues, num_match * 4))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    list_features = [i for i in plot_df.columns if i in colList_features]
    list_match = plot_df[col_grain].values
    ylim = plot_df[list_features].max().max()

    for i_main in range(num_match):
        data_temp = plot_df.loc[i_main][list_features]
        ls_venues = data_temp.index.values
        ls_count = data_temp.values
        ax = axis[i_main]
        sns.barplot(x=ls_venues, y=ls_count, ax=ax, palette=sns.color_palette(list_palette_0[i_main], num_venues, 0.9))

        ax.set_ylabel(ylabel='', fontsize=12, color='red')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ylim = max(ls_count)
        ax.set_ylim(0, ylim + 1)

        ax.set_xticklabels(ls_venues, rotation=15, horizontalalignment='right', size=12)
        # ax.text((len(ls_venues)-1)//2.5, ylim, list_match[i_main], fontsize=15)
        # ax.text(int(len(ls_venues)*0.4), ylim-1, list_match[i_main], fontsize=15)

        if i_main == 0:
            # ax.text(int(len(ls_venues)*0.4), ylim-1, 'Source City\n{}'.format(list_match[i_main]), fontsize=12)
            ax.set_title('{}\n(Source City)'.format(list_match[i_main]), loc='center', pad=-10,
                         fontdict={'verticalalignment': 'center_baseline'})
        else:
            # ax.text(int(len(ls_venues)*0.4), ylim-1, list_match[i_main], fontsize=12)
            ax.set_title('{}'.format(list_match[i_main]), loc='center', pad=-10,
                         fontdict={'verticalalignment': 'center_baseline'})

    main_label = 'Comparison of suggested Neighborhood with the base location\nnum neighborhood : {}\nnum top venues : {}'.format(
        num_match, num_venues)
    fig.text(0.5, 1.02, main_label, ha='center', va='center', rotation='horizontal', size=18, color='blue')

    plt.tight_layout()

    if show_plot:
        plt.show()

    return fig


def generate_ui_df(X_match_sorted_named=None):
    df = X_match_sorted_named.drop(columns=['index']).T.copy()
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    return df


def plot_nbhd_on_map(plot_df=None, marker_size=20, map_zoom=11, show_plot=False, col_text=col_grain):
    ls_symbol = ['triangle' for i in range(len(plot_df))]

    fig = go.Figure(go.Scattermapbox(
        mode="markers+text",
        lon=plot_df[col_longitude], lat=plot_df[col_latitude],
        marker={'size': marker_size, 'symbol': ls_symbol},
        hovertext=plot_df[col_text],
        text=plot_df[col_text], textposition="bottom right"))

    fig.update_layout(
        mapbox={
            'accesstoken': MAPBOX_TOKEN,
            'center': go.layout.mapbox.Center(
                lat=plot_df[col_latitude].mean(),
                lon=plot_df[col_longitude].mean()
            ),
            'style': "outdoors", 'zoom': map_zoom},
        showlegend=False)

    # fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    if show_plot:
        fig.show()
    return fig
