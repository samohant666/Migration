import os
import re
import streamlit as st

code = """<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-611FPFGKT5"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'UA-611FPFGKT5');
</script>"""


a = os.path.dirname(st.__file__)+'/static/index.html'
with open(a, 'r') as f:
    data = f.read()
    if len(re.findall('UA-', data)) == 0:
        with open(a, 'w') as ff:
            newdata = re.sub('<head>', '<head>'+code, data)
            ff.write(newdata)


from utils import perform_match_wrapper, visualize_venue_match_results_wrapper, generate_ui_df, \
    LIST_CITY_DATA_FILE_NAME, read_data_file, get_common_feature_list, LIST_CITY, col_grain, colList_meta, \
    plot_nbhd_on_map, col_feature_name

WELCOME_MSG = """<p class="big-font">We understand! Moving to a new city is stressful. A neighborhood that can give you a similar lifestyle. If you ask people, you would get biased opinions. No worries! Using the "Machine Learning" algorithms, we solve the problem for you. We scan your current neighborhood and recommend the most suitable neighborhood in the new city.</p>"""


#@st.cache()
def read_dest_city_data(dest_city_name):
    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if dest_city_name in i][0]
    X_dest = read_data_file(file_name=file_name, data_type='artifact_app')
    return X_dest


#@st.cache()
def caching_perform_match_wrapper(source_name=None, X_source=None, X_dest=None):
    colList_features = get_common_feature_list(X_source=X_source, X_dest=X_dest)
    X_match, X_meta_mapper = perform_match_wrapper(X_source=X_source, X_dest=X_dest, source_name=source_name,
                                                   num_match=None, precise_match=True,
                                                   colList_features=colList_features, colList_meta=colList_meta)
    return X_match, X_meta_mapper, colList_features


#@st.cache()
def caching_visualize_venue_match_results_wrapper(X_source=None, X_match=None, X_meta_mapper=None, source_name=None,
                                                  colList_features=None, num_match=None, num_venues=None, show_plot=False):
    X_match_sorted_named, graph_count = visualize_venue_match_results_wrapper(X_source=X_source, X_match=X_match,
                                                                              X_meta_mapper=X_meta_mapper,
                                                                              source_name=source_name,
                                                                              colList_features=colList_features,
                                                                              num_match=num_match,
                                                                              num_venues=num_venues,
                                                                              show_plot=show_plot)
    return X_match_sorted_named, graph_count


#@st.cache()
#@st.cache(suppress_st_warning=True)
def plot_count_graph(graph_count=None, dest_city_name=None):
    with st.spinner('Generating the Venue Comparison plot of neighborhood present in {} ....'.format(dest_city_name)):
        st.pyplot(graph_count)


def get_source_city_info(source_city_name):
    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if source_city_name in i][0]
    X_source = read_data_file(file_name=file_name, data_type='artifact_app')
    list_sources_venues = list(X_source[col_grain].values)
    return X_source, list_sources_venues


#@st.cache()
#@st.cache(suppress_st_warning=True)
def plot_match_df(df=None):
    df = df.copy()
    df = df.drop_duplicates(subset=[col_grain])
    df = df.drop(columns=['index'])
    st.dataframe(df)


def prepare_nbhd_map(dest_city_name=None, X_match_df=None):
    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if dest_city_name in i][0]
    X_dest_raw = read_data_file(file_name=file_name, data_type='raw')
    # map1
    plot_df_map_nbhd = X_dest_raw[X_dest_raw[col_grain].isin(X_match_df[col_grain])].copy()
    plot_df_map_nbhd = plot_df_map_nbhd.drop_duplicates(subset=[col_grain])
    map_graph_nbhd = plot_nbhd_on_map(plot_df=plot_df_map_nbhd, marker_size=20, map_zoom=11,
                                      show_plot=False)
    return map_graph_nbhd


def plot_nbhd_graph(plotly_map=None, dest_city_name=None):
    st.markdown('')
    st.markdown('### Neighborhood map of {}:'.format(dest_city_name))
    c_arr = st.beta_columns(7)
    c_arr[1].plotly_chart(plotly_map)


def main():
    html_temp = """ 
    <div style ="background-color:yellow;padding:1px"> 
    <h1 style ="color:black;text-align:center;">Neighborhood Recommender</h1> 
    </div> 
    """

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.title('About')
    st.markdown("""
                    <style>
                    .big-font {
                        font-size:12px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
    st.sidebar.markdown(WELCOME_MSG, unsafe_allow_html=True)
    st.sidebar.markdown("""<p class="big-font">Isn't it cool? Try it!</p>""", unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.beta_columns(2)

    # L1-inputs
    SOURCE_CITY = col1.selectbox('Moving from city:', LIST_CITY)

    # L3 input
    same_city = st.checkbox("moving within same city")
    if same_city:
        DEST_CITY = SOURCE_CITY
        st.markdown("#### Awesome! Let's find a new neighborhood for you in {}.".format(DEST_CITY))
    else:
        DEST_CITY = col2.selectbox('Moving to city:', LIST_CITY)
        st.markdown("#### Great! Let's explore {}.".format(DEST_CITY))

    # L2 input
    X_source, list_sources_venues = get_source_city_info(source_city_name=SOURCE_CITY)
    SOURCE_VENUE = st.selectbox('To get suitable locations, please select the neighborhood you are moving from:',
                                list_sources_venues)

    # L2-inputs
    text = "Choose number of matching neighborhood to be displayed: "
    NUM_MATCH = st.sidebar.slider(text, min_value=3, max_value=8, value=4, step=1)
    text = "Choose number of venues to be displayed: "
    NUM_VENUES = st.sidebar.slider(text, min_value=4, max_value=15, value=10, step=1)

    # caching
    X_dest = read_dest_city_data(dest_city_name=DEST_CITY)

    if st.button("Search"):
        # caching
        X_match, X_meta_mapper, colList_features = caching_perform_match_wrapper(source_name=SOURCE_VENUE,
                                                                                 X_source=X_source,
                                                                                 X_dest=X_dest)

        num_match = NUM_MATCH
        num_venues = NUM_VENUES
        source_name = SOURCE_VENUE
        with st.spinner('Finding the right neighborhood for you ....'):
            X_match_sorted_named, graph_count = caching_visualize_venue_match_results_wrapper(X_source=X_source, X_match=X_match,
                                                                                              X_meta_mapper=X_meta_mapper,
                                                                                              source_name=source_name,
                                                                                              colList_features=colList_features,
                                                                                              num_match=num_match,
                                                                                              num_venues=num_venues, show_plot=False)

            st.success('Here are a few neighborhood/s suggestion for you. Good luck!')
            plot_match_df(df=X_match_sorted_named)

            with st.spinner('Generating the map of neighborhood present in {} ....'.format(DEST_CITY)):
                # use mean values of lat/long (Imppp)
                map_graph_nbhd = prepare_nbhd_map(dest_city_name=DEST_CITY, X_match_df=X_match_sorted_named)
                plot_nbhd_graph(plotly_map=map_graph_nbhd, dest_city_name=DEST_CITY)

            plot_count_graph(graph_count=graph_count, dest_city_name=DEST_CITY)


if __name__ == '__main__':
    main()
