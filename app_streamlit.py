import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import regex as re


#%% Functions
def normalize(col):
    return preprocessing.normalize(col.to_numpy().reshape(1, -1))[0]

def d_max(col):
    col_min = col.min()
    col_max = col.max()

    return (col - col_min)/(col_max-col_min)

def d_min(col):
    col_min = col.min()
    col_max = col.max()

    return abs((col - col_max)/(col_min-col_max))

def create_num_filter(df,col):
    step = 0.1 if 'float' in str(df[col].dtypes) else 1
    st.slider(col,df[col].min(),df[col].max(),(df[col].min(),df[col].max()),step=step,key=re.sub('[ \(\)]','_',col)+'1')

def create_cat_filter(df,col):
    st.multiselect(col,options=df[col].drop_duplicates().to_list(),key=re.sub('[ \(\)]','_',col)+'1')


#%% App

st.set_page_config(
     page_title='Optimal Projects',
     layout="wide",
)

st.markdown("## Optimal Projects")

col1, col2, col3 = st.columns([3,3,6],gap='medium')

with col1:
    read_file = st.file_uploader('Upload Projects File')
    # If file is not uploaded or not an xlsx, return warning
    if read_file is None or (read_file is not None and not bool(re.search('xlsx',str(read_file.name)))):
        st.warning('You need to upload an excel file.')
    else:
        projects_df = pd.read_excel(read_file,sheet_name='projects')

        max_vars = st.selectbox(
            'Variable to maximize',
            key='var_max',
            options=projects_df.columns[[bool(re.search('int|float',str(x))) for x in projects_df.dtypes]].to_list(),
            index=None
        )

        min_vars = st.multiselect(
            'Variables to minimize',
            key = 'var_min',
            options = projects_df.columns[[bool(re.search('int|float',str(x))) for x in projects_df.dtypes]].to_list()
        )

with col2:
    if read_file is None or (read_file is not None and not bool(re.search('xlsx',str(read_file.name)))):
        st.empty()
    else:
        filter_vars = st.multiselect(
                'Variables to filter',
                key='filter_vars',
                options = projects_df.columns
            )
        
        # Dynamically create filters
        if len(filter_vars)>0:

            cat_inputs = [x for x in filter_vars if 'object' in str(projects_df[x].dtypes)]
            num_inputs = [x for x in filter_vars if 'object' not in str(projects_df[x].dtypes)]

            if len(cat_inputs)>0:
                for i in cat_inputs:
                    create_cat_filter(projects_df,i)
            
            if len(num_inputs)>0:
                for i in num_inputs:
                    create_num_filter(projects_df,i)



with col3:
    if read_file is None or (read_file is not None and not bool(re.search('xlsx',str(read_file.name)))):
        st.empty()
    else:
        if max_vars is not None and min_vars is not None:
            df = projects_df.copy()

            # Normalize selected min/max variables
            # df[[min_vars][0] + [max_vars]] = df[[min_vars][0] + [max_vars]].apply(lambda x: normalize(x))
            if len([max_vars])==1:
                df[max_vars] = d_max(df[max_vars])
            # print(len(max_vars))
            else:
                df[max_vars] = df[max_vars].apply(lambda x: d_max(x))
            df[min_vars] = df[min_vars].apply(lambda x: d_min(x))
            # Calculate 'rank'
            df['Rank'] = .5*df[max_vars] +.5*df[min_vars].apply(np.mean,axis=1)

            if len(filter_vars)>0:
                    
                cat_inputs = [x for x in filter_vars if 'object' in str(projects_df[x].dtypes)]
                num_inputs = [x for x in filter_vars if 'object' not in str(projects_df[x].dtypes)]

                if len(cat_inputs)>0 and any([len(st.session_state[re.sub('[ \(\)]','_',i)+'1'])>0 for i in cat_inputs]):

                    cat_filters = list()

                    for i in cat_inputs:
                        cat_filters.append(
                            df[i].isin(st.session_state[re.sub('[ \(\)]','_',i)+'1'])
                        )
                    
                    cat_filters = pd.concat(cat_filters,axis=1)
                    df = df[[any(cat_filters.loc[x]) for x in range(len(cat_filters))]]
                
                if len(num_inputs)>0:

                    for i in num_inputs:
                        # Get min and max of numeric filter input
                        num_filters = [x for x in st.session_state[re.sub('[ \(\)]','_',i)+'1']]
                        # Filter df based on min/maxs
                        df = df[(df[i] >= num_filters[0]) & (df[i] <= num_filters[1])]

            st.dataframe(
                pd.concat([
                    df[['Objective','Task','Rank']],
                    projects_df[['External Hours Estimated (Median)','Internal Hours Estimated (Median)','business value','success rate']]
                ],axis=1).sort_values('Rank',ascending=False).head(min(len(df),10)),
                hide_index=True
            )

