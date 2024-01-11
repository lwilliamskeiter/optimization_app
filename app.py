from shiny import App, reactive, render, ui
import pandas as pd
import numpy as np
from sklearn import preprocessing
import regex as re
from bs4 import BeautifulSoup

# shiny run --reload optimize_projects/app.py

#%% Functions
def normalize(col):
    return preprocessing.normalize(col.to_numpy().reshape(1, -1))[0]

def numeric_filter(df,col):
    step = .01 if 'float' in str(df[col].dtypes) else 1
    return ui.input_slider(re.sub('[ \(\)]','_',str(col)+str(1)), col,
                           min=df[col].min(), max=df[col].max(),
                           value=(df[col].min(),df[col].max()), step=step)

def cat_filter(df,col):
    return ui.input_selectize(re.sub('[ \(\)]','_',str(col)+str(1)), col,
                              choices = list(df[col].replace(np.nan,'NaN').drop_duplicates()),
                                multiple=True
                              )


#%% App UI
app_ui = ui.page_fluid(
    ui.panel_title("Optimal Projects"),
    
    ui.row(
        ui.column(
            3,
            ui.br(),
            ui.input_file('projects_file','Upload Projects File',accept='.xlsx'),
            ),
    ),
    
    ui.row(
        ui.column(
            3,
            ui.br(),
            ui.output_ui('vars_to_filter'),
            ui.output_ui('filter_vars_button'),
            ui.br(),
            ui.output_ui('create_filters')
            ),
        ui.column(
            3,
            ui.br(),
            ui.output_ui('vars_to_max'),
            ui.output_ui('vars_to_min'),
            ui.output_ui('minmax_vars_button'),
            offset = .1
            ),
        
        ui.column(
            6,
            ui.br(),
            ui.output_table("table")
            )
        )
)

#%% App Server
def server(input, output, session):
    
    filter_inputs = list()
    
    # initialize values for action button values
    filter_btn_val = False
    minmax_btn_val = False
    
    def reset_filter_btn():
        filter_btn_val = False
    def reset_minmax_btn():
        minmax_btn_val = False
    
    @reactive.Calc
    @reactive.event(input.projects_file)
    def read_file():
        if input.projects_file() is not None:
            file = input.projects_file()[0]
            df = pd.read_excel(file['datapath'], sheet_name='projects')
            return df
    
    @output
    @render.ui
    @reactive.event(input.projects_file)
    def vars_to_filter():
        if input.projects_file() is not None:
            return ui.row(
                ui.input_selectize(
                    'filter_vars','Variables to filter',
                    # Only allow numeric choices
                    choices = read_file().columns.to_list(),#+['None'] ,
                    multiple = True
                    ),
                )
    
    @output
    @render.ui
    @reactive.event(input.filter_vars)
    def filter_vars_button():
        if input.projects_file() is not None:
            return ui.input_action_button('select_filters','Confirm variables to filter',width='70%',class_="btn-success",style="display: inline-block; margin-left: 15px;")
    
    @output
    @render.ui
    @reactive.event(input.min_vars)
    def minmax_vars_button():
        if input.projects_file() is not None and input.max_vars() and input.min_vars():
            return ui.input_action_button('select_mins','Confirm selection',width='70%',class_="btn-success",style="display: inline-block; margin-left: 15px;")
    
    @output
    @render.ui
    @reactive.event(input.select_filters)
    def create_filters():
        if input.projects_file() is not None:

            # if 'None' in list(input.filter_vars()):
            #     return ui.h4('No data filters confirmed!')
            if len(list(input.filter_vars()))==1:
                if re.search('int|float',str(read_file()[list(input.filter_vars())[0]].dtypes)):
                    row = numeric_filter(read_file(),list(input.filter_vars())[0])
                else:
                    row = cat_filter(read_file(),list(input.filter_vars())[0])
                    
                filter_inputs.append(row)
                return row
            
            elif len(input.filter_vars())>1:
                row = [numeric_filter(read_file(),i) if re.search('int|float',str(read_file()[i].dtypes))
                       else cat_filter(read_file(),i)
                       for i in list(input.filter_vars())]
                filter_inputs.append(row)
                return row
    
    @output
    @render.ui
    @reactive.event(input.projects_file)
    def vars_to_max():
        if input.projects_file() is not None:
            return ui.input_select(        
                'max_vars','Variable to maximize',
                # Only allow numeric choices
                choices = read_file().columns[[bool(re.search('int|float',str(x))) for x in read_file().dtypes]].to_list(),
                size = 1)
    
    @output
    @render.ui
    # @reactive.event(input.max_vars)
    def vars_to_min():
        if input.projects_file() is not None and input.max_vars():
            return ui.row(
                ui.input_selectize(
                    'min_vars','Variables to minimize',
                    # Only allow numeric choices
                    choices = [x for x in read_file().columns[[bool(re.search('int|float',str(x))) for x in read_file().dtypes]] if x not in input.max_vars()],
                    multiple = True)
                )
    
    @reactive.Calc
    @reactive.event(input.select_mins, input.select_filters)
    def projects():
        if input.projects_file() is not None:
            df = read_file().copy()
            # Normalize selected variables
            df[list([input.max_vars()])+list(input.min_vars())] = df[list([input.max_vars()])+list(input.min_vars())].apply(lambda x: normalize(x))
            # Calculate "rank"
            df['Rank'] = .5*df[input.max_vars()] +.5*df[list(input.min_vars())].apply(np.mean,axis=1)

            return df
    
    @reactive.Calc
    def filter_table():
        if input.projects_file() is not None and projects() is not None and len(input.filter_vars())>0:
            
            df = projects()
            # Initialize list for filters
            cat_filters = list()
            
            # Categorical filter inputs
            cat_inputs = [x for x in input.filter_vars() if str(read_file()[x].dtypes)=='object']
            # Numerical filter inputs
            num_inputs = [x for x in input.filter_vars() if str(read_file()[x].dtypes)!='object']
            
            # print([[str(type(x))=="<class 'str'>" for x in getattr(input, re.sub('[ \(\)]','_',str(input.filter_vars()[y]))+'1')()] 
            #        for y in range(len(input.filter_vars()))])
            
            if len(cat_inputs)>0 and any([len([x for x in getattr(input, re.sub('[ \(\)]','_',str(i))+'1')()])>0 for i in cat_inputs]):
                # Get T/F for selected categorical inputs
                for i in cat_inputs:
                    cat_filters.append(projects()[i].isin([x for x in getattr(input, re.sub('[ \(\)]','_',str(i))+'1')()]))
                # Concat categorical filter inputs
                cat_filters = pd.concat(cat_filters,axis=1)
                df = df[[any(cat_filters.loc[x]) for x in range(len(cat_filters))]]
                # print()
                # df = df[[all(cat_filters.loc[x]) for x in range(len(cat_filters))]]
            else:
                df = projects()
            
            if len(num_inputs)>0:
                for i in num_inputs:
                    # Get min and max of numeric filter input
                    num_filters = [x for x in getattr(input, re.sub('[ \(\)]','_',str(i))+'1')()]
                    # Filter df based on min/maxs
                    df = df[(df[i] >= num_filters[0]) & (df[i] <= num_filters[1])]
                                
            # Return filtered dataset
            return df
    
    @render.table
    def table():
        if input.projects_file() is not None and projects() is not None:
            
            # Categorical filter inputs
            cat_inputs = [x for x in input.filter_vars() if str(read_file()[x].dtypes)=='object']
            # Numerical filter inputs
            num_inputs = [x for x in input.filter_vars() if str(read_file()[x].dtypes)!='object']
            
            # If no categorical filters or no choices selected, just display head of full dataset
            if (all([len([x for x in getattr(input, re.sub('[ \(\)]','_',str(i))+'1')()])==0 for i in cat_inputs]) 
                and all([read_file()[i].min() >= [x for x in getattr(input, re.sub('[ \(\)]','_',str(i))+'1')()][0]
                         and read_file()[i].max() <= [x for x in getattr(input, re.sub('[ \(\)]','_',str(i))+'1')()][1]
                         for i in num_inputs])):
                return projects()[['Objective','Task','Rank']].sort_values('Rank',ascending=False).head()
            else:
                return filter_table()[['Objective','Task','Rank']].sort_values('Rank',ascending=False).head(min(len(filter_table()),10))


#%% Run app
app = App(app_ui, server)
