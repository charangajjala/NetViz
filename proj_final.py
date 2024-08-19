import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
import io
import base64
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid

from sklearn.metrics.pairwise import euclidean_distances
import os
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


train_df = pd.read_csv('./sampled_df_float.csv')
tsne_data = pd.read_csv('./tsne_final_result_2D.csv')
app = dash.Dash(__name__)

# Data Modification
def apply_scaling(data, scaling_technique):
    if scaling_technique == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_technique == 'standard':
        scaler = StandardScaler()
    elif scaling_technique == 'power':
        scaler = PowerTransformer()
    elif scaling_technique == 'quantile':
        scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')
    elif scaling_technique == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError('Invalid scaling technique provided')

    if scaling_technique not in ['power', 'quantile', 'robust']:
        features_to_normalize = data.drop(['attack_cat', 'label'], axis=1).columns
        data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
    else:
        if scaling_technique == 'power':
            features_to_normalize = data.drop(['attack_cat', 'label'], axis=1).columns
            data[features_to_normalize] = data[features_to_normalize] + 1e-6
        features_to_normalize = data.drop(['attack_cat', 'label'], axis=1).columns
        data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

    return data

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='app-content'),
    # dcc.Dropdown(id='scaling-technique-pca', style={'display': 'none'}),
    # dcc.Dropdown(id='scaling-technique-lda', style={'display': 'none'}),
    dcc.RadioItems(id='plot-type-pca', style={'display': 'none'}),
    dcc.Graph(id='pca-plot',style={'display': 'none'}),
    dcc.RadioItems(id='plot-type-lda', style={'display': 'none'}),
    dcc.Graph(id='lda-plot',style={'display': 'none'}),
    dcc.Graph(id='tsne-plot',style={'display': 'none'}),
    html.Div(id='heatmap-content',style={'display': 'none'}),
    dcc.Graph(id='bar-plot',style={'display': 'none'}),
    dcc.Graph(id='elastic-net-coef-plot',style={'display': 'none'}),
    dcc.Graph(id='elastic-net-mse-plot',style={'display': 'none'}),
    dcc.Graph(id='correration-matrix-plot',style={'display': 'none'}),
    dcc.Graph(id='important-features-plot',style={'display': 'none'}),
    dcc.Graph(id='kmeans-plot',style={'display': 'none'}),
    dcc.Graph(id='random-forest-plot',style={'display': 'none'}),
])


def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data.drop(['attack_cat', 'label'], axis=1))
    return pca_result

def apply_lda(data, n_components):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X = data.drop(['attack_cat', 'label'], axis=1)
    y = data['attack_cat']
    lda_result = lda.fit_transform(X, y)
    return lda_result


@app.callback(
        Output('app-content', 'children'), 
        [Input('url', 'pathname')],
)
def display_page(pathname):
    if pathname == '/bv':
        return basic_visulation_layout()
    elif pathname == '/fs':
        return feature_selection_layout()
    elif pathname == '/civ':
        return class_imbalance_page_layout()
    elif pathname == '/pca':
        return pca_layout()
    elif pathname == '/tsne':
        return tsne_layout()
    elif pathname == '/heatmap':
        return heatmap_layout()
    elif pathname == '/barplot':
        return barplot_layout()
    elif pathname == '/elasticnet':
        return elastic_net_layout()
    elif pathname == '/lda':
        return lda_layout()
    elif pathname == '/kmeans':
        return kmeans_layout()
    elif pathname == '/randomforest':
        return random_forest_layout()
    else:
        return home_layout()


# PCA Layout
def pca_layout():
    return html.Div([
        html.H1("PCA Plots", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        dcc.RadioItems(
            id='plot-type-pca',
            options=[
                {'label': 'PCA 2D', 'value': 'pca2d'},
                {'label': 'PCA 3D', 'value': 'pca3d'}
            ],
            labelStyle={'display': 'block'}
        ),

        # dcc.Dropdown(
        #     id = 'scaling-technique-pca',
        #     value='minmax',
        #     placeholder='Select a scaling technique',
        #     style={'display': 'none'}
        # ),

        html.Div(
            dcc.Loading(
                id='loading-pca',
                children=dcc.Graph(id='pca-plot'),
                type='circle'
            )
        )
    ], style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

# LDA Layout
def lda_layout():
    return html.Div([
        html.H1("LDA Plots", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        dcc.RadioItems(
            id='plot-type-lda',
            options=[
                {'label': 'LDA 2D', 'value': 'lda2d'},
                {'label': 'LDA 3D', 'value': 'lda3d'}
            ],
            labelStyle={'display': 'block'}
        ),

        # dcc.Dropdown(
        #     id = 'scaling-technique-pca',
        #     value='minmax',
        #     placeholder='Select a scaling technique',
        #     style={'display': 'none'}
        # ),

        html.Div(
            dcc.Loading(
                id='loading-lda',
                children=dcc.Graph(id='lda-plot'),
                type='circle'
            )
        )
    ], style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})


# PCA Callbacks
@app.callback(
    Output('pca-plot', 'figure'),
    [Input('plot-type-pca', 'value'),
    #  Input('scaling-technique-pca', 'value'),
     Input('url', 'pathname')],
    prevent_initial_call=True
)
def update_pca_plot(plot_type, pathname):
    if pathname == '/pca':
        if plot_type is None:
            return dash.no_update
        else:
            scaling_technique = 'minmax'
            scaled_data = apply_scaling(train_df.copy(), scaling_technique)
            if plot_type == 'pca2d':
                pca_result = apply_pca(scaled_data, 2)
                plot_df = pd.DataFrame(data=pca_result, columns=['Component 1', 'Component 2'])
                plot_df['Attack_Type'] = scaled_data['attack_cat']
                fig_pca = px.scatter(plot_df, x='Component 1', y='Component 2', color='Attack_Type',
                                        title='PCA 2D Plot with Attack Types', opacity=0.7)
            else:
                pca_result = apply_pca(scaled_data, 3)
                plot_df = pd.DataFrame(data=pca_result, columns=['Component 1', 'Component 2', 'Component 3'])
                plot_df['Attack_Type'] = scaled_data['attack_cat']
                fig_pca = px.scatter_3d(plot_df, x='Component 1', y='Component 2', z='Component 3',
                                        color='Attack_Type', title='PCA 3D Plot with Attack Types', opacity=0.7)
                fig_pca.update_layout(width=800, height=600)

            return fig_pca
    else:
        return dash.no_update
    
# LDA Callbacks
# Similar to the PCA Callbacks
@app.callback(
    Output('lda-plot', 'figure'),
    [Input('plot-type-lda', 'value'),
    #  Input('scaling-technique-lda', 'value'),
     Input('url', 'pathname')],
    prevent_initial_call=True
)
def update_lda_plot(plot_type, pathname):
    if pathname == '/lda':
        if plot_type is None:
            return dash.no_update
        else:
            scaling_technique = 'minmax'
            scaled_data = apply_scaling(train_df.copy(), scaling_technique)
            if plot_type == 'lda2d':
                lda_result = apply_lda(scaled_data, 2)
                plot_df = pd.DataFrame(data=lda_result, columns=['Component 1', 'Component 2'])
                plot_df['Attack_Type'] = scaled_data['attack_cat']
                fig_lda = px.scatter(plot_df, x='Component 1', y='Component 2', color='Attack_Type',
                                        title='LDA 2D Plot with Attack Types', opacity=0.7)
            else:
                lda_result = apply_lda(scaled_data, 3)
                plot_df = pd.DataFrame(data=lda_result, columns=['Component 1', 'Component 2', 'Component 3'])
                plot_df['Attack_Type'] = scaled_data['attack_cat']
                fig_lda = px.scatter_3d(plot_df, x='Component 1', y='Component 2', z='Component 3',
                                        color='Attack_Type', title='LDA 3D Plot with Attack Types', opacity=0.7)
            return fig_lda
    else:
        return dash.no_update

# tsne layout
def tsne_layout():
    return html.Div(children=[
        html.H1("t-SNE Plots", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        html.Div(
            dcc.Loading(
                id="loading-tsne",
                type="circle",
                children=dcc.Graph(id='tsne-plot')
            )
        )
    ], style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

# tsne callbacks
@app.callback(
        Output('tsne-plot','figure'),
        [Input('url', 'pathname'),],
        prevent_initial_call=True
)
def update_tsne_plot(pathname):
    if pathname == '/tsne':
        fig_tsne = px.scatter(tsne_data, x='t-SNE Component 1', y='t-SNE Component 2', color='Attack_Type',
                                title='t-SNE 2D Plot with Attack Types', opacity=0.7)
        return fig_tsne
    else:
        return dash.no_update

# Mahalanobis Distance Calculation
def calculate_mahalanobis_distances(data, attack_categories):
    centroids = {}
    for category in attack_categories:
        data_subset = data[data['attack_cat'] == category]
        centroid = data_subset.drop(['attack_cat', 'label'], axis=1).mean().values
        centroids[category] = centroid
    
    lw = LedoitWolf()
    covariance_matrix = lw.fit(data.drop(['attack_cat', 'label'], axis=1)).covariance_
    
    mahalanobis_distances = {}
    for category1 in attack_categories:
        for category2 in attack_categories:
            if category1 != category2:
                delta = centroids[category1] - centroids[category2]
                m = np.dot(np.dot(delta, np.linalg.inv(covariance_matrix)), delta)
                mahalanobis_distances[(category1, category2)] = np.sqrt(m)
    
    distance_matrix = pd.DataFrame(columns=attack_categories, index=attack_categories)
    for category1 in attack_categories:
        for category2 in attack_categories:
            if category1 == category2:
                distance_matrix.at[category1, category2] = 0.0
            else:
                distance_matrix.at[category1, category2] = mahalanobis_distances[(category1, category2)]
    distance_matrix = distance_matrix.astype(float)
    return distance_matrix

# Heatmap Generation
def generate_heatmap(data):
    scaling_techniques = ['no_scaling', 'min_max', 'standard', 'quantile', 'robust', 'power']  # Include 'no_scaling'

    assets_dir = os.path.join(os.getcwd(), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    heatmap_divs = []
    # Loop through scaling techniques
    for scaling_technique in scaling_techniques:
        # Check if the image file already exists
        img_filename = os.path.join(assets_dir, f"heatmap_{scaling_technique}.png")
        # img_filename = f"heatmap_{scaling_technique}.png"
        title = ""
        if os.path.exists(img_filename):
            with open(img_filename, 'rb') as img_file:
                img_src = 'data:image/png;base64,{}'.format(base64.b64encode(img_file.read()).decode())
        else:
            # If the image file doesn't exist, generate and save the heatmap
            if scaling_technique == 'no_scaling':
                scaled_train_df = data.copy()  # Create a copy of the original data without scaling
            else:
                if scaling_technique == 'min_max':
                    scaler = MinMaxScaler()
                elif scaling_technique == 'standard':
                    scaler = StandardScaler()
                elif scaling_technique == 'quantile':
                    scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')
                elif scaling_technique == 'robust':
                    scaler = RobustScaler()
                elif scaling_technique == 'power':
                    scaler = PowerTransformer()
                # Add conditions for other scaling techniques
            
                # Apply the respective scaler to the data
                features_to_normalize = data.drop(['attack_cat', 'label'], axis=1).columns
                scaled_train_df = data.copy()
                if scaling_technique != 'power':
                    scaled_train_df[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
                else:
                    scaled_train_df[features_to_normalize] = data[features_to_normalize] + 1e-6
                    scaled_train_df[features_to_normalize] = scaler.fit_transform(scaled_train_df[features_to_normalize])

            attack_categories = data['attack_cat'].unique()
            distance_matrix = calculate_mahalanobis_distances(scaled_train_df, attack_categories)
            
            mask = np.triu(np.ones(distance_matrix.shape), k=1)
            plt.figure(figsize=(10, 8))
            sns.heatmap(distance_matrix, cmap="RdYlGn", annot=True, fmt=".2f", linewidths=0.5, mask=mask)
            if scaling_technique == 'no_scaling':
                title = "Centroid Distance Heatmap without Scaling"
            else:
                title = "Centroid Distance Heatmap after " + scaling_technique + " scaling"
            plt.title(title)
            
            # Save the plot as a binary image
            plt.savefig(img_filename, format='png')
            
            # Create HTML img element with base64 encoded image
            with open(img_filename, 'rb') as img_file:
                img_src = 'data:image/png;base64,{}'.format(base64.b64encode(img_file.read()).decode())
            
            plt.close()  # Close the plot to prevent display in the Dash app
        
        heatmap_divs.append(
            html.Div([
                html.H3(title),
                html.Img(src=img_src, style={'width': '100%'})
            ])
        )

    return heatmap_divs

# heatmap callbacks
@app.callback(
    Output('heatmap-content', 'children'),
    # Output('loading-heatmap', 'children'),
    [Input('url', 'pathname'),],
    prevent_initial_call=True
)
def update_heatmap(pathname):
    if pathname == '/heatmap':
        heatmap_divs = generate_heatmap(train_df)
        heatmap_content = html.Div([
            html.H1("Mahalanobis Distance Heatmaps", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
            html.Div(heatmap_divs, style={'display': 'grid', 'grid-template-columns': 'repeat(3, 1fr)'})
        ])
        return heatmap_content
    else:
        return dash.no_update

# heatmap layout
def heatmap_layout():
    return html.Div(
        id='heatmap-content', style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

# Barplot Layout
def barplot_layout():
    return html.Div([
        html.H1("Attack Type Distribution", style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        dcc.Loading(
            id="loading-bar-plot",
            type="circle",
            children=[dcc.Graph(id='bar-plot')]
        )
    ], style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})


# Function to generate bar plot for attack type distribution
def generate_attack_type_barplot(data):
    attack_type_counts = data['attack_cat'].value_counts()
    fig_bar = px.bar(
        x=attack_type_counts.index, 
        y=attack_type_counts.values, 
        labels={'x': 'Attack Type', 'y': 'Count'}, 
        title='Distribution of Attack Types'
    )
    return fig_bar

# Barplot Callbacks
@app.callback(
    Output('bar-plot','figure'),
    [Input('url', 'pathname')],
    # Having this True is making the calls unpredictable having
    # it as False is making the calls predictable and no cache error
    prevent_initial_call=False
)
def update_bar_plot(pathname):
    if pathname == '/barplot':
        bar_plot = generate_attack_type_barplot(train_df)
        return bar_plot
    else:
        return dash.no_update

# Home Layout
def home_layout():
    return html.Div([
        html.H1("Welcome to the Home Page", style={'font-family': 'Arial, sans-serif', 'color': '#3366cc'}),
        html.P("Navigate to other pages using the links below:", style={'font-family': 'Arial, sans-serif'}),
        html.Br(),
        dcc.Link('Basic Visualization', href='/bv', style={'display': 'block', 'text-decoration': 'none', 'color': '#3366cc', 'margin-bottom': '10px'}),
        dcc.Link('Feature Selection', href='/fs', style={'display': 'block', 'text-decoration': 'none', 'color': '#3366cc', 'margin-bottom': '10px'}),
        dcc.Link('Class Imbalance Visualization', href='/civ', style={'display': 'block', 'text-decoration': 'none', 'color': '#3366cc', 'margin-bottom': '10px'}),
    ], style={'width': '60%', 'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})



def basic_visulation_layout():
    return html.Div([
        html.H1("Basic Visualization Page", style={'font-family': 'Arial, sans-serif', 'color': '#3366cc'}),
        html.P("Navigate to other pages using the links below:", style={'font-family': 'Arial, sans-serif'}),
        html.Br(),
        dcc.Link('Bar Plot Page', href='/barplot', style={'display': 'block', 'text-decoration': 'none', 'color': '#3366cc', 'margin-bottom': '10px'}),
        dcc.Link('Heatmap Page', href='/heatmap', style={'display': 'block', 'text-decoration': 'none', 'color': '#3366cc', 'margin-bottom': '10px'}),
    ], style={'width': '60%', 'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

def feature_selection_layout():
    return html.Div([
        html.H1("Feature Selection Page", style={'font-family': 'Arial, sans-serif', 'color': '#3366cc'}),
        html.P("Navigate to other pages using the links below:", style={'font-family': 'Arial, sans-serif'}),
        dcc.Link('Elastic Net Page', href='/elasticnet', style={'display': 'block', 'text-decoration': 'none', 'color': '#3366cc', 'margin-bottom': '10px'}),
        dcc.Link('Random Forest Page', href='/randomforest', style={'display': 'block', 'text-decoration': 'none', 'color': '#3366cc', 'margin-bottom': '10px'}),
    ], style={'width': '60%', 'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})


def class_imbalance_page_layout():
    return html.Div([
        html.H1("Class Imbalance Page", style={'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        html.P("Navigate to other pages using the links below:", style={'font-family': 'Arial, sans-serif', 'margin-bottom': '30px'}),
        dcc.Link('t-SNE Page', href='/tsne', style={'display': 'block', 'margin-bottom': '10px', 'color': '#3366cc', 'text-decoration': 'none', 'font-size': '18px'}),
        dcc.Link('PCA Page', href='/pca', style={'display': 'block', 'margin-bottom': '10px', 'color': '#3366cc', 'text-decoration': 'none', 'font-size': '18px'}),
        dcc.Link('LDA Page', href='/lda', style={'display': 'block', 'margin-bottom': '10px', 'color': '#3366cc', 'text-decoration': 'none', 'font-size': '18px'}),
        dcc.Link('KMeans Page', href='/kmeans', style={'display': 'block', 'margin-bottom': '10px', 'color': '#3366cc', 'text-decoration': 'none', 'font-size': '18px'}),
    ], style={'width': '60%', 'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})


# Elastic Net Regression
def generate_regularization_path(data):
    # Data sampling
    # data = data.sample(frac=0.1, random_state=42).

    # Data preprocessing
    print("Dropping correlated features...")
    numerical_features = data.select_dtypes(include=[np.number])
    correlation_matrix = numerical_features.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
    correlated_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    data_filtered = data.drop(correlated_features, axis=1)
    print(f"Correlated features: {correlated_features}")

    # Correlation matrix heatmap
    correlation_matrix = data.select_dtypes(include=[np.number]).corr().abs()
    fig_corr = px.imshow(correlation_matrix)
    fig_corr.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="Features",
        yaxis_title="Features",
    )

    X = data_filtered.drop(['attack_cat', 'label'], axis=1)
    y = data['label']
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    print("Scaling data...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    

    n_alphas = 100
    # l1_ratios = np.linspace(0.1, 1.0, 5)
    selected_l1_ratio = 0.5

    # as we are using l1_ratio = 0.5, we can use ElasticNetCV to find the best alpha
    # we should be getting the optimal alpha value as 0.000170306
    # the alpha values are not generated randomly instead based on a logarithamic spacing of log(alpha_min) to log(alpha_max)
    print("Searching for best alpha using elastic net...")
    elastic_net_cv = ElasticNetCV(l1_ratio=selected_l1_ratio, n_alphas=n_alphas, cv=5,random_state=42)
    elastic_net_cv.fit(X_train_scaled, y_train)

    print(f"Best alpha by elastic net: {elastic_net_cv.alpha_}")
    alphas = elastic_net_cv.alphas_
    best_alpha = elastic_net_cv.alpha_

    print(f"Alphas size: {alphas.shape}")

    print("Fitting model with best alpha...")
    elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=selected_l1_ratio, random_state=0)
    elastic_net.fit(X_train_scaled, y_train)
    
    # Evaluation metrics
    print("Evaluating model...")
    mse_values = elastic_net_cv.mse_path_.T
    mse_values_train = []
    mse_values_test = []
    r2_values_train = []
    r2_values_test = []

    # Calculating COEFS too
    coefs = []

    for alpha in tqdm(alphas, desc="Evaluation",leave=False):
        elastic_net_tmp = ElasticNet(alpha=alpha, l1_ratio=selected_l1_ratio, random_state=0)
        elastic_net_tmp.fit(X_train_scaled, y_train)
        y_pred_train = elastic_net_tmp.predict(X_train_scaled)
        y_pred_test = elastic_net_tmp.predict(X_test_scaled)
        mse_values_train.append(mean_squared_error(y_train, y_pred_train))
        mse_values_test.append(mean_squared_error(y_test, y_pred_test))
        r2_values_train.append(elastic_net_tmp.score(X_train_scaled, y_train))
        r2_values_test.append(elastic_net_tmp.score(X_test_scaled, y_test))
        coefs.append(elastic_net_tmp.coef_)
    

    # Printing Important Features
    print("Printing important features...")
    important_features = pd.Series(elastic_net.coef_, index=X.columns)

    # Important features are the ones with non-zero coefficients
    fig_imp_features = px.bar(
        x=important_features.index, 
        y=important_features.values, 
        labels={'x': 'Features', 'y': 'Importance'}, 
        title='Important Features categorized by their importance')

    important_features = important_features[important_features != 0]
    print(f"Important features: {important_features}")
    print(f"Lenght of important features: {len(important_features)}")

    # Initially used this to generate the coefs when we chose the l1 ratio as np.linspace(0.1, 1.0, 5)
    # but the size was literally (5,100) which is not what we want
    # for alpha_values in tqdm(alphas, desc="Alpha Values"):
    #     for alpha in tqdm(alpha_values, desc="Alphas", leave=False):
    #         elastic_net = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=0)
    #         elastic_net.fit(X_train_scaled, y_train)
    #         coefs.append(elastic_net.coef_)

    # as the size of alphas is (100,) so we just iterate through it only once

    print("Coefs has been computed")
    coefs = np.array(coefs)

    # Plotting
    # fig_mse = go.Figure()

    # Version 1
    # print("Plotting Mean squared error vs alpha...")
    # for each_array in alphas:
    #     for each_mse in mse_values:
    #         fig_mse = px.line(
    #             x=each_array,
    #             y=each_mse,
    #             title="Mean squared error vs alpha",
    #             labels={'x': 'Alpha', 'y': 'Mean squared error'}
    #         )

    # the size of mse_values is (100,)
    # if we choose l1 values as np.linspace(0.1, 1.0, 5) then the size of mse_values will be (5,100,5)
    # We transpose it for our convenience
    mse_values = elastic_net_cv.mse_path_.T

    print(f"MSE values shape is : {mse_values.shape}")
    print(f"Alphas shape is : {alphas.shape}")

    # Version 2
    print("Plotting Mean squared error vs alpha...")
    # fig_mse = go.Figure()
    # for i in range(len(mse_values)):
    #     fig_mse.add_trace(
    #         go.Scatter(
    #             x = alphas,
    #             y = mse_values[i],
    #             name = f"Fold {i+1}"
    #         )
    #     )
        
    # fig_mse.update_layout(
    #     title="Mean squared error vs alpha",
    #     xaxis_title="Alpha",
    #     yaxis_title="Mean squared error",
    # )
    fig_mse = go.Figure()
    fig_mse.add_trace(
        go.Scatter(
            x=alphas,
            y=mse_values_train,
            mode='lines+markers',
            name='Train MSE'
        )
    )
    fig_mse.add_trace(
        go.Scatter(
            x=alphas,
            y=mse_values_test,
            mode='lines+markers',
            name='Test MSE'
        )
    )
    fig_mse.update_layout(
        title="Mean Squared Error vs alpha",
        xaxis_title="Alpha",
        yaxis_title="Mean Squared Error",
    )

    # Plotting R Squared vs alpha
    print("Plotting R Squared vs Alpha")
    fig_r2 = go.Figure()
    fig_r2.add_trace(
        go.Scatter(
            x=alphas,
            y=r2_values_train,
            mode='lines+markers',
            name='Train R-squared'
        )
    )
    fig_r2.add_trace(
        go.Scatter(
            x=alphas,
            y=r2_values_test,
            mode='lines+markers',
            name='Test R-squared'
        )
    )
    fig_r2.update_layout(
        title="R-squared vs alpha",
        xaxis_title="Alpha",
        yaxis_title="R-squared",
    )

    # Plotting Coefficients vs alpha
    # Version 1
    print("Plotting Coefficients vs alpha...")
    fig_regularization_path = go.Figure()
    for i, feature in enumerate(X.columns):
        fig_regularization_path.add_trace(
            go.Scatter(
                x=-np.log10(alphas),
                y=coefs[:, i],
                name=feature
            )
        )

    fig_regularization_path.add_trace(
        go.Scatter(
            x = [best_alpha],
            y=np.linspace(min(np.min(coefs), 0), max(np.max(coefs), 0), 10),
            name= 'Best alpha',
            mode='lines',
            line=dict(color='black', dash='dash'),
        )
    )

    fig_regularization_path.update_layout(
        title="Coefficients vs alpha",
        xaxis_title="-log(Alpha)",
        yaxis_title="Coefficients",
    )

    # Version 2
    # fig_mse = go.Figure()
    # for i in range(len(alphas)):
    #     fig_mse.add_trace(
    #         go.Scatter(
    #             x=alphas[i],
    #             y=mse_values[i],
    #             name=f"Fold {i+1}"
    #         )
    #     )

    # print("Plotting Coefficients vs alpha...")
    # fig_regularization_path = go.Figure()
    # for i, feature in enumerate(X.columns):
    #     fig_regularization_path.add_trace(
    #         go.Scatter(
    #             x=alphas,
    #             y=coefs[:, i],
    #             name=feature
    #         )
    #     )
    
    # fig_regularization_path.add_trace(
    #     go.Scatter(
    #         x = [best_alpha],
    #         y=np.linspace(min(np.min(coefs), 0), max(np.max(coefs), 0), 10),
    #         name= 'Best alpha',
    #         mode='lines',
    #         line=dict(color='black', dash='dash'),
    #     )
    # )

    return fig_mse, fig_regularization_path, fig_r2, fig_imp_features
    
# Elastic Net Layout
def elastic_net_layout():
    return html.Div(children=[
        html.H1("Elastic Net Plots",style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        html.Div(
            dcc.Loading(
                id="loading-elastic-net",
                type="circle",
                children=[
                    dcc.Graph(id='elastic-net-mse-plot'),
                ]
            )
        ),

        html.Div(
            dcc.Loading(
                id="loading-regularization-path",
                type="circle",
                children=[
                    dcc.Graph(id='elastic-net-coef-plot'),
                ]
            )
        ),

        html.Div(
            dcc.Loading(
                id="loading-correration-matrix",
                type="circle",
                children=[
                    dcc.Graph(id='correration-matrix-plot'),
                ]
            )
        ),

        html.Div(
            dcc.Loading(
                id="loading-important-features",
                type="circle",
                children=[
                    dcc.Graph(id='important-features-plot'),
                ]
            )
        ),
    ], style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

# Regularization Path Callbacks
@app.callback(
    [Output('elastic-net-mse-plot', 'figure'),
     Output('elastic-net-coef-plot', 'figure'),
     Output('correration-matrix-plot', 'figure'),
     Output('important-features-plot', 'figure')],
    [Input('url', 'pathname')],
    prevent_initial_call=True
)
def update_regularization_path(pathname):
    if pathname == '/elasticnet':
        print("-- Entered here --")
        mse_plot,coef_plot,fig_corr,imp_features = generate_regularization_path(train_df)
        print("-- Exited here --")
        return mse_plot,coef_plot,fig_corr,imp_features
    else:
        return dash.no_update
    
# Random Forest Classifier
def generate_feature_importance_path(data):
    numerical_features = data.select_dtypes(include=[np.number])
    correlation_matrix = numerical_features.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
    correlated_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    data_filtered = data.drop(correlated_features, axis=1)
    X = data_filtered.drop(['attack_cat', 'label'], axis=1)
    y = data['label']
    y = LabelEncoder().fit_transform(y)
    # If we required we can perform necessary operations on X_test and y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    important_features = rf.feature_importances_
    print(f"Important features: {important_features}")
    print(f"Lenght of important features: {len(important_features)}")

    # Evaluation metrics
    # y_pred = rf.predict(X_train)
    # accuracy = accuracy_score(y, y_pred)
    # precision = precision_score(y, y_pred)
    # recall = recall_score(y, y_pred)
    # f1 = f1_score(y, y_pred)
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1: {f1}")


    fig_rf = go.Figure()
    fig_rf.add_trace(
        go.Bar(
            x=X.columns,
            y=important_features,
            name="Feature Importance"
        )
    )
    fig_rf.update_layout(
        title="Feature Importance based on Random Forest Classifier",
        xaxis_title="Features",
        yaxis_title="Importance",
    )

    return fig_rf

# Random Forest Layout
def random_forest_layout():
    return html.Div([
        html.H1("Random Forest Plots",style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        dcc.Loading(
            id="loading-random-forest",
            type="circle",
            children=[dcc.Graph(id='random-forest-plot')],
        )
    ], style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

# Random Forest callbacks
@app.callback(
    Output('random-forest-plot','figure'),
    [Input('url', 'pathname')],
    prevent_initial_call=True
)
def update_random_forest_plot(pathname):
    if pathname == '/randomforest':
        rf_plot = generate_feature_importance_path(train_df)
        return rf_plot
    else:
        return dash.no_update
    
# KMeans Clustering Layout
def kmeans_layout():
    return html.Div([
        html.H1("K-Means Plots",style={'textAlign': 'center', 'font-family': 'Arial, sans-serif', 'color': '#3366cc', 'margin-bottom': '20px'}),
        dcc.Loading(
            id="loading-kmeans",
            type="circle",
            children=[
                dcc.Graph(id='kmeans-plot'),
            ]
        ),
    
    ], style={'margin': 'auto', 'background-color': '#f8f8f8', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)'})

# KMeans Clustering Callbacks
@app.callback(
    Output('kmeans-plot', 'figure'),
    [Input('url', 'pathname')],
    prevent_initial_call=True
)
def update_kmeans_plot(pathname):
    if pathname == '/kmeans':
        X = apply_pca(train_df.copy(), 2)
        labels = train_df['attack_cat']

        shrunken_centroids = NearestCentroid(shrink_threshold=None)
        shrunken_centroids.fit(X, labels)
        class_centroids = shrunken_centroids.centroids_
        kmeans = KMeans(n_clusters=len(class_centroids), init=class_centroids, n_init=1, max_iter=1)
        kmeans.fit(X)

        cluster_centers = kmeans.cluster_centers_
        # cluster_labels = kmeans.labels_
        print(f"Cluster centers: {cluster_centers}")
        # intercluster_distances = euclidean_distances(cluster_centers, cluster_centers)
        cluster_sizes = np.bincount(kmeans.labels_)
        print(f"Cluster sizes: {cluster_sizes}")

        fig = go.Figure()
        for i,(x,y) in enumerate(cluster_centers[:, :2]):
            center = go.Scatter(
                x=[x],
                y=[y],
                mode='text',
                text=[str(i)],
                marker=dict(size=5, color='rgba(255,0,0,1)'),
                showlegend=False,
            )
            fig.add_trace(center)

        # We are using the cluster size to plot the circle around the center
        for i, (x, y) in enumerate(cluster_centers[:, :2]):
            circle = go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(size=np.sqrt(cluster_sizes[i]), color='rgba(155,155,223,0.56)'),
                name=f'Cluster {i}',
            )
            fig.add_trace(circle)

        # Update layout for 2D plot
        fig.update_layout(
            title='K-means Cluster Analysis: Size, Spread, and Overlap',
            xaxis=dict(title='First Dimension',showgrid=False, showticklabels=False),
            yaxis=dict(title='Second dimension', showticklabels=False, showgrid=False),
            height = 800,
        )

        return fig

    else:
        return dash.no_update
    


def run_dash_app():
    app.run_server(debug=True)

if __name__ == '__main__':
    run_dash_app()