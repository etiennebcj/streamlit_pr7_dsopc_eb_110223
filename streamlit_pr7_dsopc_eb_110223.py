import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import shap
from sklearn.cluster import KMeans
from zipfile import ZipFile
from lightgbm import LGBMClassifier
from function_pca import pca_maker
from sklearn.decomposition import PCA
# import plotly.graph_objects as go
# plt.style.use('fivethirtyeight')


@st.cache
def load_data():
	
    z = ZipFile('train_sample_30mskoriginal.zip') 
    data = pd.read_csv(z.open('train_sample_30mskoriginal.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    data = data.drop('Unnamed: 0', axis=1)
    z = ZipFile('train_sample_30m.zip')
    sample = pd.read_csv(z.open('train_sample_30m.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    X_sample = sample.iloc[:, :-1]
    
    description = pd.read_csv('HomeCredit_columns_description.csv', 
    				usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')
    				
    target = data[['TARGET']] # target = data.iloc[:, -1:]
    
    return data, sample, X_sample, target, description


def load_model():
        '''loading the trained model'''
        pickle_in = open('LGBMClassifier_best_customscore.pkl', 'rb') 
        model = pickle.load(pickle_in)
        return model


@st.cache(allow_output_mutation=True)
def load_knn(sample):
    knn = knn_training(sample)
    return knn


@st.cache
def load_gen_info(data):
    list_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]


    nb_credits = list_infos[0]
    mean_revenue = list_infos[1]
    mean_credits = list_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, mean_revenue, mean_credits, targets
    
    
def client_identity(data, id):
    data_client = data[data.index == int(id)]
    return data_client    


@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]/365), 2)
    return data_age


@st.cache
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income


@st.cache
def load_prediction(sample, id, model):
    X=sample.iloc[:, :-1]
    score = model.predict_proba(X[X.index == int(id)])[:,1]
    return score


@st.cache
def knn_training(sample):
    knn = KMeans(n_clusters=2, random_state=7).fit(sample)
    return knn 
    

@st.cache
def load_kmeans(sample, id, model):
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(X_sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:,1:].sample(10)


@st.cache
def load_pca_proj(sample, id, model):
    index_proj = sample[sample.index == int(id)].index.values
    index_proj = index_proj[0]
    data_client_proj = pd.DataFrame(X_sample.loc[sample.index, :])
    knn1 = knn_training(data_client_proj)
    df_neighbors_proj = pd.DataFrame(knn1.fit_predict(data_client_proj), index=data_client_proj.index)
    # df_neighbors_proj = pd.concat([df_neighbors_proj, sample.iloc[:, :-1]], axis=1)
    df_neighbors_proj = pd.concat([df_neighbors_proj, sample], axis=1)
    return df_neighbors_proj.iloc[:,1:].sample(10)


# @st.cache
def perform_pca(data_import):
    pca, pca_data, pca_cols, num_data = pca_maker(data_import)
    pca_1 = pca_cols[0]
    pca_2 = pca_cols[1]
    pca_data['TARGET'] = data_import['TARGET']
    pca_data['TARGET'] = pca_data['TARGET'].astype(str)
    fig = px.scatter(data_frame=pca_data, x=pca_1, y=pca_2, template="simple_white", color='TARGET', width=500, height=500)
    fig.update_traces(marker_size=10)
    return scatter_column.plotly_chart(fig)



# Global feature importance
@st.cache
def get_model_varimportance(model, train_columns, max_vars=10):
    var_imp_df = pd.DataFrame([train_columns, model.feature_importances_]).T
    var_imp_df.columns = ['feature_name', 'var_importance']
    var_imp_df.sort_values(by='var_importance', ascending=False, inplace=True)
    var_imp_df = var_imp_df.iloc[0:max_vars] 
    return var_imp_df


# Loading data
data, sample, X_sample, target, description = load_data()
id_client = sample.index.values
model = load_model()




#******************************************
# MAIN
#******************************************

# Title display
html_temp = """
<div style="background-color: LightSeaGreen; padding:5px; border-radius:10px">
	<h1 style="color: white; text-align:center">Credit Allocation Dashboard</h1>
</div>
    """
st.markdown(html_temp, unsafe_allow_html=True)




#*******************************************
# Displaying informations on the sidebar
#*******************************************

# Loading selectbox
# st.sidebar.header('Pick client ID')
# chk_id = st.sidebar.selectbox('', id_client)
chk_id = st.sidebar.selectbox('Pick client ID', id_client)

# Loading general informations
nb_credits, mean_revenue, mean_credits, targets = load_gen_info(data)

# Number of loans for clients in study
st.sidebar.markdown("<u>Total number of loans in our sample :</u>", unsafe_allow_html=True)
st.sidebar.text(nb_credits)

# Average income
st.sidebar.markdown("<u>Average income ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(int(mean_revenue))

# AMT CREDIT
st.sidebar.markdown("<u>Average loan amount ($US) :</u>", unsafe_allow_html=True)
st.sidebar.text(int(mean_credits))

# Labels explanation
st.sidebar.markdown("<u>Labels explanation :</u>", unsafe_allow_html=True)
st.sidebar.text('TARGET 1 : "Defaulted"')
st.sidebar.text('TARGET 0 : "Reimbursed"')





#******************************************
# MAIN -- suite
#******************************************

'''-------------------------------------------------------------------------------------------------'''



# Customer prediction display
prediction = load_prediction(sample, chk_id, model)
st.header("Default probability : {:.0f} %".format(round(float(prediction)*100, 2)))

infos_client = client_identity(data, chk_id)
st.write('Client label :', infos_client['TARGET'])
'''-------------------------------------------------------------------------------------------------'''



# Gauge chart
fig = go.Figure(go.Indicator(mode = "gauge+number+delta",
                             value = round(float(prediction)*100, 0),
                             domain = {'x': [0, 1], 'y': [0, 1]},
                             title = {'text':"Default probability"},
                             delta = {'reference': 25},
                             gauge = {'axis': {'range': [None, 50]},
                                      'steps' : [{'range': [0, 10], 'color': "green"},
                                      		 {'range': [10, 20], 'color': "yellow"},
                                                 {'range': [20, 30], 'color': "darkorange"},
                                                 {'range': [30, 40], 'color': "red"},
                                                 {'range': [40, 50], 'color': "firebrick"}]}))
st.plotly_chart(fig)
#-------------------------#

# Gauge chart 2
# plot_bgcolor = "#def"
plot_bgcolor = "white"

quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"] 
quadrant_text = ["", "<b>Very high</b>", "<b>High</b>", "<b>Medium</b>", "<b>Low</b>", "<b>Very low</b>"]
n_quadrants = len(quadrant_colors) - 1

current_value = round(float(prediction)*100, 0)
min_value = 0
max_value = 50
# hand_length = np.sqrt(2) / 4
hand_length = np.sqrt(2) / 4.7
hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

fig = go.Figure(
        data=[
          go.Pie(
            values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
            rotation=90,
            hole=0.6,
            marker_colors=quadrant_colors,
            text=quadrant_text,
            textinfo="text",
            hoverinfo="skip")],
        layout=
          go.Layout(
            showlegend=False,
            margin=dict(b=0,t=10,l=10,r=10),
            width=700,
            height=600,
            paper_bgcolor=plot_bgcolor,
            annotations=[
              go.layout.Annotation(
                text=f"<b>Default probability :</b><br>{current_value} %",
                x=0.5, xanchor="center", xref="paper",
                y=0.25, yanchor="bottom", yref="paper",
                showarrow=False)],
            shapes=[
              go.layout.Shape(
                type="circle",
                x0=0.48, x1=0.52,
                y0=0.48, y1=0.52,
                fillcolor="#333",
                line_color="#333"),
              go.layout.Shape(
                type="line",
                x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                line=dict(color="#333", width=4))]))
                
st.plotly_chart(fig)                           
'''-------------------------------------------------------------------------------------------------'''



# Displaying customer information : gender, age, family status, Nb of hildren etc.
st.subheader('Customer general informations')
# Display Customer ID from Sidebar
st.write('Customer selected :', chk_id)

# Age informations
# infos_client = client_identity(data, chk_id)
st.write("Gender : ", infos_client["CODE_GENDER"].values[0])
st.write("Age : {:.0f} years old".format(int(infos_client["DAYS_BIRTH"]/365)))
st.write("Family status : ", infos_client["NAME_FAMILY_STATUS"].values[0])
st.write("Number of children : {:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
'''-------------------------------------------------------------------------------------------------'''



# Financial informations   
st.subheader("Customer financial informations ($US)")
st.write("Income total : {:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
st.write("Credit amount : {:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
st.write("Credit annuities : {:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
st.write("Amount of property for credit : {:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
'''-------------------------------------------------------------------------------------------------'''
   
   
   
# Feature importance
st.subheader("Customer report")
if st.checkbox("Show (Hide) customer #{:.0f} feature importance".format(chk_id)):
   st.markdown("<h5 style='text-align: center;'>Customer feature importance</h5>", unsafe_allow_html=True)
   shap.initjs()
   X = sample.iloc[:, :-1] # X = sample.loc[:, sample.columns != 'TARGET']
   X = X[X.index == chk_id]
   number = st.slider("Chose number of features up to 10", 0, 10, 5)

   fig, ax = plt.subplots(figsize=(7,7))
   explainer = shap.TreeExplainer(load_model())
   shap_values = explainer.shap_values(X)
   shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(3, 3))
   st.pyplot(fig)

else:
   st.markdown("<i>…</i>", unsafe_allow_html=True)
'''-------------------------------------------------------------------------------------------------'''


# Global feature importance
if st.checkbox("Show(Hide) global feature imortance") :
      st.markdown("<h5 style='text-align: center;'>Global feature importance</h5>", unsafe_allow_html=True)
      feature_importance = get_model_varimportance(model, sample.iloc[:, :-1].columns) # sample.columns
      fig = px.bar(feature_importance, x='var_importance', y='feature_name', orientation='h')
      st.plotly_chart(fig)
      
      # Customer data
      st.markdown("<u>Customer data with most global important features</u>", unsafe_allow_html=True)
      st.write(client_identity(data[feature_importance.feature_name], chk_id))
      
      # Feature description
      st.markdown("<u>Feature description</u>", unsafe_allow_html=True)
      list_features = description.index.to_list()
      feature = st.selectbox('You can type first letters of the feature for proposition', list_features)
      st.table(description.loc[description.index == feature][:1]) 

else:
   st.markdown("<i>…</i>", unsafe_allow_html=True)
'''-------------------------------------------------------------------------------------------------'''
   
   
   
# Similar customer projections
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

st.subheader('Projection of similar customers')
# st.markdown("<u>Projection of similar customers to the one selected</u>", unsafe_allow_html=True)
X_proj = load_pca_proj(sample, chk_id, model)
scatter_column, settings_column = st.columns((309, 1))
perform_pca(X_proj)
'''-------------------------------------------------------------------------------------------------'''



# Similar customer to the one selected by KMeans
neighbors_nearest = st.checkbox("Show (Hide) similar customers")

if neighbors_nearest:
   knn = load_knn(sample)
   st.markdown("<u>10 closest customers to the selected one</u>", unsafe_allow_html=True)
   st.dataframe(load_kmeans(sample, chk_id, knn))
   st.markdown("<i>Target 1 = Customer high default probability</i>", unsafe_allow_html=True)
   
else:
   st.markdown("<i>…</i>", unsafe_allow_html=True)
'''-------------------------------------------------------------------------------------------------'''



# Distribution plots : age & income 

# import plotly.figure_factory as ff --> NOK, only works with 2 or more columns used for dist pl0t
#if st.checkbox("Enable (Disable) showing disribution plots"):
   #st.markdown("<h5 style='text-align: center;'>Customer age</h5>", unsafe_allow_html=True)
   #data_age = load_age_population(data)
   #data_age = pd.DataFrame(data_age)
   #data_age_labels = pd.concat([data_age, data[['TARGET']]], axis=1)
   # fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
   # fig = ff.create_distplot([data_age_labels[c] for c in data_age_labels.columns], data_age_labels.columns, show_hist=False)
   # fig = ff.create_distplot(data_age_labels[['DAYS_BIRTH']], data_age_labels.TARGET, show_hist=False)
   #fig.add_vline(x=int(infos_client["DAYS_BIRTH"].values / 365), line_width=5, line_dash="dash", line_color="orange")
   #st.plotly_chart(fig)'''
  
# Age distribution plot --> OK
if st.checkbox("Enable (Disable) showing disribution plots"):
   st.subheader('Distribution plots')
   data_age = load_age_population(data)
   data_age = pd.DataFrame(data_age)
   data_age_labels = pd.concat([data_age, data[['TARGET']]], axis=1)
   st.markdown("<h5 style='text-align: center;'>Customer age</h5>", unsafe_allow_html=True)
   # fig = px.histogram(data_age_labels, color='TARGET'), default nbins=100 
   # by default : [fig = px.histogram(data_age_labels, color='TARGET', nbins=100)
   fig = px.histogram(data_age_labels, color='TARGET', nbins=50)
   fig.update_layout(xaxis_title="Age (Years)", yaxis_title="Count") # by default : xaxis_title='value', yaxis_title='count'
   fig.add_vline(x=int(infos_client["DAYS_BIRTH"].values / 365), line_width=5, line_dash="dash", line_color="orange")
   st.plotly_chart(fig)
   
   # Income distribution plot
   st.markdown("<h5 style='text-align: center;'>Customer income</h5>", unsafe_allow_html=True)
   data_income = load_income_population(data)
   data_income = pd.DataFrame(data_income)
   data_income = pd.concat([data_income, data[['TARGET']]], axis=1)
   fig = px.histogram(data_income, color='TARGET', nbins=15) 
   fig.update_layout(xaxis_title="Income ($US)", yaxis_title="Count")
   fig.add_vline(x=int(infos_client["AMT_INCOME_TOTAL"].values[0]), line_width=5, line_dash="dash", line_color="orange")
   st.plotly_chart(fig)
'''-------------------------------------------------------------------------------------------------''' 


   
# PieChart Defaulted/Reimbursed
if st.checkbox("Enable (Disable) customer repartition"):
   st.subheader('Repartition of customers by labels')
   st.markdown("<h5 style='text-align: center;'>0 -- reimbursed | 1 -- defaulted</h5>", unsafe_allow_html=True)
   # fig = px.pie(data, names='TARGET', color_discrete_sequence=px.colors.sequential.RdBu)
   # fig = px.pie(data, names='TARGET', title='Customers : 0-defaulted / 1-reimbursed')
   fig = px.pie(data, names='TARGET') #, color_discrete_sequence=px.colors.sequential.RdBu)
   st.plotly_chart(fig)




























