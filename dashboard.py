import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.express as px

st.set_page_config(page_title='Project 7 Dashboard"', page_icon=':smiley', 
                   layout="wide", initial_sidebar_state='expanded')

sns.set_theme(style="white")

def count_plot(df):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.countplot(data=df, x='TARGET')
    return fig

def distribution(df, column_name, client_value=0, hue=None):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.kdeplot(data=df, x=column_name,  fill=True, hue=hue)
    plt.axvline(client_value, color='red')
    return fig

def boxplot(df, column_name, client_value=0):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.boxplot(y=column_name, data=df, x='TARGET', showfliers=False)
    plt.axhline(client_value, color='red')
    return fig

def scatter(df, column_name1, column_name2):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.scatterplot(data=df, x=column_name1, y=column_name2, hue="TARGET")
    return fig

# def barplot(df):
#     fig = sns.barplot(data=df, y='')

def correlation_matrix(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(5,5))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return fig

def cf_matrix(df, pred_df):
    y_pred = (pred_df.yes > 0.5) * 1
    y_true = df.TARGET
    cf_mat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cf_mat, annot=True)
    return fig


def feature_importance(df_fi):
    fig = px.bar(df_fi, x="feature", y="importance", barmode="group")
    return fig



def main():
    st.title("Project 7 Dashboard")

    st.header("Cleaned Data:")
    df = pd.read_csv("cleaned_data.csv")
    df = df.set_index('SK_ID_CURR')
    #df = df.drop('Unnamed: 0', 1)
    st.dataframe(df)

    st.header("Target repartition:")
    fig0 = count_plot(df)
    st.plotly_chart(fig0, width=0.5, height=0.5)

    
    st.header("Distributions:")
    choice = st.selectbox("Column name", df.columns)
    idx = st.number_input('Client idx', value=100002)
    client_value = df.loc[idx, choice]

    left_column, right_column = st.columns(2)
    with left_column:
        st.title('Full distribution')
        fig1 = distribution(df, choice, client_value)
        st.pyplot(fig1, width=0.5, height=0.5)
    with right_column:
        st.title('Box plot')
        fig4 = boxplot(df, choice, client_value)
        st.pyplot(fig4)

    with left_column:
        st.title('Accepted distribution')
        fig2 = distribution(df[df.TARGET == 1], choice, client_value)
        st.pyplot(fig2)

    with right_column:  
        st.title('Not accepted distribution')
        fig3 = distribution(df[df.TARGET == 0], choice, client_value)
        st.pyplot(fig3)
    
    with left_column:
        if st.checkbox('Show correlation matrix'):
            fig5 = correlation_matrix(df)
            st.pyplot(fig5)

            
    with right_column:
        if st.checkbox('Show bivariate analysis'):
            c1 = st.selectbox("Column name 1", df.columns)
            c2 = st.selectbox("Column name 2", df.columns)
            fig8 = scatter(df.sample(n=1000), c1, c2)
            st.pyplot(fig8)

    with right_column:
        if st.checkbox('Show confusion matrix'):
            pred_df = pd.read_csv('predictions.csv')
            fig6 = cf_matrix(df, pred_df)
            st.pyplot(fig6)

    with left_column:
        if st.checkbox('Feature importance'):
            df_fi = pd.read_csv('features_importance.csv')
            fig7 = feature_importance(df_fi)
            st.plotly_chart(fig7)
        
    st.write("Model predictions: [link](https://huggingface.co/spaces/Smouse/test_gradio)")

if __name__ == '__main__':
	main()