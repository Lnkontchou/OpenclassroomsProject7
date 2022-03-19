import gradio as gr
import pandas as pd
import pickle

df = pd.read_csv("cleaned_data.csv")
df = df.set_index('SK_ID_CURR')
df = df.drop(['TARGET'], 1)
model = pickle.load(open('final_model.pkl', 'rb'))
def predict(id):
    pred = model.predict_proba(df.iloc[[id]])[0]
    return ({"Accept": float(pred[0]), "Deny": float(pred[1])})


iface = gr.Interface(fn=predict, inputs=gr.inputs.Number(label='ID du client:'), outputs=gr.outputs.Label(label="Loan Score"))
iface.launch()