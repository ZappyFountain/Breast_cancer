import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def breast_cancer_prediction(thickness, size, shape, adhesion, single, nuclei, chromatin, nucleoli, mitosis):
  bc = pd.read_csv("wisconsin_breast_cancer.csv")
  bc=bc.dropna(how='any')
  bc.to_csv("cancer_edited.csv")
  X = bc[["thickness", "size", "shape", "adhesion", "single", "nuclei", "chromatin", "nucleoli", "mitosis"]]
  y = bc["class"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
  model = LogisticRegression()
  model.fit(X_train, y_train)
  prediction = model.predict("thickness", "size", "shape", "adhesion", "single", "nuclei", "chromatin", "nucleoli", "mitosis")
  if prediction  == 0:
    output = "No Breast Cancer present in prediction"
  if prediction == 1:
    output =  "Breast Cancer present in prediction"
return output

with gr.Blocks() as demo:

  fn = predict
  thickness1= gr.number(label = "Thickness")
  size1 = gr.number(label = "Size")
  shape1= gr.number(label = "Shape")
  adhesion1= gr.number(label = "Adhesion")
  single1= gr.number(label = "Single")
  nuclei1= gr.number(label = "Nuclei")
  chromatin1= gr.number(label = "Chromatin")
  nucleoli1= gr.number(label = "Nucleoli")
  mitosis1= gr.number(label = "Mitosis")
  output = predict
demo.launch()
  
