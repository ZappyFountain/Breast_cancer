import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from array import array

bc = pd.read_csv("wisconsin_breast_cancer.csv")
bc=bc.dropna(how='any')
bc.to_csv("cancer_edited.csv")
X = bc[["thickness", "size", "shape", "adhesion", "single", "nuclei", "chromatin", "nucleoli", "mitosis"]]
y = bc["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
model = LogisticRegression()
model.fit(X_train, y_train)

def breast_cancer_prediction(thickness, size, shape, adhesion, single, nuclei, chromatin, nucleoli, mitosis):
    inp_variables = [[thickness, size, shape, adhesion, single, nuclei, chromatin, nucleoli, mitosis]]
    inp_array = np.array(inp_variables)
    prediction = model.predict(inp_array)
    if prediction  == 0:
        output = "No Breast Cancer present in prediction"
    if prediction == 1:
        output =  "Breast Cancer present in prediction"
    return output

thickness1= gr.Number(label = "Thickness")
size1 = gr.Number(label = "Size")
shape1= gr.Number(label = "Shape")
adhesion1= gr.Number(label = "Adhesion")
single1= gr.Number(label = "Single")
nuclei1= gr.Number(label = "Nuclei")
chromatin1= gr.Number(label = "Chromatin")
nucleoli1= gr.Number(label = "Nucleoli")
mitosis1= gr.Number(label = "Mitosis")

demo = gr.Interface(fn=breast_cancer_prediction, 
                     inputs= [thickness1, size1, shape1, adhesion1, single1, nuclei1, chromatin1, nucleoli1, mitosis1],
                     outputs="text")
demo.launch()  
