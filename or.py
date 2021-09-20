from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, modelname, plotname):

    df = pd.DataFrame(data)

    print(df)

    X, y = prepare_data(df)

    model_OR = Perceptron(eta=eta, epochs=epochs)
    model_OR.fit(X, y)

    _ = model_OR.total_loss()

    save_model(model_OR, filename=modelname)
    save_plot(df, file_name=plotname, model=model_OR)

if __name__ == '__main__':

    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1]
    }
    ETA = 0.3  # learning rate
    EPOCHS = 10

    main(data=OR, eta=ETA, epochs=EPOCHS, modelname="or.model", plotname="or.png")
