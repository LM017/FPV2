from vetiver import VetiverModel
from vetiver import vetiver_pin_write
from vetiver import VetiverAPI
import pins
from palmerpenguins import penguins
from pandas import get_dummies
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

df = penguins.load_penguins().dropna()

X = get_dummies(df[['bill_length_mm', 'species', 'sex']], drop_first = True)
y = df['body_mass_g']

model = LinearRegression().fit(X, y)
v = VetiverModel(model, model_name = 'penguin_model', prototype_data = X)
b = pins.board_folder('data/model', allow_pickle_read=True)
vetiver_pin_write(b, v)
v = VetiverModel.from_pin(b, 'penguin_model')
app = VetiverAPI(v, check_prototype=True)

app.run(port = 8080)
