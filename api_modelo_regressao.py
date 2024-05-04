from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instancia do FastAPI
app = FastAPI()

# Criar uma classe que tera is dadis do request body da api
class request_body(BaseModel):
  horas_estudo: float

# Carregar o modelo para realizar a predicao
modelo_pontuacao = joblib.load('./modelo_regressao.pkl')

@app.post('/predict')
def predict(data : request_body):
  # Prepara os dados para pedicao
  input_feature = [[data.horas_estudo]]

  # Realiza a predicao
  y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

  return {'pontuacao': y_pred.tolist()}
