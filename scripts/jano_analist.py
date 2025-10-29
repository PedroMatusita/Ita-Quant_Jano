import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from jano_pair_selector import buscar_par_com_llm_gemini
from google import genai
from dotenv import load_dotenv
from google.genai import types


# -------------------------------

# Função para validar cointegração do par

# -------------------------------


def validar_par_estatisticamente(ticker1, ticker2):
    """
    Verifica se o par é cointegrado e retorna o spread 
    pronto para o backtest.
    """
    print(f"Analisando cointegração para {ticker1} e {ticker2}...")

    startL = datetime.now() + timedelta(days = -365*5)
    endL = datetime.now() + timedelta(days = -365)
    
    dados = yf.download([ticker1, ticker2], start=startL, end=endL)['Close'].dropna()
    Y = dados[ticker1]
    X = dados[ticker2]
    
    X_com_constante = sm.add_constant(X)
    modelo = sm.OLS(Y, X_com_constante).fit()
    
    beta = modelo.params[1]
    print(f"Hedge Ratio (Beta) encontrado: {beta:.4f}")
    
    spread = Y - beta * X
    
    teste_adf = adfuller(spread, autolag='AIC')
    p_value = teste_adf[1]
    
    print(f"Teste ADF P-Value: {p_value:.4f}")
    
    if p_value < 0.05: 
        print("Resultado: O PAR É COINTEGRADO (p-value < 0.05).")
        print("Pode prosseguir para o backtest.")
        return 1
        # media_spread = spread.rolling(window=30).mean()
        # desvio_spread = spread.rolling(window=30).std()
        
        # df_backtest = pd.DataFrame({'Spread': spread})
        # df_backtest['ZScore'] = (spread - media_spread) / desvio_spread
        # df_backtest.dropna(inplace=True)
        
        # return df_backtest
        
    else:
        print("Resultado: O PAR NÃO É COINTEGRADO (p-value >= 0.05).")
        print("Rejeitar este par. Tente outros tickers.")
        return None

    
# --- Exemplo de uso ---
if __name__ == "__main__":
    # ticker1 = "PETR4.SA"
    # ticker2 = "PETR3.SA"
    
    # 1. Configuração do Sistema (System Instruction)
    prompt_sistema = """
    Você é um analista financeiro sênior especializado em estratégias
    market-neutral na B3 (bolsa brasileira).
    """

    # 2. Schema de Saída (JSON Schema)
    # Define o formato exato que a LLM deve retornar
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "par": types.Schema(
                type=types.Type.ARRAY,
                description="Uma lista de dois tickers de ações (incluindo o sufixo .SA).",
                items=types.Schema(type=types.Type.STRING)
            ),
            "setor": types.Schema(
                type=types.Type.STRING,
                description="O nome do setor ao qual o par pertence."
            ),
            "justificativa": types.Schema(
                type=types.Type.STRING,
                description="Uma breve explicação sobre a escolha do par (correlação, setor, etc.)."
            )
        },
        required=["par", "setor", "justificativa"]
    )

    # df_estrategia = validar_par_estatisticamente(ticker1, ticker2)
    load_dotenv() 
    client = genai.Client()
    chat = client.chats.create(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            system_instruction=prompt_sistema,
            # AVISO: Veja a seção de atenção sobre o JSON abaixo.
            # Por enquanto, vamos desativar a força de JSON em um chat livre:
            response_mime_type="application/json", 
            response_schema=response_schema
        )
    )
    rerun = False

    for i in range(10):
        sugestao_llm = buscar_par_com_llm_gemini(chat, rerun)
    
        if sugestao_llm and 'par' in sugestao_llm and len(sugestao_llm['par']) == 2:
            ticker1, ticker2 = sugestao_llm['par']
            print(f"\nSugestão recebida da LLM:")
            print(f"Par: {ticker1} e {ticker2}")
            resultado = validar_par_estatisticamente(ticker1, ticker2)
            if resultado is None:
                rerun = True
            else:
                break
    
    print("PAR COINTEGRADO - SEGUIR PARA BACKTESTING")


    # if df_estrategia is not None:
    #     print("\n--- Dados da Estratégia (Z-Score) ---")
    #     print(df_estrategia.tail())