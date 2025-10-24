import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# -------------------------------

# Função para validar cointegração do par

# -------------------------------


def validar_par_estatisticamente(ticker1, ticker2, periodo="5y"):
    """
    Verifica se o par é cointegrado e retorna o spread 
    pronto para o backtest.
    """
    print(f"Analisando cointegração para {ticker1} e {ticker2}...")
    
    dados = yf.download([ticker1, ticker2], period=periodo)['Close'].dropna()
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
        
        media_spread = spread.rolling(window=30).mean()
        desvio_spread = spread.rolling(window=30).std()
        
        df_backtest = pd.DataFrame({'Spread': spread})
        df_backtest['ZScore'] = (spread - media_spread) / desvio_spread
        df_backtest.dropna(inplace=True)
        
        return df_backtest
        
    else:
        print("Resultado: O PAR NÃO É COINTEGRADO (p-value >= 0.05).")
        print("Rejeitar este par. Tente outros tickers.")
        return None

    
# --- Exemplo de uso ---
if __name__ == "__main__":
    ticker1 = "ITUB4.SA"
    ticker2 = "BBDC4.SA"
    
    df_estrategia = validar_par_estatisticamente(ticker1, ticker2)
    
    if df_estrategia is not None:
        print("\n--- Dados da Estratégia (Z-Score) ---")
        print(df_estrategia.tail())