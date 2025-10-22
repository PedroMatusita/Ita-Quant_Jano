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
    
    # 1. Baixar dados
    dados = yf.download([ticker1, ticker2], period=periodo)['Close'].dropna()
    Y = dados[ticker1]
    X = dados[ticker2]
    
    # 2. Preparar dados para regressão (achar o "beta" da coleira)
    X_com_constante = sm.add_constant(X)
    modelo = sm.OLS(Y, X_com_constante).fit()
    
    # O "beta" (hedge ratio) é o coeficiente de X
    beta = modelo.params[1]
    print(f"Hedge Ratio (Beta) encontrado: {beta:.4f}")
    
    # 3. Calcular o Spread (A "coleira")
    # Spread = TickerA - (Beta * TickerB)
    spread = Y - beta * X
    
    # 4. Testar se o Spread é Estacionário (se ele volta à média)
    # Usamos o Teste "Augmented Dickey-Fuller" (ADF)
    # Hipótese Nula (p-value alto): O spread NÃO é estacionário.
    # Hipótese Alternativa (p-value baixo): O spread É estacionário.
    
    teste_adf = adfuller(spread, autolag='AIC')
    p_value = teste_adf[1]
    
    print(f"Teste ADF P-Value: {p_value:.4f}")
    
    if p_value < 0.05: # Nível de significância de 5%
        print("Resultado: O PAR É COINTEGRADO (p-value < 0.05).")
        print("Pode prosseguir para o backtest.")
        
        # 5. Calcular o Z-Score do Spread
        media_spread = spread.rolling(window=30).mean()
        desvio_spread = spread.rolling(window=30).std()
        
        # DataFrame final para o backtest
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
    # Exemplo de um par historicamente cointegrado no Brasil (Bancos)
    ticker1 = "ITUB4.SA"
    ticker2 = "BBDC4.SA"
    
    df_estrategia = validar_par_estatisticamente(ticker1, ticker2)
    
    if df_estrategia is not None:
        print("\n--- Dados da Estratégia (Z-Score) ---")
        print(df_estrategia.tail())