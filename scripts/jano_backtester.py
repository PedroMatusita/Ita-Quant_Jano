from backtesting import Backtest, Strategy
import pandas as pd
from jano_analist import validar_par_estatisticamente

# -------------------------------

# Backtester para Estratégia de Pair Trading baseada em Cointegração

# -------------------------------

class EstrategiaJanoPairs(Strategy):
    limite_entrada = 2.0  
    limite_saida = 0.5    

    def init(self):
        self.zscore = self.I(lambda: self.data.ZScore, name="ZScore")

    def next(self):
        z = self.zscore[-1] 

        # --- Lógica de VENDA do Spread (Short) ---
        # Se o Z-Score está muito alto (acima de 2.0) e não estamos vendidos
        if z > self.limite_entrada and not self.position.is_short:
            # "Vende" o spread (Vende Ticker A, Compra Ticker B)
            self.sell() 

        # --- Lógica de COMPRA do Spread (Long) ---
        # Se o Z-Score está muito baixo (abaixo de -2.0) e não estamos comprados
        elif z < -self.limite_entrada and not self.position.is_long:
            # "Compra" o spread (Compra Ticker A, Vende Ticker B)
            self.buy()

        # --- Lógica de Saída (Fechar Posição) ---
        # Se o Z-Score voltou para perto da média
        elif abs(z) < self.limite_saida:
            self.position.close()


# --- Função Principal ---
if __name__ == "__main__":
    ticker1 = "ITUB4.SA"
    ticker2 = "BBDC4.SA"
    df_estrategia = validar_par_estatisticamente(ticker1, ticker2)

    if df_estrategia is not None:
        dados_ohlc_spread = pd.DataFrame({
            'Open': df_estrategia['Spread'],
            'High': df_estrategia['Spread'],
            'Low': df_estrategia['Spread'],
            'Close': df_estrategia['Spread'],
            'Volume': [1000] * len(df_estrategia), 
            'ZScore': df_estrategia['ZScore'] 
        })

        print("\nIniciando o Backtest da estratégia de Cointegração...")
        bt = Backtest(dados_ohlc_spread, 
                      EstrategiaJanoPairs, 
                      cash=100000, 
                      commission=.001)
        
        stats = bt.run()
        
        print(stats)
        bt.plot()