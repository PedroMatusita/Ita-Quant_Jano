from backtesting import Backtest, Strategy
import pandas as pd
from jano_analist import validar_par_estatisticamente

# 1. Defina a Estratégia de Pair Trading (baseada no Z-Score)
class EstrategiaJanoPairs(Strategy):
    # Limites para operar
    limite_entrada = 2.0  # Entrar quando Z-Score for 2.0
    limite_saida = 0.5    # Sair quando Z-Score voltar para 0.5

    def init(self):
        # "Contrabandeia" o Z-Score para dentro da estratégia
        # self.data.ZScore nos dá acesso à coluna ZScore
        self.zscore = self.I(lambda: self.data.ZScore, name="ZScore")

    def next(self):
        z = self.zscore[-1] # Pega o valor mais recente do Z-Score

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
    # 1. Buscar o par e os dados da estratégia
    ticker1 = "ITUB4.SA"
    ticker2 = "BBDC4.SA"
    df_estrategia = validar_par_estatisticamente(ticker1, ticker2)

    if df_estrategia is not None:
        # 2. "Enganar" o backtester com o formato OHLC
        # Vamos usar o Spread como se fosse o preço de um ativo
        dados_ohlc_spread = pd.DataFrame({
            'Open': df_estrategia['Spread'],
            'High': df_estrategia['Spread'],
            'Low': df_estrategia['Spread'],
            'Close': df_estrategia['Spread'],
            'Volume': [1000] * len(df_estrategia), # Volume fictício
            'ZScore': df_estrategia['ZScore'] # Coluna extra
        })

        # 3. Rodar o Backtest
        print("\nIniciando o Backtest da estratégia de Cointegração...")
        bt = Backtest(dados_ohlc_spread, 
                      EstrategiaJanoPairs, 
                      cash=100000, 
                      commission=.001)
        
        stats = bt.run()
        
        # 4. Ver Resultados
        print(stats)
        bt.plot()