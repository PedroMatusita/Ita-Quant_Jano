import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np

# --- 1. FUNÇÕES AUXILIARES (MÉTRICAS E PLOTS) ---

def calcular_metricas(cumulative_returns, daily_returns):
    """
    Calcula as métricas de performance essenciais da estratégia.
    """
    if cumulative_returns.empty or daily_returns.empty:
        return {
            "retorno_total": 0,
            "retorno_anualizado": 0,
            "volatilidade_anualizada": 0,
            "sharpe_ratio_anualizado": 0,
            "max_drawdown": 0
        }

    # 1. Retorno Total
    retorno_total = (cumulative_returns.iloc[-1] - 1) * 100

    # 2. Retorno Anualizado
    dias_negocio = len(cumulative_returns)
    retorno_anualizado = ((1 + (retorno_total / 100)) ** (252.0 / dias_negocio) - 1) * 100

    # 3. Volatilidade Anualizada
    volatilidade_anualizada = (daily_returns.std() * np.sqrt(252)) * 100

    # 4. Sharpe Ratio Anualizado (assumindo taxa livre de risco 0)
    sharpe_ratio = 0
    if volatilidade_anualizada > 0:
        sharpe_ratio = (retorno_anualizado / volatilidade_anualizada)

    # 5. Max Drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    metrics = {
        "retorno_total": f"{retorno_total:.2f}%",
        "retorno_anualizado": f"{retorno_anualizado:.2f}%",
        "volatilidade_anualizada": f"{volatilidade_anualizada:.2f}%",
        "sharpe_ratio_anualizado": f"{sharpe_ratio:.2f}",
        "max_drawdown": f"{max_drawdown:.2f}%"
    }
    return metrics

def plotar_resultados(df_backtest, metrics, ticker1, ticker2, entry_z, exit_z, nome_estrategia="Estratégia"):
    """
    Plota os gráficos de performance da estratégia.
    """
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # --- GRÁFICO 1: Z-SCORE E SINAIS ---
    axs[0].set_title(f'Z-Score do Spread ({ticker1} vs {ticker2}) e Sinais de Operação', fontsize=14)
    df_backtest['z_score'].plot(ax=axs[0], label='Z-Score', color='blue')
    
    # Bandas de entrada e saída
    axs[0].axhline(entry_z, color='red', linestyle='--', label=f'Entry (+{entry_z}σ)')
    axs[0].axhline(-entry_z, color='green', linestyle='--', label=f'Entry (-{entry_z}σ)')
    axs[0].axhline(exit_z, color='gray', linestyle=':', label=f'Exit (±{exit_z}σ)')
    axs[0].axhline(-exit_z, color='gray', linestyle=':')
    axs[0].axhline(0, color='black', linestyle='-', lw=1)
    
    # Sinais de compra e venda (visual)
    axs[0].plot(df_backtest[df_backtest['sinal'] == 1].index, 
                df_backtest['z_score'][df_backtest['sinal'] == 1], 
                '^', markersize=10, color='g', label='Sinal de Compra (Long Spread)')
    
    axs[0].plot(df_backtest[df_backtest['sinal'] == -1].index, 
                df_backtest['z_score'][df_backtest['sinal'] == -1], 
                'v', markersize=10, color='r', label='Sinal de Venda (Short Spread)')

    axs[0].set_ylabel('Z-Score (Desvios Padrão)')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # --- GRÁFICO 2: CURVA DE CAPITAL (LUCRO) ---
    axs[1].set_title(f'Curva de Capital da Estratégia ({nome_estrategia})', fontsize=14)
    df_backtest['retorno_acumulado'].plot(ax=axs[1], label='Patrimônio da Estratégia', color='darkgreen', lw=2)
    
    # Adiciona as métricas como texto no gráfico
    metrics_text = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor']).to_string()
    axs[1].text(0.02, 0.4, metrics_text, 
                transform=axs[1].transAxes, 
                fontsize=11, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axs[1].set_ylabel('Retorno Acumulado (R$)')
    axs[1].set_xlabel('Data')
    axs[1].legend(loc='upper left')
    axs[1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f'JANO Backtester: {ticker1} vs {ticker2}', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

# --- 2. FUNÇÃO PRINCIPAL DO BACKTESTER ---

def rodar_backtest_media_normal(ticker1, ticker2,
                            start_calibracao, end_calibracao,
                            start_backtest, end_backtest,
                            entry_z=2.0, exit_z=0.5,
                            nome_estrategia="JANO Pair"):
    """
    Executa um backtest de pair trading completo, separando
    períodos de calibração e teste para evitar look-ahead bias.

    Como um analista sênior, explico o processo:
    1. CALIBRAÇÃO: Usamos dados *passados* (train) para
       encontrar os parâmetros (Hedge Ratio, Média, Desvio Padrão).
       Este período NÃO GERA LUCRO, apenas calibra o modelo.
    2. TESTE: Aplicamos esses parâmetros *fixos* em dados
       *novos* (test) para ver se a estratégia teria funcionado
       de forma "realista" (out-of-sample).
    """

    print(f"--- Iniciando Backtest JANO para {ticker1} vs {ticker2} ---")
    
    # --- 1. COLETA DE DADOS ---
    # Coletamos todos os dados de uma vez (calibração + teste)
    start_geral = pd.to_datetime(start_calibracao)
    end_geral = pd.to_datetime(end_backtest)
    
    try:
        dados = yf.download([ticker1, ticker2], start=start_geral, end=end_geral)['Close'].dropna()
        Y = dados[ticker1]
        X = dados[ticker2]
    except Exception as e:
        print(f"Erro ao baixar dados do yfinance: {e}")
        return None

    # --- 2. CALIBRAÇÃO (IN-SAMPLE / TRAINING) ---
    print(f"Iniciando Calibração ({start_calibracao} a {end_calibracao})...")

    # Separando dados de calibração
    Y_train = Y.loc[start_calibracao:end_calibracao]
    X_train = X.loc[start_calibracao:end_calibracao]
    
    if Y_train.empty or X_train.empty:
        print("Erro: Período de calibração não retornou dados.")
        return None

    # Encontra o Hedge Ratio (Beta)
    X_train_const = sm.add_constant(X_train)
    modelo = sm.OLS(Y_train, X_train_const).fit()
    hedge_ratio = modelo.params[ticker2]
    
    # Calcula o spread no período de calibração
    spread_train = Y_train - hedge_ratio * X_train
    
    # --- 3. TESTE DE COINTEGRAÇÃO (CRÍTICO) ---
    # Verificamos se a relação existia NO PASSADO (calibração)
    adf_result = adfuller(spread_train, autolag='AIC')
    p_value = adf_result[1]
    
    if p_value > 0.05:
        print(f"*** ATENÇÃO: PAR NÃO COINTEGRADO (P-Value: {p_value:.4f}) ***")
        print("O spread não é estacionário no período de calibração.")
        print("Continuar o backtest não é recomendado. Abortando.")
        return None
    
    # Se passou, calibramos os parâmetros
    media_spread = spread_train.mean()
    desvio_spread = spread_train.std()
    
    print(f"Calibração bem-sucedida (P-Value: {p_value:.4f}).")
    print(f"Parâmetros encontrados:")
    print(f"  Hedge Ratio (Beta): {hedge_ratio:.4f}")
    print(f"  Média do Spread: {media_spread:.4f}")
    print(f"  Desvio Padrão do Spread: {desvio_spread:.4f}")
    
    # --- 4. EXECUÇÃO (OUT-OF-SAMPLE / TESTING) ---
    print(f"Iniciando Execução do Backtest ({start_backtest} a {end_backtest})...")

    # Separando dados de teste
    Y_test = Y.loc[start_backtest:end_backtest]
    X_test = X.loc[start_backtest:end_backtest]
    
    if Y_test.empty or X_test.empty:
        print("Erro: Período de backtest não retornou dados.")
        return None

    df_backtest = pd.DataFrame(index=Y_test.index)
    df_backtest['Y'] = Y_test
    df_backtest['X'] = X_test

    # Calcula o spread (out-of-sample) USANDO OS PARÂMETROS DA CALIBRAÇÃO
    df_backtest['spread'] = df_backtest['Y'] - hedge_ratio * df_backtest['X']
    
    # Calcula o Z-Score (out-of-sample)
    df_backtest['z_score'] = (df_backtest['spread'] - media_spread) / desvio_spread

    # --- 5. GERAÇÃO DE SINAIS E POSIÇÕES ---
    df_backtest['posicao'] = 0
    df_backtest['sinal'] = 0 # Para visualização no gráfico

    # Variáveis de estado
    posicao_atual = 0
    for i in range(len(df_backtest)):
        z = df_backtest['z_score'].iloc[i]
        
        if posicao_atual == 0:
            if z < -entry_z: # Spread muito baixo, comprar (Long Spread)
                posicao_atual = 1
                df_backtest.iloc[i, df_backtest.columns.get_loc('sinal')] = 1
            elif z > entry_z: # Spread muito alto, vender (Short Spread)
                posicao_atual = -1
                df_backtest.iloc[i, df_backtest.columns.get_loc('sinal')] = -1
        
        elif posicao_atual == 1: # Estou comprado no spread
            if z >= -exit_z: # Spread voltou para perto da média
                posicao_atual = 0
                
        elif posicao_atual == -1: # Estou vendido no spread
            if z <= exit_z: # Spread voltou para perto da média
                posicao_atual = 0

        df_backtest.iloc[i, df_backtest.columns.get_loc('posicao')] = posicao_atual
        
    # --- 6. CÁLCULO DE PERFORMANCE E LUCRO ---
    
    # Calculamos o retorno diário do spread
    # (Comprar Y e Vender X)
    retorno_Y = df_backtest['Y'].pct_change()
    retorno_X = df_backtest['X'].pct_change()
    
    # Retorno diário da carteira (long/short)
    # Posição 1 = Long Y, Short X
    # Posição -1 = Short Y, Long X
    # Usamos .shift(1) pois a posição é aberta no dia anterior (sinal)
    # e o retorno é realizado no dia seguinte.
    df_backtest['retorno_diario'] = (
        retorno_Y * df_backtest['posicao'].shift(1) - \
        retorno_X * df_backtest['posicao'].shift(1) * hedge_ratio
    )
    
    # Ajuste de normalização: O hedge ratio ajusta as proporções financeiras
    # (Ex: 1 ação de Y para cada 'hedge_ratio' ações de X)
    # A normalização do capital é complexa, vamos focar no retorno do portfólio
    # normalizado pela volatilidade do spread.
    
    # Uma forma mais simples e robusta (sem alavancagem):
    # Retorno do Spread (Y - Beta*X).
    df_backtest['retorno_spread'] = df_backtest['spread'].pct_change()
    
    # O retorno da estratégia é o retorno do spread no dia
    # se estivermos na posição correta.
    df_backtest['retorno_estrategia'] = df_backtest['retorno_spread'] * df_backtest['posicao'].shift(1)
    df_backtest['retorno_estrategia'].fillna(0, inplace=True)
    
    # Curva de capital (Lucro Acumulado)
    df_backtest['retorno_acumulado'] = (1 + df_backtest['retorno_estrategia']).cumprod()

    # --- 7. MÉTRICAS FINAIS E PLOTS ---
    print("\n--- Backtest Concluído. Métricas (Out-of-Sample): ---")
    
    metrics = calcular_metricas(
        df_backtest['retorno_acumulado'], 
        df_backtest['retorno_estrategia']
    )
    
    # Imprime a tabela de métricas
    for key, value in metrics.items():
        print(f"  {key:<25}: {value}")

    # Plota os gráficos
    plotar_resultados(df_backtest, metrics, ticker1, ticker2, entry_z, exit_z, nome_estrategia)

    return df_backtest, metrics


# --- 3. BLOCO DE EXECUÇÃO PRINCIPAL (EXEMPLO) ---

if __name__ == "__main__":
    
    # --- Parâmetros do Desafio ---
    TICKER_Y = "ITUB4.SA" # Sugerido pelo Jano Pair Selector
    TICKER_X = "BBDC4.SA" # Sugerido pelo Jano Pair Selector
    
    # Períodos de Calibração (Train) e Teste (Test)
    # IMPORTANTE: Não podem ter sobreposição (overlap)
    
    # Calibração: 2 anos de dados
    CAL_START = "2022-01-01"
    CAL_END = "2023-12-31"
    
    # Backtest: Próximos 10 meses (dados "novos")
    TEST_START = "2024-01-01"
    TEST_END = "2024-10-28" # Data de hoje (exemplo)
    
    # Limiares de Z-Score
    ENTRY_THRESHOLD = 2.0  # Entrar quando o spread estiver a 2 desvios padrão
    EXIT_THRESHOLD = 0.5   # Sair quando voltar a 0.5 desvio padrão
    
    # Roda o backtester completo
    rodar_backtest_media_normal(
        TICKER_Y, TICKER_X,
        CAL_START, CAL_END,
        TEST_START, TEST_END,
        ENTRY_THRESHOLD, EXIT_THRESHOLD,
        nome_estrategia="Duelo dos Bancos" # Nome da IA
    )