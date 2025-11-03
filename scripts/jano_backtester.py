# ---------------------------------------------------------------------
# MÓDULO: JANO BACKTESTER (Versão Final e Estável)
# DESCRIÇÃO: Backtest OOS com Z-Score Adaptativo (Rolling).
# ---------------------------------------------------------------------

import traceback
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from google import genai
from dotenv import load_dotenv
import numpy as np
from google.genai import types

from jano_analist import analisar_backtest_com_llm, encontra_par_valido_com_gemini

# --- 1. CONFIGURAÇÕES GLOBAIS ---

# Constantes de custos (essenciais para um backtest realista)
COSTOS = {
    "CORRETAGEM_FIXA_POR_ORDEM": 4.50,
    "TAXA_B3_PCT_SOBRE_VALOR": 0.0003, 
    "ALIQUOTA_IR_SWING_TRADE": 0.15,
    "IOF_DAY_TRADE": 0.01 
}

# Parâmetros da Estratégia
CAPITAL_INICIAL = 100000.00
ENTRY_THRESHOLD = 1.0  
EXIT_THRESHOLD = 0.7   
ROLLING_WINDOW_ZSCORE = 30 # Janela móvel de 30 dias para Z-Score

# --- 2. FUNÇÕES PRINCIPAIS DO BACKTESTER ---

def executar_backtest_oos(
    ticker1,
    ticker2,
    dados_5a,
    params_calibracao,
    calib_end_date,
    benchmark_ticker = "^BVSP",
    rolling_window = 30
):
    """
    Função principal que orquestra o backtest Out-of-Sample (OOS) com Z-Score adaptativo.
    """
    print("--- INICIANDO JANO BACKTESTER (ADAPTATIVO / ROLLING Z-SCORE) ---")
    
    # 1. Preparar Dados de Teste (O ano 5)
    df_teste = dados_5a.loc[dados_5a.index > calib_end_date].copy()
    if df_teste.empty:
        print(f"Erro: Não há dados de teste após {calib_end_date}.")
        return

    print(f"Período de Teste: {df_teste.index.min().date()} a {df_teste.index.max().date()}")
    
    # 2. Calcular Z-Score (usando hedge_ratio fixo e rolling stats)
    hedge_ratio_calibrado = params_calibracao['hedge_ratio']
    
    _calcular_zscore_oos(
        df_teste, 
        ticker1, 
        ticker2, 
        hedge_ratio_calibrado, 
        rolling_window
    )
    
    # 3. Gerar Sinais e Posições
    _gerar_sinais_e_posicoes(df_teste, ENTRY_THRESHOLD, EXIT_THRESHOLD)
    
    # 4. Simular Log de Operações e Calcular P/L, Custos e Impostos
    df_log_operacoes = _simular_log_operacoes(
        df_teste, 
        ticker1, 
        ticker2, 
        hedge_ratio_calibrado, 
        CAPITAL_INICIAL
    )
    
    # 5. Gerar Curva de Capital Diária
    df_curva_capital = _gerar_curva_capital(df_teste, df_log_operacoes, CAPITAL_INICIAL)
    
    # 6. Baixar e Preparar Benchmark
    df_benchmark = _baixar_benchmark(
        benchmark_ticker, 
        df_teste.index.min(), 
        df_teste.index.max()
    )
    
    # 7. Gerar Análises (Tabelas e Gráficos)
    print("\n--- GERANDO RELATÓRIO DE PERFORMANCE ---")
    tabela_resumo_financeiro = _gerar_tabela_resumo_financeiro(df_log_operacoes, CAPITAL_INICIAL)
    tabela_metricas_risco = _gerar_tabela_metricas_risco(df_curva_capital, df_benchmark)
    
    # ✅ Substituído por visualização gráfica
    # _gerar_tabela_resumo_financeiro(df_log_operacoes, CAPITAL_INICIAL, exibir_plot=True)
    # _gerar_tabela_metricas_risco(df_curva_capital, df_benchmark, exibir_plot=True)
    _plotar_tabela_operacoes(df_log_operacoes)    
    
    print("\n[TABELA 1: RESUMO FINANCEIRO (LÍQUIDO)]")
    print(tabela_resumo_financeiro.to_markdown(floatfmt=".2f"))
    
    print("\n[TABELA 2: MÉTRICAS DE RISCO E PERFORMANCE]")
    print(tabela_metricas_risco.to_markdown(floatfmt=".2f"))
    
    print("\n[TABELA 3: LOG DE OPERAÇÕES REALIZADAS - PRIMEIRAS 5]")
    # Usamos o tail se o df for muito grande
    print(df_log_operacoes.head().to_markdown(floatfmt=".2f"))
    
    # 8. Plotar Gráficos
    _plotar_curva_capital(df_curva_capital, df_benchmark)
    _plotar_zscore_sinais(df_teste, ticker1, ticker2)

    
    print("--- JANO BACKTESTER FINALIZADO ---")

    # ✅ Consolidar métricas em um dicionário para uso posterior (LLM)
    metrics_dict = {
        "Ticker 1": ticker1,
        "Ticker 2": ticker2,
        "Capital Inicial": CAPITAL_INICIAL,
        "Capital Final": float(tabela_resumo_financeiro.loc["Capital Final", "Resumo Financeiro"]),
        "Lucro Líquido Total": float(tabela_resumo_financeiro.loc["Lucro Líquido Total", "Resumo Financeiro"]),
        "Retorno Líquido (%)": float(tabela_resumo_financeiro.loc["Retorno Líquido (%)", "Resumo Financeiro"]),
        "Total de Operações": int(tabela_resumo_financeiro.loc["Total de Operações", "Resumo Financeiro"]),
        "Taxa de Acerto (%)": float(tabela_resumo_financeiro.loc["Taxa de Acerto (%)", "Resumo Financeiro"]),
        "Retorno Total (%)": float(tabela_metricas_risco.loc["Estratégia JANO", "Retorno Total (%)"]),
        "Retorno Anualizado (%)": float(tabela_metricas_risco.loc["Estratégia JANO", "Retorno Anualizado (%)"]),
        "Volatilidade Anualizada (%)": float(tabela_metricas_risco.loc["Estratégia JANO", "Volatilidade Anualizada (%)"]),
        "Sharpe Ratio (Anualizado)": float(tabela_metricas_risco.loc["Estratégia JANO", "Sharpe Ratio (Anualizado)"]),
        "Max Drawdown (%)": float(tabela_metricas_risco.loc["Estratégia JANO", "Max Drawdown (%)"]),
    }

    return df_log_operacoes, tabela_resumo_financeiro, tabela_metricas_risco, metrics_dict


    print("--- JANO BACKTESTER FINALIZADO ---")

    return df_log_operacoes, tabela_resumo_financeiro, tabela_metricas_risco


# --- 3. FUNÇÕES AUXILIARES (Passo a Passo) ---

def _calcular_zscore_oos(df, t1, t2, hedge_ratio, rolling_window):
    """Calcula o spread e o z-score usando um HEDGE RATIO FIXO e uma JANELA MÓVEL."""
    df['spread'] = df[t1] - (hedge_ratio * df[t2])
    
    df['spread_mean_rolling'] = df['spread'].rolling(window=rolling_window).mean()
    df['spread_std_rolling'] = df['spread'].rolling(window=rolling_window).std()
    
    df['z_score'] = (df['spread'] - df['spread_mean_rolling']) / df['spread_std_rolling']
    

def _gerar_sinais_e_posicoes(df, entry_z, exit_z):
    """Gera as posições desejadas (1, -1, 0) com base no Z-Score."""
    df['posicao'] = 0
    posicao_atual = 0
    
    for i in range(len(df)):
        z = df['z_score'].iloc[i]
        
        # Ignorar NaNs iniciais da janela móvel
        if pd.isna(z):
            continue
            
        if posicao_atual == 0:
            if z < -entry_z:
                posicao_atual = 1
            elif z > entry_z:
                posicao_atual = -1
        
        elif posicao_atual == 1:
            if z >= -exit_z:
                posicao_atual = 0
                
        elif posicao_atual == -1:
            if z <= exit_z:
                posicao_atual = 0
        
        df.at[df.index[i], 'posicao'] = posicao_atual

def _simular_log_operacoes(df_teste, t1, t2, hedge_ratio, capital):
    """Simula as operações trade-a-trade para criar um log detalhado com custos e impostos."""
    log_operacoes = []
    trade_aberto = {}
    capital_atual = capital
    
    sinais_de_mudanca = df_teste['posicao'].diff().fillna(0)
    
    for i in range(len(df_teste)):
        data = df_teste.index[i]
        sinal_hoje = df_teste['posicao'].iloc[i]
        
        if sinais_de_mudanca.iloc[i] == 0:
            continue
            
        # --- 1. Abertura de Posição ---
        if sinal_hoje != 0 and not trade_aberto:
            trade_aberto = {
                'data_entrada': data,
                'tipo': 'LONG_SPREAD' if sinal_hoje == 1 else 'SHORT_SPREAD',
                'preco_Y_entrada': df_teste[t1].iloc[i],
                'preco_X_entrada': df_teste[t2].iloc[i],
            }
            
            valor_alocado_Y = capital_atual * 0.5
            trade_aberto['qtd_Y'] = valor_alocado_Y / trade_aberto['preco_Y_entrada']
            trade_aberto['qtd_X'] = trade_aberto['qtd_Y'] * hedge_ratio
            
            # Custos de Entrada (4 ordens)
            custos_fixos = COSTOS['CORRETAGEM_FIXA_POR_ORDEM'] * 2
            taxas_Y = (trade_aberto['qtd_Y'] * trade_aberto['preco_Y_entrada']) * COSTOS['TAXA_B3_PCT_SOBRE_VALOR']
            taxas_X = (trade_aberto['qtd_X'] * trade_aberto['preco_X_entrada']) * COSTOS['TAXA_B3_PCT_SOBRE_VALOR']
            trade_aberto['custos_operacionais'] = custos_fixos + taxas_Y + taxas_X
            
            capital_atual -= trade_aberto['custos_operacionais']

        # --- 2. Fechamento de Posição ---
        elif sinal_hoje == 0 and trade_aberto:
            preco_Y_saida = df_teste[t1].iloc[i]
            preco_X_saida = df_teste[t2].iloc[i]
            
            # Custos de Saída (4 ordens totais)
            custos_fixos_saida = COSTOS['CORRETAGEM_FIXA_POR_ORDEM'] * 2
            taxas_Y = (trade_aberto['qtd_Y'] * preco_Y_saida) * COSTOS['TAXA_B3_PCT_SOBRE_VALOR']
            taxas_X = (trade_aberto['qtd_X'] * preco_X_saida) * COSTOS['TAXA_B3_PCT_SOBRE_VALOR']
            custos_saida = custos_fixos_saida + taxas_Y + taxas_X
            
            trade_aberto['custos_operacionais'] += custos_saida
            
            # Calcular P/L (Profit/Loss)
            p_l_Y = (preco_Y_saida - trade_aberto['preco_Y_entrada']) * trade_aberto['qtd_Y']
            p_l_X = (trade_aberto['preco_X_entrada'] - preco_X_saida) * trade_aberto['qtd_X'] # Invertido, pois X foi vendido
            
            # Ajuste de P/L para SHORT SPREAD
            if trade_aberto['tipo'] == 'SHORT_SPREAD':
                # Long Spread: (Long Y) + (Short X). P/L é P_L_Y + P_L_X.
                # Short Spread: (Short Y) + (Long X). P/L é P_L_Y*(-1) + P_L_X*(-1)
                p_l_Y *= -1
                p_l_X *= -1
            
            trade_aberto['p_l_bruto'] = p_l_Y + p_l_X
            trade_aberto['p_l_liq_antes_ir'] = trade_aberto['p_l_bruto'] - trade_aberto['custos_operacionais']
            
            trade_aberto['imposto_ir'] = max(0, trade_aberto['p_l_liq_antes_ir'] * COSTOS['ALIQUOTA_IR_SWING_TRADE'])
            trade_aberto['p_l_liquido'] = trade_aberto['p_l_liq_antes_ir'] - trade_aberto['imposto_ir']
            
            capital_atual += trade_aberto['p_l_liquido']
            
            trade_aberto['data_saida'] = data
            trade_aberto['duracao_dias'] = (data - trade_aberto['data_entrada']).days
            trade_aberto['capital_final_trade'] = capital_atual
            
            log_operacoes.append(trade_aberto)
            trade_aberto = {}
            
    return pd.DataFrame(log_operacoes)

def _gerar_curva_capital(df_teste, df_log, capital_inicial):
    """Cria uma série temporal diária do valor do portfólio."""
    df_capital = pd.DataFrame(index=df_teste.index)
    df_capital['capital'] = np.nan
    
    if df_log.empty:
        df_capital['capital'] = capital_inicial
        return df_capital.ffill()

    df_capital.at[df_log['data_entrada'].iloc[0], 'capital'] = capital_inicial
    
    for _, row in df_log.iterrows():
        df_capital.at[row['data_saida'], 'capital'] = row['capital_final_trade']
        
    # Correção do FutureWarning (usando ffill() e não fillna(method='ffill'))
    df_capital['capital'] = df_capital['capital'].ffill()
    df_capital['capital'] = df_capital['capital'].fillna(capital_inicial) 
    
    df_capital['retorno_diario'] = df_capital['capital'].pct_change().fillna(0)
    return df_capital


def _baixar_benchmark(ticker, start, end):
    df_bm = yf.download(ticker, start=start, end=end)
    
    # Extrai corretamente o fechamento, seja qual for o formato retornado
    if isinstance(df_bm.columns, pd.MultiIndex):
        df_bm = df_bm['Close']
    else:
        df_bm = df_bm['Close']
        
    df_bm = df_bm.pct_change().fillna(0)
    df_bm = (1 + df_bm).cumprod() * CAPITAL_INICIAL
    df_bm.name = 'Benchmark'
    return df_bm

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- 4. FUNÇÕES DE GERAÇÃO E PLOTAGEM DE TABELAS --- #

import matplotlib.pyplot as plt
import textwrap

def _plotar_tabela(df, titulo, cor_cabecalho="#1E90FF", cor_fundo="#F8F8F8"):
    """
    Exibe uma tabela pandas formatada usando matplotlib,
    com quebra de linha automática e ajuste dinâmico de tamanho.
    """
    # --- Quebra automática de texto nos cabeçalhos ---
    wrapped_col_labels = [
        "\n".join(textwrap.wrap(str(col), width=18)) for col in df.columns
    ]
    wrapped_row_labels = [
        "\n".join(textwrap.wrap(str(idx), width=15)) for idx in df.index
    ]

    # --- Cria figura proporcional ao número de colunas e linhas ---
    fig_width = max(8, len(df.columns) * 1.2)
    fig_height = 1.5 + 0.4 * len(df)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # --- Gera tabela formatada ---
    tabela = ax.table(
        cellText=df.round(2).values,
        colLabels=wrapped_col_labels,
        rowLabels=wrapped_row_labels,
        cellLoc="center",
        loc="center"
    )

    # --- Estilo de fonte e escala ---
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.1, 1.2)

    # --- Estilo de cores e alinhamento ---
    for (row, col), cell in tabela.get_celld().items():
        if row == 0:  # Cabeçalho
            cell.set_facecolor(cor_cabecalho)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("white")
        else:
            cell.set_facecolor(cor_fundo)

        # Ajuste de margens internas
        cell.PAD = 0.02

    plt.title(titulo, fontsize=14, weight="bold", pad=15)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    plt.show()


def _gerar_tabela_resumo_financeiro(df_log, capital_inicial, exibir_plot=True):
    """Cria e (opcionalmente) exibe a tabela de resumo financeiro."""
    if df_log.empty:
        df = pd.Series({
            "Faturamento Bruto (P/L)": 0,
            "Total Custos (Corretagem + Taxas)": 0,
            "Total Imposto de Renda (IR)": 0,
            "Lucro Líquido Total": 0,
            "Capital Inicial": capital_inicial,
            "Capital Final": capital_inicial,
            "Retorno Líquido (%)": 0,
            "Total de Operações": 0,
            "Taxa de Acerto (%)": 0
        }).to_frame('Resumo Financeiro')
    else:
        lucro_liquido_total = df_log['p_l_liquido'].sum()
        capital_final = capital_inicial + lucro_liquido_total

        total_operacoes = len(df_log)
        operacoes_vencedoras = len(df_log[df_log['p_l_liquido'] > 0])
        taxa_acerto = (operacoes_vencedoras / total_operacoes) * 100 if total_operacoes > 0 else 0

        resumo = {
            "Faturamento Bruto (P/L)": df_log['p_l_bruto'].sum(),
            "Total Custos (Corretagem + Taxas)": df_log['custos_operacionais'].sum(),
            "Total Imposto de Renda (IR)": df_log['imposto_ir'].sum(),
            "Lucro Líquido Total": lucro_liquido_total,
            "Capital Inicial": capital_inicial,
            "Capital Final": capital_final,
            "Retorno Líquido (%)": (lucro_liquido_total / capital_inicial) * 100,
            "Total de Operações": total_operacoes,
            "Taxa de Acerto (%)": taxa_acerto
        }
        df = pd.Series(resumo).to_frame('Resumo Financeiro')

    if exibir_plot:
        _plotar_tabela(df.T, "📊 Tabela 1: Resumo Financeiro (Líquido)", cor_cabecalho="#2E8B57")
    return df


def _gerar_tabela_metricas_risco(df_capital, df_benchmark, exibir_plot=True):
    """Gera métricas de risco (Sharpe, Drawdown) e plota como tabela visual."""
    dias_teste = len(df_capital)
    anos_teste = dias_teste / 252.0

    retorno_diario_strat = df_capital['retorno_diario']
    retorno_total_strat = float((df_capital['capital'].iloc[-1] / df_capital['capital'].iloc[0]) - 1)
    retorno_anual_strat = float((1 + retorno_total_strat) ** (1 / anos_teste) - 1)
    vol_anual_strat = float(retorno_diario_strat.std() * np.sqrt(252))
    sharpe_strat = float(retorno_anual_strat / vol_anual_strat) if vol_anual_strat > 0 else 0.0

    cum_ret = df_capital['capital']
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    max_drawdown_strat = float(drawdown.min())

    retorno_total_bm = float((df_benchmark.iloc[-1] / df_benchmark.iloc[0]) - 1)
    retorno_anual_bm = float((1 + retorno_total_bm) ** (1 / anos_teste) - 1)

    metricas = {
        "Retorno Total (%)": [retorno_total_strat * 100, retorno_total_bm * 100],
        "Retorno Anualizado (%)": [retorno_anual_strat * 100, retorno_anual_bm * 100],
        "Volatilidade Anualizada (%)": [vol_anual_strat * 100, np.nan],
        "Sharpe Ratio (Anualizado)": [sharpe_strat, np.nan],
        "Max Drawdown (%)": [max_drawdown_strat * 100, np.nan],
    }

    df = pd.DataFrame(metricas, index=['Estratégia JANO', 'Benchmark (IBOV)']).astype(float)
    if exibir_plot:
        _plotar_tabela(df, "📈 Tabela 2: Métricas de Risco e Performance", cor_cabecalho="#1E90FF")
    return df


def _plotar_tabela_operacoes(df_log):
    """Mostra visualmente as 5 primeiras operações realizadas."""
    if df_log.empty:
        print("⚠️ Nenhuma operação realizada.")
        return

    df_head = df_log.head(5)[[
        "data_entrada", "data_saida", "tipo", 
        "p_l_liquido", "custos_operacionais", 
        "imposto_ir", "duracao_dias"
    ]]
    df_head = df_head.rename(columns={
        "data_entrada": "Entrada",
        "data_saida": "Saída",
        "tipo": "Tipo",
        "p_l_liquido": "P/L Líquido",
        "custos_operacionais": "Custos",
        "imposto_ir": "IR",
        "duracao_dias": "Dias"
    })
    _plotar_tabela(df_head, "🧾 Tabela 3: Log de Operações (Top 5)", cor_cabecalho="#8B008B")


# --- 5. FUNÇÕES DE PLOTAGEM ---

def _plotar_curva_capital(df_capital, df_benchmark):
    """Plota a curva de capital da estratégia contra o benchmark."""
    plt.figure(figsize=(14, 7))
    plt.plot(df_capital.index, df_capital['capital'], label='Estratégia JANO (Líquido)', color='g', lw=2)
    plt.plot(df_benchmark.index, df_benchmark, label='Benchmark (IBOV)', color='gray', linestyle='--')
    
    plt.title('Curva de Capital (Out-of-Sample) vs. Benchmark', fontsize=16)
    plt.ylabel('Patrimônio (R$)', fontsize=12)
    plt.xlabel('Data', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # ✅ CORRIGIDO:
    import matplotlib.ticker as mticker
    plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('R${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()

def _plotar_zscore_sinais(df_teste, t1, t2):
    """Plota o Z-Score e as posições de entrada/saída."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Z-Score e Bandas
    df_teste['z_score'].plot(ax=ax, label='Z-Score (Out-of-Sample)', color='b', lw=1.5)
    ax.axhline(ENTRY_THRESHOLD, color='red', linestyle='--', label=f'Entry (±{ENTRY_THRESHOLD}σ)')
    ax.axhline(-ENTRY_THRESHOLD, color='red', linestyle='--')
    ax.axhline(EXIT_THRESHOLD, color='gray', linestyle=':', label=f'Exit (±{EXIT_THRESHOLD}σ)')
    ax.axhline(-EXIT_THRESHOLD, color='gray', linestyle=':')
    ax.axhline(0, color='black', linestyle='-', lw=0.5)
    
    # Sinais de Posição
    ax.fill_between(df_teste.index, df_teste['z_score'].min(), df_teste['z_score'].max(), 
                    where=df_teste['posicao'] == 1, 
                    facecolor='green', alpha=0.2, label='Posição Long Spread')
    
    ax.fill_between(df_teste.index, df_teste['z_score'].min(), df_teste['z_score'].max(), 
                    where=df_teste['posicao'] == -1, 
                    facecolor='red', alpha=0.2, label='Posição Short Spread')
    
    plt.title(f'Z-Score e Posições da Estratégia ({t1} vs {t2})', fontsize=16)
    plt.ylabel('Z-Score (Desvios Padrão)', fontsize=12)
    plt.xlabel('Data', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- 6. BLOCO DE EXECUÇÃO PRINCIPAL (Orquestrador) ---
if __name__ == "__main__":
    print("--- SIMULAÇÃO DE EXECUÇÃO ---")
    load_dotenv()
    client = genai.Client()
    ticker1, ticker2, dados5, hedge, calib_end_date = encontra_par_valido_com_gemini(client)

    parametros_calibracao = {"hedge_ratio": hedge}

    try:
        df_log, tab_fin, tab_risk, metrics = executar_backtest_oos(
            ticker1=ticker1,
            ticker2=ticker2,
            dados_5a=dados5,
            params_calibracao=parametros_calibracao,
            calib_end_date=calib_end_date,
            benchmark_ticker="^BVSP",
            rolling_window=ROLLING_WINDOW_ZSCORE
        )

        # --- Análise Automática ---
        if metrics:
            _ = analisar_backtest_com_llm(client, metrics, nome_estrategia=f"JANO ({ticker1}/{ticker2})")
    except Exception as e:
        print(f"Erro ao processar dados: {e}")
        traceback.print_exc()
