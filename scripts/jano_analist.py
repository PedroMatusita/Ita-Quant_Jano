import json
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from jano_pair_selector import buscar_par_com_llm_gemini
from google import genai
from dotenv import load_dotenv
import numpy as np
from google.genai import types


# -------------------------------

# Função para validar cointegração do par

# -------------------------------


def validar_par_estatisticamente(ticker1, ticker2, calib_end): #realiza a calibração
    """
    Verifica se o par é cointegrado e retorna o spread 
    pronto para o backtest.
    True = par válido
    False = par inválido
    """

    print(f"Analisando cointegração para {ticker1} e {ticker2}...")

    #pega os últimos calib_end+1 anos
    startL = datetime.now() + timedelta(days = -365*(calib_end+1))
    endL = datetime.now()
    try:
        trainset = np.arange(0, 252*calib_end) #calibra com os primeiros calib_end anos
        dados = yf.download([ticker1, ticker2], start=startL, end=endL)['Close'].dropna()
        Y_train = dados[ticker1].iloc[trainset]
        X_train = dados[ticker2].iloc[trainset]
        
        X_com_constante = sm.add_constant(X_train)
        modelo = sm.OLS(Y_train, X_com_constante).fit()
    except Exception as e:
        print(f"Erro download dos dados do yfinance ou no fit, busque por outro par. {e}")
        return False, None, None
    
    beta = modelo.params.iloc[1]
    print(f"Hedge Ratio (Beta) encontrado: {beta:.4f}")
    
    spread = dados[ticker1] - beta * dados[ticker2]
    
    #testa cointegração para o período todo de 5 anos
    teste_adf = adfuller(spread, autolag='AIC')
    p_value = teste_adf[1]
    
    print(f"Teste ADF P-Value: {p_value:.4f}")
    
    if p_value < 0.05: 
        print("Resultado: O PAR É COINTEGRADO (p-value < 0.05).")
        print("Pode prosseguir para o backtest.")
        return True, dados, beta
        
        # return df_backtest
        
    else:
        print("Resultado: O PAR NÃO É COINTEGRADO (p-value >= 0.05).")
        print("Rejeitar este par. Tente outros tickers.")
        return False, None, None

def encontra_par_valido_com_gemini(client: genai.Client):
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
    for i in range(15):
        sugestao_llm = buscar_par_com_llm_gemini(chat, rerun)
    
        if sugestao_llm and 'par' in sugestao_llm and len(sugestao_llm['par']) == 2:
            ticker1, ticker2 = sugestao_llm['par']
            print(f"\nSugestão recebida da LLM:")
            print(f"Par: {ticker1} e {ticker2}")


            calib_end = 4
            # 'calib_end' define o número de anos usados para calibrar os parâmetros
            # os dados retornados são referentes a calib_end+1 anos anteriores ao dia atual
            # a calibragem é feita com dados[0, calib_end*252] (pois há 252 dias de trade num ano, desconsiderando feriados e etc)
            #    trainset = np.arange(0, 252*calib_end)
            #
            # sugestão 1: como na calibração foram usados os calib_end primeiros anos, usar para teste o resto do conjunto de dados
            #           para evitar viés de look-ahead, ou seja, para tudo usar o intervalo 'testset' de dados
            #
            #    testset = np.arange(trainset.shape[0], dados.shape[0])
            #
            #    exemplo de uso do intervalo testset para separar os dados para teste
            #        ticker1_testset = dados[ticker1].iloc[testset]
            #        ticker2_testset = dados[ticker2].iloc[testset]
            # 
            # sugestão 2: fazer um modo de backtest usando médias móveis e um usando média do conjunto todo para comparar
            
            resultado, dados, hedge = validar_par_estatisticamente(ticker1, ticker2, calib_end)
            if resultado is False:
                rerun = True
            else:
                print(f"O par {ticker1} e {ticker2} é um par válido, continue para o backtesting.")
                print(f"Hedge Ratio dos {calib_end} anos de calibração é: {hedge}")
                return ticker1, ticker2, dados, hedge
    # Caso não encontre em 15 tentativas, retorna nada
    return None, None, None, None

def analisar_backtest_com_llm(client: genai.Client, metrics: dict, nome_estrategia: str):
    """
    Usa a GenAI para escrever a conclusão do backtest,
    como se fosse para o comitê gestor.
    """
    print("\n--- JANO Analyst (GenAI) analisando resultados do backtest... ---")
    
    prompt_sistema = """
    Você é um analista quantitativo sênior e cético.
    Sua tarefa é analisar os resultados (métricas) de um backtest 
    'out-of-sample' e escrever uma breve conclusão para um comitê 
    gestor. Seja honesto sobre os pontos fortes e fracos.
    """
    
    # Converte as métricas em texto
    metrics_str = json.dumps(metrics, indent=2)
    
    prompt_usuario = f"""
    Analise as métricas de performance da estratégia '{nome_estrategia}':
    
    {metrics_str}
    
    Escreva uma breve análise (em 3 parágrafos) respondendo:
    1. A estratégia foi lucrativa e o retorno compensou o risco 
       (analise o Sharpe Ratio e o Max Drawdown)?
    2. A performance 'out-of-sample' valida a hipótese de cointegração?
    3. Você recomendaria estudar esta estratégia mais a fundo para implementação?
    """
    
    # (Lógica da chamada ao Gemini, sem JSON forçado, 
    # pois queremos texto corrido)
    
    # response = client.models.generate_content(...)
    # print("--- CONCLUSÃO DO COMITÊ (Gerada por IA) ---")
    # print(response.text)



# --- Exemplo de uso ---
if __name__ == "__main__":
    load_dotenv() 
    client = genai.Client()
    encontra_par_valido_com_gemini(client)