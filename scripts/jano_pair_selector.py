import os
import json
import pandas as pd
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# -------------------------------

# Usar LLM para sugerir par de ações para Pair Trading

# -------------------------------


# --- Configuração de Segurança ---
load_dotenv() 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def buscar_par_com_llm():
    """
    Envia um prompt para a LLM para sugerir um par de ações
    para pair trading.
    """
    print("Conectando ao assistente LLM para buscar sugestão de par...")
    
    # Este prompt é crucial. 
    # Pedimos explicitamente um formato JSON para facilitar o parsing.
    prompt_sistema = """
    Você é um analista financeiro sênior especializado em estratégias 
    market-neutral na B3 (bolsa brasileira).
    """
    
    prompt_usuario = """
    Sugira um par de ações brasileiras (B3) que sejam historicamente 
    correlacionadas e pertençam ao mesmo setor, ideais para 
    uma estratégia de pair trading.
    
    Por favor, forneça sua resposta APENAS em formato JSON, 
    com os dois tickers (incluindo o sufixo .SA).
    
    Exemplo de formato:
    {
      "par": ["TICKER1.SA", "TICKER2.SA"],
      "setor": "Nome do Setor",
      "justificativa": "Uma breve explicação."
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            response_format={"type": "json_object"}
        )
        
        # Extrai e converte a resposta string-JSON em um dicionário Python
        output_text = response.choices[0].message.content
        sugestao = json.loads(output_text)
        
        return sugestao

    except Exception as e:
        print(f"Erro ao contatar a API da OpenAI: {e}")
        return None
    

# --- Função Principal ---
if __name__ == "__main__":
    sugestao_llm = buscar_par_com_llm()
    
    if sugestao_llm and 'par' in sugestao_llm and len(sugestao_llm['par']) == 2:
        ticker1, ticker2 = sugestao_llm['par']
        
        print(f"\nSugestão recebida da LLM:")
        print(f"Par: {ticker1} e {ticker2}")
        print(f"Setor: {sugestao_llm.get('setor', 'N/A')}")
        print(f"Justificativa: {sugestao_llm.get('justificativa', 'N/A')}\n")
        
    else:
        print("Não foi possível obter uma sugestão de par válida da LLM.")