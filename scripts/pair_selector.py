import os
import json
import pandas as pd
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

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

def validar_par_estatisticamente(ticker1, ticker2, periodo="5y"):
    """
    Pega os dois tickers, baixa os dados históricos e calcula a 
    correlação dos preços de fechamento.
    """
    print(f"Validando estatisticamente o par: {ticker1} vs {ticker2}")
    
    try:
        # Baixa os dados históricos dos dois tickers de uma vez
        dados = yf.download([ticker1, ticker2], period=periodo)
        
        # Pega apenas os preços de fechamento
        fechamento = dados['Close'].dropna()
        
        if fechamento.empty:
            print("Não foi possível obter dados históricos para o par.")
            return

        # Calcula a correlação
        # .corr() em um DataFrame calcula a matriz de correlação
        correlacao = fechamento.corr().iloc[0, 1]
        
        print("\n--- Resultado da Validação ---")
        print(f"Período de análise: {periodo}")
        print(f"Correlação de Fechamento ({ticker1} vs {ticker2}): {correlacao:.4f}")
        
        if correlacao > 0.8:
            print("Resultado: ALTA CORRELAÇÃO. O par parece promissor para análise.")
        elif correlacao > 0.5:
            print("Resultado: CORRELAÇÃO MODERADA. Analisar com cautela.")
        else:
            print("Resultado: BAIXA CORRELAÇÃO. A sugestão do LLM pode não ser boa.")

    except Exception as e:
        print(f"Erro ao baixar ou processar dados do yfinance: {e}")


# --- Função Principal ---
if __name__ == "__main__":
    sugestao_llm = buscar_par_com_llm()
    
    if sugestao_llm and 'par' in sugestao_llm and len(sugestao_llm['par']) == 2:
        ticker1, ticker2 = sugestao_llm['par']
        
        print(f"\nSugestão recebida da LLM:")
        print(f"Par: {ticker1} e {ticker2}")
        print(f"Setor: {sugestao_llm.get('setor', 'N/A')}")
        print(f"Justificativa: {sugestao_llm.get('justificativa', 'N/A')}\n")
        
        # Agora, a face "quant" do Jano entra em ação:
        validar_par_estatisticamente(ticker1, ticker2)
    else:
        print("Não foi possível obter uma sugestão de par válida da LLM.")