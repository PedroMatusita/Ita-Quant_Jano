import json
from google import genai
from dotenv import load_dotenv

# -------------------------------

# Usar LLM para sugerir par de ações para Pair Trading

# -------------------------------


# --- Configuração de Segurança ---
load_dotenv() 
client = genai.Client()


def buscar_par_com_llm_gemini(chat, rerun: bool):
    """
    Envia um prompt para a LLM (Gemini) para sugerir um par de ações
    para pair trading na B3, com saída no formato JSON.

    Args:
        client: Uma instância do cliente 'genai.Client'.
    """
    print("----------------------------------------------------------------------")
    print("Conectando ao assistente LLM (Gemini) para buscar sugestão de par...")

    # A API do Gemini usa a 'configuração do sistema' e o 'schema de resposta'

    if (rerun is False):
        # 3. Prompt do Usuário
        prompt_usuario = """
        Sugira um par de ações brasileiras (B3) que sejam historicamente
        correlacionadas e pertençam ao mesmo setor, ideais para
        uma estratégia de pair trading.
        """
    else:
        prompt_usuario = """
        As ações sugeridas anteriormente não são COINTEGRADAS. 
        Sugira um novo par de ações que sejam historicamente
        correlacionadas e pertençam ao mesmo setor, ideais para
        uma estratégia de pair trading.
        """

    try:
        response = chat.send_message(prompt_usuario)

        # A resposta (response.text) já é uma string JSON válida,
        # aderindo ao schema
        output_text = response.text
        sugestao = json.loads(output_text)

        return sugestao

    except Exception as e:
        print(f"Erro ao contatar a API do Gemini: {e}")
        return None
    

# --- Função Principal ---
if __name__ == "__main__":
    sugestao_llm = buscar_par_com_llm_gemini(client)
    
    if sugestao_llm and 'par' in sugestao_llm and len(sugestao_llm['par']) == 2:
        ticker1, ticker2 = sugestao_llm['par']
        
        print(f"\nSugestão recebida da LLM:")
        print(f"Par: {ticker1} e {ticker2}")
        print(f"Setor: {sugestao_llm.get('setor', 'N/A')}")
        print(f"Justificativa: {sugestao_llm.get('justificativa', 'N/A')}\n")
        
    else:
        print("Não foi possível obter uma sugestão de par válida da LLM.")