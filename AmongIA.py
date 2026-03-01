import logging
import torch
import re
import numpy as np
import random
import os
from tqdm import tqdm
import language_tool_python
from sentence_transformers import SentenceTransformer, util
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForMaskedLM,
)

logger = logging.getLogger(__name__)

# Parâmetros de configuração do DetectGPT.
CONFIG = {
    "GENERATOR_MODEL_ID": "Qwen/Qwen2.5-7B-Instruct",  # modelo alternativo: mistralai/Mistral-7B-Instruct-v0.2
    "PERTURBATION_MODEL_ID": "neuralmind/bert-base-portuguese-cased",
    "MIN_TEXT_LENGTH": 1000,
    "PERTURBATION_COUNT": 50,
    "MASKING_RATE": 0.08, 
}

def setup_logging():
    """Configura o sistema de logging do DetectGPT"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
    
    # Evita logs duplicados
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('DetectGPT.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def validate_environment():
    """Verifica se o ambiente de execucao (PyTorch, CUDA, bitsandbytes) esta configurado corretamente."""
    logger.info("Iniciando validacao do ambiente...")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch nao detectou uma GPU com CUDA.")
        logger.info(f"PyTorch com CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"bitsandbytes importado com sucesso.")
        test_q_config = BitsAndBytesConfig(load_in_4bit=True)
        _ = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", quantization_config=test_q_config)
        logger.info("Teste de carregamento de modelo quantizado bem-sucedido.")
        logger.info("Validacao do ambiente concluida com sucesso.")
        return True
    except Exception as e:
        logger.exception("FALHA na validacao do ambiente.")
        return False

def load_generator_model(model_id):
    """Carrega o modelo gerador quantizado."""
    logger.info(f"Carregando modelo gerador '{model_id}' com quantizacao...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config,
        device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Alguns modelos como Qwen precisam de um pad_token definido
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info("Modelo gerador carregado.")
    return model, tokenizer

def load_perturbation_model(model_id):
    """Carrega o modelo de perturbacao baseado em BERT."""
    logger.info(f"Carregando modelo de perturbacao BERT '{model_id}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    logger.info(f"Modelo de perturbacao pronto no dispositivo: {device}.")
    return model, tokenizer

def perturb_text(text, p_model, p_tokenizer, config, debug=False):
    """Gera múltiplas perturbações evitando subwords e erros de tokenização do BERT."""
    words = text.split()
    if not words: return [text] * config["PERTURBATION_COUNT"]
    num_words_to_mask = int(len(words) * config["MASKING_RATE"])
    if num_words_to_mask == 0: num_words_to_mask = 1
    
    perturbations = []
    desc = "Gerando Perturbacoes (BERT)" if not debug else None
    
    for i in tqdm(range(config["PERTURBATION_COUNT"]), desc=desc, disable=debug):
        mask_indices = sorted(random.sample(range(len(words)), num_words_to_mask))
        temp_words = list(words)
        
        for word_idx in mask_indices:
            original_word = temp_words[word_idx]
            
            match = re.match(r'^([^\w]*)(.*?)([^\w]*)$', original_word, re.UNICODE)
            prefix, core, suffix = ("", original_word, "")
            if match:
                prefix, core, suffix = match.groups()
                
            if not core:
                continue

            temp_words[word_idx] = f"{prefix}{p_tokenizer.mask_token}{suffix}"
            masked_text = " ".join(temp_words)
            
            with torch.no_grad():
                inputs = p_tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512).to(p_model.device)
                outputs = p_model(**inputs)
                
                mask_token_indices = torch.where(inputs["input_ids"] == p_tokenizer.mask_token_id)[1]
                
                if mask_token_indices.nelement() == 0:
                    predicted_word = core
                else:
                    mask_pos = mask_token_indices[0]
                    logits = outputs.logits[0, mask_pos]
                    
                    # Pega as top 15 opções para ter folga ao filtrar sujeira
                    top_k_logits, top_k_indices = torch.topk(logits, 15)
                    
                    valid_indices = []
                    valid_logits = []
                    
                    for logit, idx in zip(top_k_logits, top_k_indices):
                        token_str = p_tokenizer.convert_ids_to_tokens(idx.item())
                        
                        # Filtra pedaços de palavras (##), tokens especiais e exige que sejam letras (\w+)
                        if not token_str.startswith("##") and token_str not in p_tokenizer.all_special_tokens:
                            if re.match(r'^\w+$', token_str, re.UNICODE):
                                valid_indices.append(idx)
                                valid_logits.append(logit)
                    
                    if not valid_indices:
                        predicted_word = core
                    else:
                        # Seleciona apenas das top 5 opções VÁLIDAS restantes
                        valid_indices = torch.tensor(valid_indices[:5])
                        valid_logits = torch.tensor(valid_logits[:5])
                        
                        chosen_idx = valid_indices[torch.multinomial(torch.nn.functional.softmax(valid_logits, dim=-1), 1).item()].item()
                        predicted_word = p_tokenizer.decode([chosen_idx]).replace(" ", "").strip()
            
            temp_words[word_idx] = f"{prefix}{predicted_word}{suffix}"

        final_perturbed_text = " ".join(temp_words)
        perturbations.append(final_perturbed_text)

    return perturbations

def calculate_burstiness(text):
    """Calcula a 'Burstiness' de um texto baseada no desvio padrão do tamanho das frases. Quanto maior o valor, mais 'humano o texto tende a ser"""
    # Divide o texto em frases usando pontuação como barreira.
    phrases = re.split(r'[.!?]+', text)
    
    # Limpa espaços em branco e filtra frases vazias.
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    
    if not phrases:
        return 0.0
    
    # Conta o número de palavras por frase.
    phrases_lengths = [len(phrase.split()) for phrase in phrases]
    
    if len(phrases_lengths) < 2:
        return 0.0
    
    # Calcula o desvio padrão dos tamanhos das frases
    burstiness = np.std(phrases_lengths, ddof=1)
    
    return float(burstiness)

def calculate_error_rate(text, tool):
    """Calcula a taxa de erros gramaticais/ortográficos por palavra a cada 100 palavras. Quanto maior, mais 'humano' o texto tende a ser."""
    errors = tool.check(text)
    error_count = len(errors)
    word_count = len(text.split())
    
    if word_count == 0:
        return 0.0
    
    error_rate = (error_count / word_count) * 100
    return float(error_rate)

def calculate_semantic_cohesion(text, model):
    """Calcula a similaridade média entre frases consecutivas. IAs tendem a ter uma coesão mais alta (>.40) enquanto humanos tendem a ter saltos lógicos (coesão menor)."""
    phrases = re.split(r'[.!?]+', text)
    phrases = [f.strip() for f in phrases if len(f.strip().split()) > 3]
    
    if len(phrases) < 2:
        return 0.0
    
    embeddings = model.encode(phrases)
    
    similarities = []
    
    for i in range(len(embeddings) - 1):
        sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
        similarities.append(sim)
    
    cohesion_score = float(np.mean(similarities))
    return cohesion_score
    
def calculate_log_prob(text, model, tokenizer):
    """Calcula a log-probabilidade media de um texto sob um determinado modelo."""
    if not text: return -100.0
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
        input_ids = inputs.input_ids
        if input_ids.shape[1] == 0: return -100.0
        outputs = model(input_ids, labels=input_ids)
        return -outputs.loss.item()

def _calculate_scores_for_text(text, models, config, debug=False):
    """Funcao auxiliar para calcular o score DetectGPT completo para um unico texto."""
    g_model, g_tokenizer, p_model, p_tokenizer = models
    original_log_prob = calculate_log_prob(text, g_model, g_tokenizer)
    
    perturbations = perturb_text(text, p_model, p_tokenizer, config, debug=debug)
    
    perturbed_log_probs = [calculate_log_prob(p, g_model, g_tokenizer) for p in tqdm(perturbations, desc="Avaliando Perturbacoes (com o modelo gerador) ", disable=False)]
    
    mean_perturbed = np.mean(perturbed_log_probs)
    std_perturbed = np.std(perturbed_log_probs)
    score = (original_log_prob - mean_perturbed) / std_perturbed if std_perturbed > 1e-6 else 0.0
    return score, original_log_prob, mean_perturbed

def calculate_probability_fuzzy(score_z, score_burstiness, score_error_rate, score_cohesion):
    """
    Sistema de Inferência Fuzzy corrigido para evitar erros de atributo e limites.
    """
    
    # REGRA ESPECIALISTA: Se o modelo aponta um valor extremamente alto no Z-Score (>3.8) isso é um indicador certo de IA, devido a natureza do DetectGPT. Essa regra tem precedência máxima.
    if score_z > 3.8:
        return 95.0
    
    # 1. Definição do Universo (Antecedents e Consequent)
    # Entradas
    z_score = ctrl.Antecedent(np.arange(0, 7.1, 0.1), 'z_score')
    burstiness = ctrl.Antecedent(np.arange(0, 25.1, 0.1), 'burstiness')
    error_rate = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'error_rate')
    cohesion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cohesion')
    # Saída
    probability = ctrl.Consequent(np.arange(0, 101, 1), 'probability')

    # 2. Funções de Pertinência
    # Entradas: Z-Score, Burstiness, Taxa de Erros e Coesão Semântica
    # Z-Score
    z_score['low'] = fuzz.trapmf(z_score.universe, [0, 0, 2.2, 2.5])
    z_score['medium'] = fuzz.trimf(z_score.universe, [2.4, 2.65, 2.9])
    z_score['high'] = fuzz.trapmf(z_score.universe, [2.8, 3.1, 7.0, 7.0])

    # Burstiness
    burstiness['low'] = fuzz.trapmf(burstiness.universe, [0, 0, 8.0, 9.5])
    burstiness['medium'] = fuzz.trimf(burstiness.universe, [9.0, 10.5, 11.5])
    burstiness['high'] = fuzz.trapmf(burstiness.universe, [11.0, 12.5, 25.0, 25.0])

    # Taxa de Erros Gramaticais e Ortográficos
    error_rate['low'] = fuzz.trapmf(error_rate.universe, [0, 0, 0.5, 1.2])
    error_rate['medium'] = fuzz.trimf(error_rate.universe, [0.8, 1.5, 2.5])
    error_rate['high'] = fuzz.trapmf(error_rate.universe, [2.0, 3.0, 5.0, 5.0])

    # Coesão Semântica
    cohesion['low'] = fuzz.trapmf(cohesion.universe, [0, 0, 0.30, 0.35])
    cohesion['medium'] = fuzz.trimf(cohesion.universe, [0.33, 0.39, 0.45])
    cohesion['high'] = fuzz.trapmf(cohesion.universe, [0.42, 0.48, 1.0, 1.0])
    
    # Saída: Probabilidade de ser IA
    probability['low'] = fuzz.trapmf(probability.universe, [0, 0, 30, 45])
    probability['medium'] = fuzz.trimf(probability.universe, [35, 50, 65])
    probability['high'] = fuzz.trapmf(probability.universe, [55, 75, 100, 100])

    # 3. Regras Lógicas
    # -- Regras baseadas em observações empíricas e intuição sobre o comportamento de textos humanos vs IA --
    
    # 3. Regras Lógicas
    rule0 = ctrl.Rule(z_score['high'], probability['high'])
    rule1 = ctrl.Rule(z_score['medium'], probability['medium'])
    rule2 = ctrl.Rule(z_score['low'], probability['low'])
    rule3 = ctrl.Rule(z_score['high'] & burstiness['low'], probability['high'])
    rule4 = ctrl.Rule(z_score['medium'] & burstiness['low'] & cohesion['high'], probability['high'])
    rule5 = ctrl.Rule(burstiness['high'], probability['low'])
    rule6 = ctrl.Rule(error_rate['high'] & cohesion['low'], probability['low'])
    rule7 = ctrl.Rule(z_score['low'] & burstiness['high'] & error_rate['high'], probability['low'])
    rule8 = ctrl.Rule(z_score['low'] & cohesion['high'] & burstiness['medium'], probability['medium'])
    rule9 = ctrl.Rule(z_score['high'] & burstiness['high'], probability['medium'])
    rule10 = ctrl.Rule(z_score['medium'] & burstiness['medium'] & error_rate['medium'] & cohesion['medium'], probability['medium'])
    rule11 = ctrl.Rule(burstiness['low'] & error_rate['low'], probability['high'])


    # 4. Simulação 
    control_sys = ctrl.ControlSystem([rule0, rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
    simulation = ctrl.ControlSystemSimulation(control_sys)

    simulation.input['z_score'] = np.clip(score_z, 0, 7.0)
    simulation.input['burstiness'] = np.clip(score_burstiness, 0, 25.0)
    simulation.input['error_rate'] = np.clip(score_error_rate, 0, 5.0)
    simulation.input['cohesion'] = np.clip(score_cohesion, 0, 1.0)

    simulation.compute()
    
    return simulation.output['probability']
    

def classify_text_file(file_path, models, config, language_tool, sentence_model):
    """
    Lê um arquivo .txt e retorna a classificação humano ou IA através da Lógica Fuzzy.
    """
    if not os.path.exists(file_path):
        print(f"\n[ERRO] O arquivo '{file_path}' não foi encontrado.")
        print("Por favor, certifique-se de fazer o upload ou criar o arquivo no Colab antes de rodar.\n")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception as e:
        print(f"\n[ERRO] Não foi possível ler o arquivo: {e}\n")
        return
    
    # Verifica se o texto tem tamanho mínimo para análise.
    if len(text) < config["MIN_TEXT_LENGTH"]:
        print(f"\nTexto muito curto ({len(text)} caracteres). O AmongIA precisa de pelo menos {config['MIN_TEXT_LENGTH']} caracteres para uma análise confiável.\n")
        return
    
    # Verifica se o texto tem um número mínimo de frases válidas para garantir que as métricas de Burstiness e Coesão sejam significativas.
    phrases = re.split(r'[.!?]+', text)
    valid_phrases = [p.strip() for p in phrases if len(p.strip().split()) > 3]
    if len(valid_phrases) < 3:
        print(f"\nTexto com poucas frases válidas ({len(valid_phrases)}).\n")
        return
    
    print("\n" + "="*55)
    print(" INICIANDO A ANÁLISE MULTIDIMENSIONAL DO AmongIA ")
    print("="*55)
    
    # 1. Z-Score / DetectGPT (A matemática original)
    print("[1/5] Calculando probabilidade de IA (DetectGPT)...")
    score_ia, _, _ = _calculate_scores_for_text(text, models, config)
    
    # 2. Burstiness
    print("[2/5] Calculando Burstiness (variação no tamanho das frases)...")
    score_burstiness = calculate_burstiness(text)
    
    # 3. Taxa de Erros
    print("[3/5] Calculando taxa de erros (gramática e ortografia)...")
    score_erros = calculate_error_rate(text, language_tool)
    
    # 4. Coesão Semântica
    print("[4/5] Calculando coesão semântica (similaridade entre frases)...")
    score_coesao = calculate_semantic_cohesion(text, sentence_model)
    
    # 5. Lógica Fuzzy
    print("[5/5] Processando lógica Fuzzy (veredito final)...")
    try:
        probabilidade_final = calculate_probability_fuzzy(
            score_ia, score_burstiness, score_erros, score_coesao
        )
    except Exception as e:
        print(f"\n[AVISO] Falha ao processar a lógica Fuzzy: {e}")
        probabilidade_final = 50.0
    
    # Classificação final baseada na probabilidade calculada pela lógica Fuzzy
    if probabilidade_final <= 35.0:
        classificacao = "Texto muito provavelmente escrito por humano. 👤"
    elif probabilidade_final > 35.0 and probabilidade_final <= 50.0:
        classificacao = "Texto provavelmente gerado por humano, mas com características de IA. 👤🤖"
    elif probabilidade_final > 50.0 and probabilidade_final <= 70.0:
        classificacao = "Texto provavelmente gerado por IA, mas com características humanas. 🤖👤"
    elif probabilidade_final > 70.0:
        classificacao = "Texto muito provavelmente gerado por IA. 🤖"
    
    # --- EXIBIÇÃO E EXPORTAÇÃO DO PAINEL DE RESULTADOS ---
    laudo = f"""
    {"="*55}
                      LAUDO FINAL AMONGIA                  
    {"="*55}
    ARQUIVO: {os.path.basename(file_path)}
    {"="*55}
    MÉTRICAS ISOLADAS:
    - Z-Score (DetectGPT) : {score_ia:.4f}
    - Burstiness (Ritmo)  : {score_burstiness:.4f}
    - Erros (por 100 pal.): {score_erros:.2f}
    - Coesão Semântica    : {score_coesao:.4f}
    {"="*55}
    VEREDITO FUZZY: {probabilidade_final:.1f}% de chance de ser IA.
    CLASSIFICAÇÃO FINAL: {classificacao}
    {"="*55}
    """
    
    print(f"\n{laudo}\n")
    
    # Salva o resultado em um arquivo de texto para análise
    base_name = os.path.splitext(file_path)[0]
    output_file = f"{base_name}Result.txt"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(laudo)
        print(f"[SUCESSO] Laudo salvo em: {output_file}\n")
    except Exception as e:
        print(f"\n[AVISO] Não foi possível salvar o arquivo de laudo: {e}\n")

def process_directory(directory_path, models, config, language_tool, sentence_model):
    """Processa todos os arquivos .txt em um diretório e gera laudos individuais para cada um."""
    path = Path(directory_path)
    txt_files = [f for f in path.rglob ("*.txt") if not f.name.endswith("Result.txt")]
                     
    if not txt_files:
        print(f"\n[AVISO] Nenhum arquivo .txt encontrado no diretório: {directory_path}\n")
        return
    
    print(f"\n[INÍCIO] Processando {len(txt_files)} arquivos no diretório: {directory_path}\n")
    
    for txt_file in txt_files:
        print(f"Lendo arquivo: {txt_file.name}")
        classify_text_file(str(txt_file), models, config, language_tool, sentence_model)
    
def main():
    setup_logging() 
    validate_environment()

    print("\n[INICIALIZAÇÃO 1/2] Carregando modelos de linguagem (Gerador e Perturbação)...")
    models = (
        *load_generator_model(CONFIG["GENERATOR_MODEL_ID"]),
        *load_perturbation_model(CONFIG["PERTURBATION_MODEL_ID"])
    )
    
    print("[INICIALIZAÇÃO 2/2] Carregando ferramentas de análise (LanguageTool e SentenceTransformer)...")
    language_tool = language_tool_python.LanguageTool('pt-BR')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    text_folder = "textos_analise"  
    
    # Se a pasta não existir, criamos ela
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)
        print(f"\n[AVISO] O diretório '{text_folder}' não existia e foi criado. Por favor, adicione os arquivos .txt para análise e execute novamente.\n")
        return
    else:
        process_directory(text_folder, models, CONFIG, language_tool, sentence_model)
    
    language_tool.close()
    
if __name__ == "__main__":
    main()