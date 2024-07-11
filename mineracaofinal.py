import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Lista de 70 itens
itens = [
    "leite", "pão", "arroz", "feijão", "açúcar", "sal", "café", "óleo", "manteiga", "queijo",
    "presunto", "mortadela", "iogurte", "suco", "refrigerante", "cerveja", "vinho", "água", 
    "chocolate", "biscoito", "bolacha", "sabonete", "shampoo", "condicionador", "pasta de dente", 
    "papel higiênico", "frango", "carne", "peixe", "ovos", "cenoura", "batata", "tomate", 
    "alface", "maçã", "banana", "laranja", "limão", "melancia", "abacaxi", "uva", "manga", 
    "morango", "abacate", "cebola", "alho", "pimentão", "pepino", "abobrinha", "berinjela", 
    "brócolis", "couve-flor", "espinafre", "almeirão", "chuchu", "beterraba", "mandioca", 
    "aipim", "farinha", "fermento", "leite condensado", "creme de leite", "molho de tomate", 
    "maionese", "ketchup", "mostarda", "azeite", "vinagre", "canela", "pimenta", "orégano"
]

# Gerar transações aleatórias
num_transacoes = 1000
transacoes = []

for _ in range(num_transacoes):
    num_itens = 6
    transacao = np.random.choice(itens, num_itens, replace=False).tolist()
    transacoes.append(transacao)

# Dividir os dados em conjunto de treinamento e teste (80-20)
train_data, test_data = train_test_split(transacoes, test_size=0.2, random_state=42)

# Converter para DataFrame
te = TransactionEncoder()
te_ary_train = te.fit(train_data).transform(train_data)
df_train = pd.DataFrame(te_ary_train, columns=te.columns_)

# Algoritmo Apriori no conjunto de treinamento
conjuntos_frequentes = apriori(df_train, min_support=0.01, use_colnames=True)

# Regras de associação
regras = association_rules(conjuntos_frequentes, metric="lift", min_threshold=1.0)

# Função para encontrar recomendações com base em uma transação
def recomendar(transacao, regras, max_recomendacoes=6):
    recomendacoes = set()
    for index, row in regras.iterrows():
        antecedente = set(row['antecedents'])
        consequente = set(row['consequents'])
        if antecedente.issubset(transacao):
            recomendacoes = recomendacoes.union(consequente)
    recomendacoes = recomendacoes - set(transacao)
    return list(recomendacoes)[:max_recomendacoes]

# Avaliar a precisão das recomendações no conjunto de teste
def avaliar_recomendacoes(test_data, regras):
    acertos = 0
    total_recomendacoes = 0

    for transacao in test_data:
        recomendacoes = recomendar(transacao, regras)
        if recomendacoes:
            total_recomendacoes += 1
            for item in recomendacoes:
                if item in transacao:
                    acertos += 1

    return acertos / total_recomendacoes if total_recomendacoes else 0

precisao = avaliar_recomendacoes(test_data, regras)
print(f'Precisão das recomendações: {precisao:.2f}')

# Função para gerar gráfico de coocorrência de itens
def plot_coocorrencia(regras, item):
    if 'consequents' not in regras.columns:
        print(f"Coluna 'consequents' não encontrada em regras")
        return None
    regras_item = regras[regras['antecedents'].apply(lambda x: item in x)]
    if regras_item.empty:
        print(f"Sem regras para o item: {item}")
        return None
    consequentes = regras_item['consequents'].apply(list).explode()
    frequencia = consequentes.value_counts(normalize=True) * 100
    frequencia = frequencia[frequencia.index != item] 

    if frequencia.empty:
        print(f"Nenhum item frequente encontrado para o item: {item}")
        return None

    item_mais_frequente = frequencia.idxmax()
    porcentagem = frequencia.max()

    plt.figure(figsize=(10, 6))
    plt.bar([item_mais_frequente], [porcentagem])
    plt.xlabel('Item Recomendado')
    plt.ylabel('Porcentagem de Coocorrência (%)')
    plt.title(f'Item mais Recomendado para quem compra "{item}"')
    plt.xticks(rotation=90)
    plt.show()

    return item_mais_frequente

# Função para visualização das recomendações
def plot_recomendacoes(transacao, recomendacoes):
    fig, ax = plt.subplots(figsize=(10, 6))
    transacao_str = ', '.join(transacao)
    recomendacoes_str = ', '.join(recomendacoes) if recomendacoes else 'Nenhuma recomendação'
    ax.text(0.5, 0.8, f'Transação: {transacao_str}', ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.5, f'Recomendações: {recomendacoes_str}', ha='center', va='center', fontsize=12)
    ax.axis('off')
    plt.title('Recomendações de Compra')
    plt.show()

# Exibir gráfico de quantidade de compras
def plot_quantidade_compras(transacoes):
    item_counts = pd.Series([item for transacao in transacoes for item in transacao]).value_counts()
    item_counts_percentage = (item_counts / len(transacoes)) * 100

    plt.figure(figsize=(15, 8))
    plt.bar(item_counts.index, item_counts.values, color='blue', alpha=0.7)
    for i, (count, pct) in enumerate(zip(item_counts.values, item_counts_percentage.values)):
        plt.text(i, count, f'{count}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=8)
    plt.xlabel('Itens')
    plt.ylabel('Quantidade de Compras')
    plt.title('Quantidade de Vezes que Cada Item Foi Comprado')
    plt.xticks(rotation=90)
    plt.show()

plot_quantidade_compras(transacoes)

# Selecionar uma transação aleatória do conjunto de teste
transacao_aleatoria = random.choice(test_data)
print("Transação aleatória:", transacao_aleatoria)

# Gerar gráficos de coocorrência para cada item na transação aleatória
recomendacoes = []
for item in transacao_aleatoria:
    recomendacao = plot_coocorrencia(regras, item)
    if recomendacao:
        recomendacoes.append(recomendacao)

# Remover duplicatas das recomendações
recomendacoes = list(set(recomendacoes))

# Exibir a transação e as recomendações
plot_recomendacoes(transacao_aleatoria, recomendacoes)

# Função para inserir uma transação de teste e gerar recomendações
def inserir_transacao_e_recomendar():
    transacao_input = input("Insira uma transação (itens separados por vírgula): ")
    transacao = [item.strip() for item in transacao_input.split(',')]

    print("Transação inserida:", transacao)

    recomendacoes = []
    for item in transacao:
        recomendacao = plot_coocorrencia(regras, item)
        if recomendacao:
            recomendacoes.append(recomendacao)

    # Remover duplicatas das recomendações
    recomendacoes = list(set(recomendacoes))

    # Exibir a transação e as recomendações
    plot_recomendacoes(transacao, recomendacoes)

# Chamar a função para inserir uma transação e gerar recomendações
inserir_transacao_e_recomendar()
