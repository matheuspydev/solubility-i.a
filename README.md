# solubility-i.a
Meu primeiro trabalho correlacionando inteligência artificial e química.
# %% [markdown]
# Intalação das bibliotecas

# %%
%pip install pdfminer.six pandas scikit-learn matplotlib



# %% [markdown]
# Importação do dataset

# %%
import os
import pandas as pd

# Define o diretório atual como o local de execução do script
current_dir = os.getcwd()  # Pega o diretório de trabalho atual
dataset_path = os.path.join(current_dir, 'dataset.csv')  # Substitua pelo nome do dataset

# Carregar o dataset existente
if os.path.exists(dataset_path):
    dataset = pd.read_csv(dataset_path)
    print("Dataset carregado com sucesso.")
else:
    raise FileNotFoundError(f"Arquivo {dataset_path} não encontrado. Verifique o caminho e tente novamente.")

# Exibir as primeiras linhas para confirmação
print(dataset.head())

# Operações adicionais (se necessário)
# Exemplo: Adicionar novos dados ao dataset
new_data = [
    ["AgCl", "Ag +", "Cl -", 9.75, 1, 1.12, 13.32],
    ["PbI2", "Pb 2+", "I -", 8.13, 2, 1.33, 14.52],
]  # Adicione aqui novos dados manualmente

new_data_df = pd.DataFrame(new_data, columns=["COMPOSTO", "CATION", "ANION", "pKs", "CARGA", "RAIO_IONICO", "E_IONIZACAO"])

# Evitar duplicatas ao combinar os datasets
updated_dataset = pd.concat([dataset, new_data_df]).drop_duplicates().reset_index(drop=True)

# Salvar o dataset atualizado
updated_dataset.to_csv(os.path.join(current_dir, 'updated_dataset.csv'), index=False)
print("Dataset atualizado salvo como 'updated_dataset.csv'")


# %% [markdown]
# Adição da randon forest e Rede neural

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score

def configure_models():
    """
    Configura e retorna um RandomForestRegressor e um MLPRegressor configuráveis.
    """
    # Configuração do Random Forest
    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )
    
    # Configuração inicial do MLP Regressor
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(150, 100, 50),
        max_iter=1,  # Número inicial baixo para controle manual das épocas
        warm_start=True,  # Permitir continuar o treinamento em múltiplas iterações
        learning_rate_init=0.01,
        alpha=0.001,
        random_state=42
    )
    
    return random_forest, mlp_regressor

def train_mlp(mlp_model, X_train, y_train, X_test, y_test, epochs):
    """
    Treina o MLPRegressor por um número configurável de épocas, mostrando o progresso.
    """
    for epoch in range(1, epochs + 1):
        mlp_model.fit(X_train, y_train)
        train_pred = mlp_model.predict(X_train)
        test_pred = mlp_model.predict(X_test)
        
        # Cálculo das métricas em cada época
        mae = mean_absolute_error(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred)
        
        print(f"Época {epoch}/{epochs}")
        print(f"  -> MAE: {mae:.4f}, MSE: {mse:.4f}")
        
    return mlp_model

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo em várias métricas para problemas de regressão.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print("Avaliação do Modelo:")
    print(f"  -> MAE: {mae:.4f}")
    print(f"  -> MSE: {mse:.4f}")
    
    return mae, mse


# %%

from sklearn.model_selection import train_test_split

# Substitua pelo seu DataFrame com as colunas corretas
# Exemplo de colunas: ["CARGA", "RAIO_IONICO", "E_IONIZACAO", "pKs"]
X = dataset.drop(columns=["pKs"])  # Remove a variável dependente
y = dataset["pKs"]  # Variável dependente

# Divisão inicial em treino e temporário (validação + teste)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Divisão do temporário em validação e teste
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Exibir os tamanhos dos conjuntos
print(f"Tamanho do conjunto de treino: {len(X_train)}")
print(f"Tamanho do conjunto de validação: {len(X_val)}")
print(f"Tamanho do conjunto de teste: {len(X_test)}")


# %%
import re

# Função para contar o número de átomos de um composto
def count_atoms(compound):
    return sum(int(num) if num else 1 for _, num in re.findall(r"([A-Z][a-z]*)(\d*)", compound))

# Adicionar nova coluna com o número de átomos
dataset["NUM_ATOMS"] = dataset["COMPOSTO"].apply(count_atoms)

# Atualizar X para incluir a nova coluna
X = dataset[["NUM_ATOMS", "CARGA", "RAIO_IONICO", "E_IONIZACAO"]]
y = dataset["pKs"]



# %%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

random_forest.fit(X_train, y_train)
evaluate_model(random_forest, X_test, y_test)

train_mlp(mlp_regressor, X_train, y_train, X_test, y_test, epochs=50)
evaluate_model(mlp_regressor, X_test, y_test)


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import numpy as np

# 1. Pré-processamento dos Dados
# Substitua pelas colunas relevantes no seu dataset
X = dataset[["CARGA", "RAIO_IONICO", "E_IONIZACAO"]]
y = dataset["COMPOSTO_ENCODED"]  # Codifique "COMPOSTO" antes de usar

# Dividir em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Treinamento e Avaliação dos Modelos
# Random Forest
random_forest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
random_forest.fit(X_train, y_train)
rf_predictions = random_forest.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# MLP Classifier
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(150, 100, 50),
    max_iter=5000,
    random_state=42,
    learning_rate_init=0.01
)
mlp_classifier.fit(X_train, y_train)
mlp_predictions = mlp_classifier.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

# 3. Visualização da Curva de Aprendizado
def plot_learning_curve(model, X_train, y_train):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = int(size * len(X_train))
        X_subset = X_train[:subset]
        y_subset = y_train[:subset]
        model.fit(X_subset, y_subset)
        train_scores.append(model.score(X_subset, y_subset))
        val_scores.append(model.score(X_val, y_val))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label="Treino", marker="o")
    plt.plot(train_sizes, val_scores, label="Validação", marker="o")
    plt.title("Curva de Aprendizado")
    plt.xlabel("Tamanho do Conjunto de Treinamento")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid()
    plt.show()

# Plotar a curva de aprendizado para o Random Forest
plot_learning_curve(random_forest, X_train.to_numpy(), y_train.to_numpy())

# 4. Matriz de Confusão
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, rf_predictions, "Random Forest - Matriz de Confusão")
plot_confusion_matrix(y_test, mlp_predictions, "MLP Classifier - Matriz de Confusão")

# 5. Resultados
print("Resultados Random Forest:")
print(f"Acurácia: {rf_accuracy:.4f}")

print("Resultados MLP Classifier:")
print(f"Acurácia: {mlp_accuracy:.4f}")


# %%
import matplotlib.pyplot as plt

# Comparação de pKs Real vs Predito para Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, random_forest.predict(X_test), alpha=0.7, label="Random Forest")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Linha Ideal")
plt.title("pKs Predito vs Real - Random Forest", fontsize=14)
plt.xlabel("pKs Real", fontsize=12)
plt.ylabel("pKs Predito", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Comparação de pKs Real vs Predito para MLP
plt.figure(figsize=(8, 6))
plt.scatter(y_test, mlp_regressor.predict(X_test), alpha=0.7, label="MLP Regressor")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Linha Ideal")
plt.title("pKs Predito vs Real - MLP Regressor", fontsize=14)
plt.xlabel("pKs Real", fontsize=12)
plt.ylabel("pKs Predito", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
