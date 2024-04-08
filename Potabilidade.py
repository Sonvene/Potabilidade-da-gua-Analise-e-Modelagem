import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Configurações visuais
sns.set_style('darkgrid')
plt.rcParams.update({'font.size': 12})  # Ajusta o tamanho da fonte para os gráficos

def main():
    # Carrega os dados
    df = pd.read_csv('water_potability.csv')

    # Realiza análise exploratória e constrói modelos
    potabilidade(df)
    correlacao_matrix(df)
    plot_boxplot(df)
    plot_histograma(df)
    preprocessamento(df)
    preparacao_dados(df)

def potabilidade(df):
    # Plota o gráfico de contagem de potabilidade
    potabilidade_counts = df['Potability'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=potabilidade_counts.index, y=potabilidade_counts)
    plt.xlabel('Potabilidade', fontsize=15)
    plt.ylabel('Contador', fontsize=15)
    plt.title('Contagem de Potabilidade', fontsize=16)
    plt.show()

def correlacao_matrix(df):
    # Plota a matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlação', fontsize=16)
    plt.show()

def plot_boxplot(df):
    # Plota boxplots para cada variável exceto a variável de resposta 'Potability'
    for column in df.columns[:-1]:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Potability', y=column)
        plt.title(f'Boxplot de {column}', fontsize=16)
        plt.xlabel('Potabilidade', fontsize=14)
        plt.ylabel(column, fontsize=14)
        plt.show()

def plot_histograma(df):
    # Plota histogramas para cada variável exceto 'Potability'
    for feature in df.columns[:-1]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Contagem', fontsize=14)
        plt.title(f'Histograma de {feature}', fontsize=16)
        plt.show()

def preprocessamento(df):
    # Preenche valores nulos com médias condicionais baseadas na potabilidade
    for col in ['ph', 'Sulfate', 'Trihalomethanes']:
        df[col].fillna(df.groupby('Potability')[col].transform('mean'), inplace=True)

def preparacao_dados(df):
    # Prepara os dados para modelagem
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Divide os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Padroniza os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modelagem e avaliação dos modelos
    models = [
        ('Logistic Regression', LogisticRegression()),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('Gaussian Naive Bayes', GaussianNB()),
        ('Support Vector Machine', SVC())
    ]

    model_accuracies = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies.append((name, accuracy))

    # Plota a acurácia dos modelos
    df_accuracies = pd.DataFrame(model_accuracies, columns=['Modelo', 'Acurácia'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Acurácia', y='Modelo', data=df_accuracies, palette='viridis')
    plt.title('Acurácia dos Modelos', fontsize=16)
    plt.xlabel('Acurácia', fontsize=14)
    plt.ylabel('Modelo', fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
