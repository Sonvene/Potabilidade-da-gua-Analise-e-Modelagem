import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sns.set_style('darkgrid')
warnings.filterwarnings('ignore')

df = pd.read_csv('water_potability.csv')

# Análise exploratória
pot_lbl = df.Potability.value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=pot_lbl.index, y=pot_lbl)
plt.xlabel('Potabilidade', fontsize=15)
plt.ylabel('Contador', fontsize=15)
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlação Matrix', fontsize=20)
plt.show()

for column in df.columns[:-1]:
    plt.figure(figsize=(8, 5))
    sns.boxplot(df[column])
    plt.title('Box plot of {}'.format(column), fontsize=20)
plt.show()

for feature in df.columns[:-1]:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True)
    plt.xlabel(feature, fontsize=20)
    plt.ylabel('count', fontsize=20)
    plt.title('Histograma de {}'.format(feature), fontsize=20)
plt.show()

# Pré-processamento
for feature in df.columns:
    print('{} \t {:.1f}% valores ausentes'.format(feature, (df[feature].isnull().sum() / len(df)) * 100))

ph_man = df[df['Potability'] == 0]['ph'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['ph'].isna()), 'ph'] = ph_man

ph_man_1 = df[df['Potability'] == 1]['ph'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['ph'].isna()), 'ph'] = ph_man_1

sulf_man = df[df['Potability'] == 0]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Sulfate'].isna()), 'Sulfate'] = sulf_man

sulf_man_1 = df[df['Potability'] == 1]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Sulfate'].isna()), 'Sulfate'] = sulf_man_1

traih_man = df[df['Potability'] == 0]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = traih_man

traih_man_1 = df[df['Potability'] == 1]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = traih_man_1

# Preparação dos dados
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Construção e avaliação dos modelos
models_acc = []
models = [LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier(), GaussianNB(), SVC()]

for model in models:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    models_acc.append(accuracy_score(y_test, pred))

res = pd.DataFrame({
    'model accuracy': models_acc,
    'model name': ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'GaussianNB', 'SVC']
})

plt.figure(figsize=(10, 5))
sns.barplot(x='model accuracy', y='model name', data=res)
plt.xlabel('Acurácia do Modelo', fontsize=15)
plt.ylabel('Nome do Modelo', fontsize=15)
plt.show()
