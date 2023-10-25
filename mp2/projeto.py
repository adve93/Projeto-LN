import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('treebank')
from nltk.corpus import treebank

# guardar conteudo do ficheiro train.txt exceto as palavras "TRUTHFULPOSITIVE", "TRUTHFULNEGATIVE", "DECEPTIVEPOSITIVE" e "DECEPTIVENEGATIVE"
f = open("train.txt", "r")
train = f.read()
f.close()

train = train.replace("TRUTHFULPOSITIVE", "")
train = train.replace("TRUTHFULNEGATIVE", "")
train = train.replace("DECEPTIVEPOSITIVE", "")
train = train.replace("DECEPTIVENEGATIVE", "")

# tokenizar o texto
def tokenize():
    tokens = nltk.word_tokenize(train)
    return tokens

# 10 palavras mais frequentes
def most_frequent(t):
    freq = nltk.FreqDist(t)
    print(freq.most_common(10))

# desenhar a árvore
def draw():
    t = treebank.parsed_sents('wsj_0001.mrg')[0]
    t.draw()

# create instances
# recebe o numero de timesteps
# cria as instancias e guarda num ficheiro
def create_instances(t):
    # ler ficheiro
    tempo = 20 * t
    with open("train.txt", "r") as f:
        lines = f.readlines()

    # criar dicionário de classificações
    classifications = {}
    for line in lines:
        classification, text = line.strip().split("\t", 1)
        if classification not in classifications:
            classifications[classification] = len(classifications)

    # fazer as instâncias
    instances = []
    for i in range(len(lines) - tempo):
        current_line = lines[i].strip()
        classification, text = current_line.split("\t", 1)
        instance_text = text

        for j in range(1, tempo + 1):
            next_line = lines[i + j].strip()
            next_classification, next_text = next_line.split(" ", 1)

            if next_classification == classification:
                instance_text += " " + " ".join(next_text.split()[-3:])

        instances.append(f"{instance_text}")

    # escrever no ficheiro
    with open("instances.txt", "w") as f:
        for instance in instances:
            f.write(instance + "\n")


# create k fold sets
# recebe o numero de fold sets
# cria os k fold sets com ids diferentes e forma logo os ficheros de treino e teste para cada fold set
def create_k_fold_sets(k):
    # criar k fold sets
    k_fold_sets = []
    with open("instances.txt", "r") as f:
        lines = f.readlines()
        # ver quandos ids existem
        ids = []
        for i in range(len(lines)):
            linha = lines[i].replace("\n", "")
            linha = linha.split("\t")
            ids.append(linha[-1])
        ids = set(ids)
        ids = len(ids)
        print(linha[-1])
        # vai existir ids/k fold sets
        id_fold_set = ids // k
        id_fold_set_rest = ids % k
        # percorrer o ficheiro e meter id_fold_set ids em cada fold set
        aux = -1
        ids_aux = []
        for i in range(ids):
            ids_aux.append([])
        for i in range(k):
            k_fold_sets.append([])
        ids_ = []
        for i in range(len(lines)):
            linha = lines[i].replace("\n", "")
            linha = linha.split("\t")
            id_i = linha[-1]
            if id_i not in ids_:
                aux += 1
                ids_.append(id_i)
            ids_aux[aux].append(lines[i])
        # meter os ids no fold set
        for i in range(k):
            for j in range(id_fold_set):
                k_fold_sets[i].append(ids_aux[i*id_fold_set+j])
        # meter os ids restantes nos fold sets
        for i in range(id_fold_set_rest):
            k_fold_sets[i].append(ids_aux[ids-id_fold_set_rest+i])
        # escrever nos treinos e testes
        # sendo dois fold sets para teste e o resto para treino
        fold_train = []
        fold_test = []
        for i in range(k):
            fold_train.append([])
            fold_test.append([])
            for j in range(k):
                if i == j or i == j+1:
                    fold_test[i].append(k_fold_sets[j])
                    if j == k-1:
                        fold_test[i].append(k_fold_sets[0])
                else:
                    fold_train[i].append(k_fold_sets[j])


        # criar ficheiros de trein

        # criar e escrever para o ficheiro
        for i in range(k):
            # juntar os k_fold_sets todos menos dois deles e guardar num ficheiro
            with open("csv/train/train_set_"+str(i)+".csv", "w") as f:
                for j in fold_train[i]:
                    for n in j:
                        # list to string
                        n = "\t".join(n)
                        f.write(n+"\n")
            # guardar os dois k_fold_sets num ficheiro teste
            with open("csv/test/test_validation_set_"+str(i)+".csv", "w") as f:
                for j in fold_test[i]:
                    for n in j:
                        n = "\t".join(n)
                        f.write(n+"\n")

# normaliza os dados
# Recebe o numero de fold sets
# Guarda os dados normalizados no ficheiro treino e teste
# Vai ler o ficheiro de treino ver os valores maximos e minimos de cada coluna
# E depois vai ler o ficheiro de teste e vai normalizar os dados
# o mesmo para o ficheiro de validacao
def normalize(k):
    # ler o ficheiro de treino
    with open("csv/train/train_set_0.csv", "r") as f:
        lines = f.readlines()
    # ver quantas colunas existem
    colunas = lines[0].split("\t")
    colunas = len(colunas)
    # ver o maximo e o minimo de cada coluna
    maximos = []
    minimos = []
    for i in range(colunas):
        maximos.append(0)
        minimos.append(100000)
    for i in range(len(lines)):
        linha = lines[i].split("\t")
        for j in range(colunas):
            if float(linha[j]) > maximos[j]:
                maximos[j] = float(linha[j])
            if float(linha[j]) < minimos[j]:
                minimos[j] = float(linha[j])
    # normalizar os dados
    for i in range(k):
        # ler o ficheiro de treino
        with open("csv/train/train_set_"+str(i)+".csv", "r") as f:
            lines = f.readlines()
        # criar ficheiro de treino normalizado
        with open("csv/train/train_set_norm_"+str(i)+".csv", "w") as f:
            for j in range(len(lines)):
                linha = lines[j].split("\t")
                for n in range(colunas):
                    linha[n] = (float(linha[n]) - minimos[n]) / (maximos[n] - minimos[n])
                linha = "\t".join(str(x) for x in linha)
                f.write(linha)
        # ler o ficheiro de teste
        with open("csv/test/test_validation_set_"+str(i)+".csv", "r") as f:
            lines = f.readlines()
        # criar ficheiro de teste normalizado
        with open("csv/test/test_validation_set_norm_"+str(i)+".csv", "w") as f:
            for j in range(len(lines)):
                linha = lines[j].split("\t")
                for n in range(colunas):
                    linha[n] = (float(linha[n]) - minimos[n]) / (maximos[n] - minimos[n])
                linha = "\t".join(str(x) for x in linha)
                f.write(linha)





def main():
    #tokens = tokenize()
    #print("Número de palavras:", len(tokens))
    #most_frequent(tokens)

    # POS tagging
    #tagged = nltk.pos_tag(tokens)
    #print(tagged[0:7])

    create_instances(1)

    k = 10
    create_k_fold_sets(k)
    print("k fold sets criados")
    normalize(k)
    print("dados normalizados")


main()