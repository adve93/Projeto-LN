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
    tempo = 20*t
    with open("train.txt", "r") as f:
        lines = f.readlines()
        # ir buscar as classificações e guardar os classificcações num array
        classifications = []
        tam = len(lines)
        for i in range(1, tam):
            aux = lines[i].replace("\n", "")
            aux = aux.split("\t")
            classifications.append(aux[0])
        classifications = set(classifications)
        # meter por ordem alfabética
        classifications = sorted(classifications)
        # fazer as instancias
        instance = []
        for i in range(1, tam):
            linha = ""
            # tirar o \n
            linha_inicial = lines[i].replace("\n", "")
            # separar por \t
            linha_inicial = linha_inicial.split("\t")
            # tirar o tempo
            id_i = linha_inicial[0]
            # index da classificação 0, 1, 2, 3 (como se tem 4 classificações)
            classification_i = classifications.index(linha_inicial[0])
            # adicionar a ultima linha
            linha = "\t".join(linha_inicial[-1:])
            # ler as tx20 linhas seguintes se eistirem e adicionar ao fim da linha as 3 ultimas palavras
            if i + tempo >= tam:
                break
            aux = 0
            for j in range((i+1), ((tempo)+i+1)):
                if (tempo)+i < tam:
                    linha_final = lines[j].replace("\n", "")
                    linha_final = linha_final.split("\t")
                    if linha_final[0] == id_i and linha_final[0] == linha_inicial[0]:
                        aux = aux + 1
                        linha = linha + "\t" + "\t".join(linha_final[-1:])
                    else:
                        break
                else:
                    break
            if aux == tempo:
                # adicionar a linha ao instance
                linha = "".join(linha) + "\t" + str(classification_i) + "\t" + id_i

                instance.append(linha)

        # escrever no ficheiro
        with open("instances.txt", "w") as f:
            for i in instance:
                f.write(i + "\n")






def main():
    #tokens = tokenize()
    #print("Número de palavras:", len(tokens))
    #most_frequent(tokens)

    # POS tagging
    #tagged = nltk.pos_tag(tokens)
    #print(tagged[0:7])

    create_instances(20)


main()