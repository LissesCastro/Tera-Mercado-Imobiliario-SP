# Tera-Mercado-Imobiliario-SP

## Documentação em construção

### Os notebooks com os códigos estão disponíveis nesses links:
- ### [EDA](Codigo_EDA.ipynb)
- ### [Modelo Preditivo](modelo_preditivo.ipynb)
 
 O mercado imobiliário compreende a venda e locação de milhares de imóveis todos os dias, sob diversas formas de negócio e abrangendo uma quantidade muito diversa de tipologias, além de um público muito variado. Por mais distintas que as pessoas sejam, é seguro afirmar que a grande maioria delas tem de lidar com algum processo de compra, venda, ou aluguel de algum imóvel pelo menos uma vez na vida. 
 
 A cidade de São Paulo, junto de sua região metropolitana, é um dos maiores polos de desenvolvimento imobiliário do Brasil: segundo dados da SECOVI-SP, no ano de 2022 foram lançadas 75.692 unidades habitacionais residenciais na cidade de São Paulo, somando um Valor Geral de Vendas (VGV) previsto de mais de 40 bilhões de reais. 
Havendo uma gama tão vasta de empreendimentos em uma dinâmica imobiliária acelerada e com um público alvo muito diverso, é intuitivo pensar que a definição do preço de um produto imobiliário (uma casa ou um apartamento, por exemplo) é um dos principais problemas na hora da elaboração e lançamento de um empreendimento. A alta oferta de unidades desfavorece a determinação de valores muito acima dos do mercado, paralelamente, o empreendimento imobiliário requer alto aporte inicial e tem lenta liquidez, o que torna arriscada a venda de unidades em valores muito abaixo dos praticados no mercado. 

 A precificação é um dos fatores mais fundamentais do empreendimento imobiliário e deve ser realizada ainda nas primeiras fases do projeto, nesse sentido, como é possível definir o preço de venda de uma unidade habitacional de modo que o valor seja competitivo no mercado? Quais variáveis podem ajudar melhor a prever o valor final de uma casa ou apartamento e qual o impacto delas sobre o produto? Esse certamente é um problema complexo, mas o estudo do que foi lançado em São Paulo pode indicar um caminho no entendimento e mesmo na predição dos preços.

A predição do preço da unidade é interessante para o mercado e para o usuário, uma vez que permite estipular as possibilidades de ganhos antes mesmo do início do empreendimento, enquanto permite averiguar a diferença do valor da unidade em relação ao previsto, ajudando na obtenção de melhores negócios. A antecipação dos preços de venda da unidade não é uma tarefa fácil mas, por sorte, o python mantém uma série de ferramentas úteis para o entendimento dos dados e também para a criação de modelos preditivos a partir de algoritmos de Machine Learning. 

 Esse código explora o potencial da ferramenta na resolução de problemas de precificação a partir de duas etapas distintas: primeiro, realiza uma etapa exploratória dos dados (popularmente conhecida como EDA - Exploratory Data Analysis), onde identifica as características da base de dados, encontra padrões de distribuição, verifica dados faltantes e elabora leituras sobre o conjunto. Todo o processo de EDA é realizado, principalmente, através da manipulação de Dataframes que as bibliotecas Pandas e Numpy permitem realizar com praticidade; A segunda parte do texto se dedica à elaboração de modelos preditivos do valor da unidade, realizados a partir do aprendizado supervisionado de uma base já ajustada durante a EDA e com o apoio de algoritmos de regressão das bibliotecas de Statsmodel e ScikitLearn do python.
