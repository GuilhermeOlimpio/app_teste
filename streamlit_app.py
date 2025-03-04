import streamlit as st
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from nltk.util import ngrams
from pyvis.network import Network
import base64
import io
import pandas as pd
import itertools

# Certifique-se de baixar as stopwords (apenas uma vez)
nltk.download('stopwords')

# Função para converter figuras matplotlib em bytes para embed no HTML
def plt_to_img_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()

st.title("Análise Interativa de PDF com Streamlit")

st.markdown("""
Este aplicativo permite que você faça o upload de um arquivo PDF, adicione stopwords personalizadas e gere diversas análises:
- **WordCloud** das palavras mais frequentes
- **Mapa de Calor** das associações entre os 25 termos mais frequentes
- **Gráfico de Bigramas** mais frequentes
- **Rede de Co-ocorrência** interativa (arquivo HTML para download)
- **Relatório HTML** completo com todas as visualizações
""")

# 1. Upload do PDF e entrada de stopwords adicionais
uploaded_file = st.file_uploader("Escolha o arquivo PDF", type="pdf")
custom_stopwords = st.text_input("Insira stopwords adicionais (separadas por vírgula)", "")

if uploaded_file is not None:
    # Extração do texto do PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    
    st.subheader("Texto Extraído (primeiros 500 caracteres)")
    st.write(text[:500])
    
    # 2. Pré-processamento: separar linhas, limpar e tokenizar
    comments = [line.strip() for line in text.split("\n") if line.strip() != ""]
    
    def clean_text(txt):
        txt = txt.lower()
        txt = re.sub(r'\d+', '', txt)
        txt = re.sub(r'[^\w\s]', '', txt)
        return txt

    comments_clean = [clean_text(c) for c in comments]
    
    # Stopwords: padrão do NLTK e customizadas pelo usuário
    default_stopwords = set(stopwords.words("portuguese"))
    custom_list = {w.strip() for w in custom_stopwords.split(",")} if custom_stopwords else set()
    total_stopwords = default_stopwords.union(custom_list)
    
    def tokenize(txt):
        tokens = txt.split()
        return [t for t in tokens if t not in total_stopwords]
    
    comments_tokens = [tokenize(c) for c in comments_clean]
    
    # 3. Análise de Frequência e WordCloud
    freq = Counter(token for tokens in comments_tokens for token in tokens)
    
    st.subheader("Palavras Mais Frequentes (Top 20)")
    st.write(freq.most_common(20))
    
    from wordcloud import WordCloud
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(freq)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.subheader("WordCloud")
    st.pyplot(fig_wc)
    
    # 4. Mapa de Calor dos 25 termos mais frequentes
    top25 = [t for t, _ in freq.most_common(25)]
    co_occurrence = pd.DataFrame(0, index=top25, columns=top25)
    for tokens in comments_tokens:
        tokens_set = set(tokens).intersection(top25)
        for token1, token2 in itertools.combinations(tokens_set, 2):
            co_occurrence.loc[token1, token2] += 1
            co_occurrence.loc[token2, token1] += 1
        for token in tokens_set:
            co_occurrence.loc[token, token] += 1

    fig_heat, ax_heat = plt.subplots(figsize=(12, 10))
    sns.heatmap(co_occurrence, annot=True, fmt="d", cmap="YlGnBu", ax=ax_heat)
    st.subheader("Mapa de Calor (Top 25 Termos)")
    st.pyplot(fig_heat)
    
    # 5. Gráfico de Bigramas
    bigrams = []
    for tokens in comments_tokens:
        bigrams.extend(list(ngrams(tokens, 2)))
    bigrams_freq = Counter(bigrams)
    top_bigrams = bigrams_freq.most_common(20)
    bigrams_labels = [" ".join(b) for b, _ in top_bigrams]
    bigrams_values = [count for _, count in top_bigrams]
    
    fig_bi, ax_bi = plt.subplots(figsize=(12, 6))
    sns.barplot(x=bigrams_values, y=bigrams_labels, palette="viridis", ax=ax_bi)
    ax_bi.set_title("Top 20 Bigramas Mais Frequentes")
    st.subheader("Bigramas")
    st.pyplot(fig_bi)
    
    # 6. Rede de Co-ocorrência Interativa com PyVis
    peso_minimo = 2  # Limite para filtrar conexões mais fracas
    G = nx.Graph()
    for tokens in comments_tokens:
        for token1, token2 in itertools.combinations(set(tokens), 2):
            if G.has_edge(token1, token2):
                G[token1][token2]['weight'] += 1
            else:
                G.add_edge(token1, token2, weight=1)
    # Filtra a rede removendo arestas com peso abaixo do limiar
    G_filtered = nx.Graph(((u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= peso_minimo))
    
    net = Network(height='750px', width='100%', notebook=False)
    for node in G_filtered.nodes():
        net.add_node(node, label=node)
    for u, v, data in G_filtered.edges(data=True):
        net.add_edge(u, v, value=data["weight"], title=f'Peso: {data["weight"]}')
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 14,
          "face": "Tahoma"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 250,
          "springConstant": 0.001
        },
        "minVelocity": 0.75
      }
    }
    """)
    rede_html = "rede_interativa.html"
    net.save_graph(rede_html)
    st.subheader("Rede de Co-ocorrência Interativa")
    st.markdown(f"Baixe a rede interativa: [Clique aqui]({rede_html})")
    
    # 7. Geração de um relatório HTML completo
    # Converte figuras para base64
    img_wc = base64.b64encode(plt_to_img_bytes(fig_wc)).decode()
    img_heat = base64.b64encode(plt_to_img_bytes(fig_heat)).decode()
    img_bi = base64.b64encode(plt_to_img_bytes(fig_bi)).decode()
    
    html_report = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Relatório de Análise de PDF</title>
      </head>
      <body>
        <h1>Relatório de Análise de PDF</h1>
        <h2>Texto Extraído (primeiros 500 caracteres)</h2>
        <p>{text[:500]}</p>
        <h2>Palavras Mais Frequentes (Top 20)</h2>
        <p>{freq.most_common(20)}</p>
        <h2>WordCloud</h2>
        <img src="data:image/png;base64,{img_wc}" width="800"/>
        <h2>Mapa de Calor (Top 25 Termos)</h2>
        <img src="data:image/png;base64,{img_heat}" width="800"/>
        <h2>Bigramas (Top 20)</h2>
        <img src="data:image/png;base64,{img_bi}" width="800"/>
        <h2>Rede de Co-ocorrência</h2>
        <p>A rede interativa foi salva como <a href="{rede_html}" target="_blank">rede_interativa.html</a>.</p>
      </body>
    </html>
    """
    
    st.subheader("Relatório Completo")
    st.download_button("Baixar Relatório HTML", data=html_report, file_name="relatorio.html", mime="text/html")

