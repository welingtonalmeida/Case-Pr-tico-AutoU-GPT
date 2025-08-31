import os
import pdfplumber
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configuração Flask
app = Flask(__name__)

# Modelo básico (treinado com dados mais diversificados)
EXAMPLES = [
    # Exemplos Produtivos
    ("Preciso de ajuda com meu sistema", "Produtivo"),
    ("Favor verificar o chamado aberto", "Produtivo"),
    ("Abri um novo chamado no sistema para o problema X.", "Produtivo"),
    ("Poderia me ajudar a resolver a falha na tela de login?", "Produtivo"),
    ("Minha fatura está incorreta, por favor verifiquem.", "Produtivo"),
    ("Gostaria de solicitar um suporte técnico urgente.", "Produtivo"),
    
    # Exemplos Improdutivos
    ("Obrigado pela atenção", "Improdutivo"),
    ("Parabéns pelo seu trabalho", "Improdutivo"),
    ("Tenha um ótimo dia!", "Improdutivo"),
    ("Qual o horário de funcionamento da loja?", "Improdutivo"),
    ("Olá, tudo bem? Gostaria de saber mais sobre o serviço.", "Improdutivo"),
    ("Gostaria de me descadastrar da newsletter, obrigado.", "Improdutivo"),
]

texts, labels = zip(*EXAMPLES)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression().fit(X, labels)

# Função para processar texto e classificar
def classify_email(text):
    X_new = vectorizer.transform([text])
    prediction = model.predict(X_new)
    return prediction[0]

def generate_response(category):
    if category == "Produtivo":
        response = "Olá! Recebemos sua solicitação e retornaremos em breve."
    else:
        response = "Obrigado pela sua mensagem!"
    return response

# Página inicial
@app.route("/")
def index():
    return render_template("index.html")

# API de classificação
@app.route("/api/classify", methods=["POST"])
def api_classify():
    email_text = ""

    # 1. Prioriza a verificação do texto copiado e colado
    email_text_from_form = request.form.get("email_text", "").strip()

    if email_text_from_form:
        email_text = email_text_from_form
    else:
        # 2. Se a caixa de texto estiver vazia, verifica se o campo de arquivo foi preenchido
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            if file.filename.endswith(".txt"):
                email_text = file.read().decode("utf-8")
            elif file.filename.endswith(".pdf"):
                with pdfplumber.open(file) as pdf:
                    email_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    if not email_text:
        return jsonify({"error": "Nenhum texto encontrado"}), 400

    category = classify_email(email_text)
    response = generate_response(category)
    return jsonify({"category": category, "response": response})
