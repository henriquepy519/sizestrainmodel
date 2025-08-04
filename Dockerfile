FROM python:3.9-slim

# Cria diretório da aplicação
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências do Python
COPY requirements-arm.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-arm.txt

# Copia o restante da aplicação
COPY . .

# Expõe a porta 80
EXPOSE 80

# Usa gunicorn para rodar a aplicação Flask na porta 80
CMD ["gunicorn", "--bind", "0.0.0.0:80", "main:app"]