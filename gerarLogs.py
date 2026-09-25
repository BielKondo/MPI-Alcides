from mpi4py import MPI
import random

# Inicialização do MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# -----------------------------
# Função para gerar logs
# -----------------------------
def gerar_logs(qtd):
    ips = [f"192.168.1.{i}" for i in range(1, 255)]
    endpoints = [
        "/",
        "/login",
        "/products",
        "/cart",
        "/checkout",
        "/api/users",
        "/api/orders"
    ]
    metodos = ["GET", "POST"]
    status = ["200", "200", "200", "404", "500"]
    logs = []

    # INSIRA AQUI O SEU CÓDIGO PARA GERAR OS DADOS DO LOG
    for _ in range(qtd):
        linha = f"{random.choice(ips)} {random.choice(metodos)} {random.choice(endpoints)} {random.choice(status)}"
        logs.append(linha)

    return logs


# -----------------------------
# Processo 0 gera o dataset
# -----------------------------
logs_divididos = None
TOTAL_LOGS = 1000000

if rank == 0:
    print("\nGerando dataset de logs...\n")
    logs = gerar_logs(TOTAL_LOGS)

    # DIVIDIR AQUI O DATASET ENTRE OS PROCESSOS
    chunk = len(logs) // size
    logs_divididos = [
        logs[i * chunk:(i + 1) * chunk]
        for i in range(size)
    ]

# -----------------------------
# Distribuição usando Scatter
# -----------------------------
# DISTRIBUIR OS DADOS USANDO SCATTER
logs_locais = comm.scatter(logs_divididos, root=0)

# -----------------------------
# Processamento local
# -----------------------------
# INSIRA AQUI O CÓDIGO DE PROCESSAMENTO LOCAL DO LOG
erros = 0

for linha in logs_locais:
    campos = linha.split()
    status_code = campos[3]
    if status_code in ("404", "500"):
        erros += 1

# -----------------------------
# Resultado local
# -----------------------------
print(
    f"Processo {rank} analisou {len(logs_locais)} linhas "
    f"e encontrou {erros} erros."
)