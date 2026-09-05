from mpi4py import MPI
import random
import time

N = 1000 # trocar para 600 e 1000 nos outros testes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 1. Somente o rank 0 gera as matrizes A e B
if rank == 0:
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]
else:
    A = None
    B = None

# Sincroniza todos antes de comecar a medir o tempo
comm.Barrier()
t_inicio = time.time()

# 2. Envia as matrizes para todos os processos com broadcast
A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

# 3. Divisao das linhas entre os processos
# (o resto e distribuido entre os primeiros ranks, para funcionar
#  mesmo quando N nao e divisivel pelo numero de processos)
linhas = N // size
resto = N % size

inicio = rank * linhas + min(rank, resto)
fim = inicio + linhas + (1 if rank < resto else 0)

# Cada processo calcula apenas as suas linhas da matriz resultado
C_local = [[0] * N for _ in range(fim - inicio)]

for i in range(inicio, fim):
    for j in range(N):
        soma = 0
        for k in range(N):
            soma += A[i][k] * B[k][j]
        C_local[i - inicio][j] = soma

# 4. Envia as partes calculadas para o processo 0 com gather
partes = comm.gather(C_local, root=0)

# 5. O processo 0 monta a matriz final
if rank == 0:
    C = []
    for parte in partes:
        for linha in parte:
            C.append(linha)

    t_fim = time.time()

    # 6. Exibe o tempo de execucao em milissegundos
    print("N:", N, "| Processos:", size)
    print("Tempo distribuido:", (t_fim - t_inicio) * 1000, "ms")
    print("Linhas da matriz final:", len(C), "x", len(C[0]))