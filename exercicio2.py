from mpi4py import MPI
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10000000
N_local = N // size

random.seed(rank)

dentro = 0

comm.Barrier()
inicio = time.time()

for _ in range(N_local):
    x = random.random()
    y = random.random()
    if x*x + y*y <= 1:
        dentro += 1

print(f"Rank (ID) {rank}: pontos locais dentro do circulo: {dentro}")

total_dentro = comm.reduce(dentro, op=MPI.SUM, root=0)

# calculo final, feito apenas pelo processo 0
if rank == 0:
    pi = 4 * total_dentro / (N_local * size)
    fim = time.time()
    print("PI aproximado:", pi)
    print("Tempo:", (fim-inicio)*1000, "ms")