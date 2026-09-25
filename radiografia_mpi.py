from mpi4py import MPI
import numpy as np
import time

def main():
    # Etapa 1 - Inicialização do ambiente MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()

    linhas_totais = 2000
    colunas_totais = 2000
    
    # Ajuste simples para garantir que as linhas sejam divisíveis pelo número de processos
    linhas_totais = (linhas_totais // size) * size 
    
    img = None
    parametros = None

    if rank == 0:
        # Etapa 2 - Geração da radiografia simulada no processo root
        # Fundo intermediário simulando tórax (valores ~100)
        img = np.random.randint(80, 120, (linhas_totais, colunas_totais), dtype='i')
        
        # Simulando pulmões (mais escuros, valores ~40) no centro-esquerda e centro-direita
        img[:, 200:900] = np.random.randint(30, 60, (linhas_totais, 700), dtype='i')
        img[:, 1100:1800] = np.random.randint(30, 60, (linhas_totais, 700), dtype='i')
        
        # Adicionando áreas suspeitas aleatórias (valores altos)
        mascara_suspeitos = np.random.random((linhas_totais, colunas_totais)) > 0.995
        img[mascara_suspeitos] = np.random.randint(201, 255, np.sum(mascara_suspeitos))

        # Etapa 3 - Definição dos parâmetros de análise
        parametros = {
            'linhas': linhas_totais,
            'colunas': colunas_totais,
            'limiar_leve': 200,
            'limiar_alta': 230,
            'perc_critico': 0.05
        }

    # Broadcast dos parâmetros para todos os processos
    parametros = comm.bcast(parametros, root=0)

    # Etapa 4 - Sincronização inicial com MPI_Barrier
    comm.Barrier()

    # Etapa 5 - Divisão da radiografia com MPI_Scatter
    linhas_por_processo = parametros['linhas'] // size
    img_local = np.empty((linhas_por_processo, parametros['colunas']), dtype='i')
    
    sendbuf = None
    if rank == 0:
        sendbuf = np.array_split(img, size, axis=0)
    
    img_local = comm.scatter(sendbuf, root=0)

    # Etapa 6 - Análise local de cada processo
    pixels_locais = img_local.size
    soma_intensidade_local = int(np.sum(img_local))
    maior_intensidade_local = int(np.max(img_local))
    
    mascara_leve = img_local > parametros['limiar_leve']
    mascara_alta = img_local > parametros['limiar_alta']
    
    qtd_suspeitos_local = int(np.sum(mascara_leve))
    qtd_alta_suspeita_local = int(np.sum(mascara_alta))
    
    # Identificação pulmão esquerdo e direito
    metade = parametros['colunas'] // 2
    qtd_suspeitos_esq = int(np.sum(mascara_leve[:, :metade]))
    qtd_suspeitos_dir = int(np.sum(mascara_leve[:, metade:]))

    # Etapa 7 - Classificação local da faixa analisada
    perc_suspeitos = qtd_suspeitos_local / pixels_locais
    if perc_suspeitos > parametros['perc_critico']:
        classificacao = "crítica"
    elif perc_suspeitos > (parametros['perc_critico'] / 2) or qtd_alta_suspeita_local > 0:
        classificacao = "atenção"
    else:
        classificacao = "normal"

    # Etapa 8 - Simulação de cluster heterogêneo
    if rank % 2 != 0:
        time.sleep(1) # Atraso artificial para processos ímpares

    # Etapa 9 - Sincronização antes da consolidação com MPI_Barrier
    comm.Barrier()

    # Etapa 10 - Consolidação numérica com MPI_Reduce
    total_pixels = comm.reduce(pixels_locais, op=MPI.SUM, root=0)
    soma_global = comm.reduce(soma_intensidade_local, op=MPI.SUM, root=0)
    total_suspeitos = comm.reduce(qtd_suspeitos_local, op=MPI.SUM, root=0)
    total_alta_suspeita = comm.reduce(qtd_alta_suspeita_local, op=MPI.SUM, root=0)
    total_esq = comm.reduce(qtd_suspeitos_esq, op=MPI.SUM, root=0)
    total_dir = comm.reduce(qtd_suspeitos_dir, op=MPI.SUM, root=0)
    max_global = comm.reduce(maior_intensidade_local, op=MPI.MAX, root=0)

    # Etapa 11 - Coleta de estatísticas detalhadas com MPI_Gather
    linha_inicio = rank * linhas_por_processo
    linha_fim = linha_inicio + linhas_por_processo - 1
    
    relatorio_local = {
        'rank': rank,
        'faixa': f"{linha_inicio} a {linha_fim}",
        'pixels': pixels_locais,
        'suspeitos': qtd_suspeitos_local,
        'alta_suspeita': qtd_alta_suspeita_local,
        'max_intensidade': maior_intensidade_local,
        'classificacao': classificacao
    }
    
    relatorios_detalhados = comm.gather(relatorio_local, root=0)

    # Etapa 12 - Relatório final do processo root
    if rank == 0:
        tempo_total = time.time() - start_time
        media_global = soma_global / total_pixels
        perc_suspeitos_global = (total_suspeitos / total_pixels) * 100
        
        lado_maior = "Pulmão Esquerdo" if total_esq > total_dir else "Pulmão Direito"
        
        if perc_suspeitos_global > (parametros['perc_critico'] * 100):
            classificacao_final = "Exame com alta concentração de áreas suspeitas"
        elif total_suspeitos > 0:
            classificacao_final = "Atenção clínica"
        else:
            classificacao_final = "Sem indícios relevantes"

        print("="*50)
        print(" RELATÓRIO FINAL DA ANÁLISE DE RADIOGRAFIA")
        print("="*50)
        print(f"-> Informações Gerais:")
        print(f"   Tamanho da imagem: {parametros['linhas']}x{parametros['colunas']}")
        print(f"   Processos MPI: {size}")
        print(f"   Tempo total de execução: {tempo_total:.4f} segundos")
        print(f"   Limiares: Leve > {parametros['limiar_leve']} | Alta > {parametros['limiar_alta']}")
        
        print(f"\n-> Estatísticas Globais:")
        print(f"   Total de pixels analisados: {total_pixels}")
        print(f"   Intensidade média global: {media_global:.2f}")
        print(f"   Maior intensidade global: {max_global}")
        print(f"   Total de pixels suspeitos: {total_suspeitos}")
        print(f"   Total altamente suspeitos: {total_alta_suspeita}")
        print(f"   Percentual de área suspeita: {perc_suspeitos_global:.4f}%")
        
        print(f"\n-> Comparação entre os Lados:")
        print(f"   Suspeitos no pulmão esquerdo: {total_esq}")
        print(f"   Suspeitos no pulmão direito: {total_dir}")
        print(f"   Lado com maior concentração: {lado_maior}")
        
        print(f"\n-> Classificação Geral:")
        print(f"   Resultado: {classificacao_final}")
        
        print(f"\n-> Estatísticas por Processo:")
        for rel in relatorios_detalhados:
            print(f"   Rank {rel['rank']} (Linhas {rel['faixa']}):")
            print(f"     Suspeitos: {rel['suspeitos']} | Máx Int.: {rel['max_intensidade']} | Classe: {rel['classificacao']}")
        print("="*50)

if __name__ == "__main__":
    main()