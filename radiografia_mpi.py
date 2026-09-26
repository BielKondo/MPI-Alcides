from mpi4py import MPI
import numpy as np
import time

def main():
    # Etapa 1 - Inicialização do ambiente MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    tempo_inicio = time.time()

    linhas = 2000
    colunas = 2000
    
    imagem_completa = None
    parametros = None

    if rank == 0:
        # Etapa 2 - Geração da radiografia no processo root (sem descartar nada)
        imagem_completa = np.random.randint(80, 120, (linhas, colunas), dtype='i')
        imagem_completa[:, 200:900] = np.random.randint(30, 60, (linhas, 700), dtype='i')
        imagem_completa[:, 1100:1800] = np.random.randint(30, 60, (linhas, 700), dtype='i')
        
        mascara = np.random.random((linhas, colunas)) > 0.995
        imagem_completa[mascara] = np.random.randint(201, 255, np.sum(mascara))

        # Etapa 3 - Definição dos parâmetros de análise
        parametros = {
            'total_linhas': linhas,
            'total_colunas': colunas,
            'limiar_leve': 200,
            'limiar_alta': 230,
            'porcentagem_critica': 0.05
        }

    # Broadcast dos parâmetros
    parametros = comm.bcast(parametros, root=0)

    # Etapa 4 - Sincronização inicial
    comm.Barrier()

    total_linhas = parametros['total_linhas']
    
    # Calculando quantas linhas cada processo vai receber (distribuindo o resto)
    linhas_base = total_linhas // size
    resto = total_linhas % size
    
    # Lista com a quantidade de linhas de cada processo
    sendcounts_linhas = [linhas_base + 1 if i < resto else linhas_base for i in range(size)]
    
    # Convertendo para o tamanho total de elementos (linhas * colunas) para o Scatterv
    sendcounts = [l * colunas for l in sendcounts_linhas]
    
    # Calculando os deslocamentos (displacements) de onde cada pedaço começa na matriz
    displs = [sum(sendcounts[:i]) for i in range(size)]
    
    # Criando o array local com o tamanho exato que este processo vai receber
    minha_faixa = np.empty((linhas_linhas := sendcounts_linhas[rank], colunas), dtype='i')
    
    # O Scatterv distribui os dados brutos respeitando os tamanhos variáveis de cada processo
    comm.Scatterv([imagem_completa, sendcounts, displs, MPI.INT], minha_faixa, root=0)

    # Etapa 6 - Análise local de cada processo
    meus_pixels = minha_faixa.size
    minha_soma = int(np.sum(minha_faixa))
    meu_maximo = int(np.max(minha_faixa))
    
    meus_suspeitos = int(np.sum(minha_faixa > parametros['limiar_leve']))
    meus_altamente_suspeitos = int(np.sum(minha_faixa > parametros['limiar_alta']))
    
    meio = parametros['total_colunas'] // 2
    metade_esquerda = minha_faixa[:, :meio]
    metade_direita = minha_faixa[:, meio:]
    
    suspeitos_esq = int(np.sum(metade_esquerda > parametros['limiar_leve']))
    suspeitos_dir = int(np.sum(metade_direita > parametros['limiar_leve']))

    # Etapa 7 - Classificação local
    minha_porcentagem = meus_suspeitos / meus_pixels if meus_pixels > 0 else 0
    if minha_porcentagem > parametros['porcentagem_critica']:
        minha_classificacao = "CRÍTICA"
    elif minha_porcentagem > (parametros['porcentagem_critica'] / 2) or meus_altamente_suspeitos > 0:
        minha_classificacao = "ATENÇÃO"
    else:
        minha_classificacao = "NORMAL"

    # Etapa 8 - Simulação de cluster heterogêneo
    if rank % 2 != 0:
        time.sleep(1) 

    # Etapa 9 - Sincronização antes de juntar tudo
    comm.Barrier()

    # Etapa 10 - Consolidação numérica (Reduce)
    total_pixels = comm.reduce(meus_pixels, op=MPI.SUM, root=0)
    total_soma = comm.reduce(minha_soma, op=MPI.SUM, root=0)
    total_suspeitos = comm.reduce(meus_suspeitos, op=MPI.SUM, root=0)
    total_altamente_suspeitos = comm.reduce(meus_altamente_suspeitos, op=MPI.SUM, root=0)
    total_suspeitos_esq = comm.reduce(suspeitos_esq, op=MPI.SUM, root=0)
    total_suspeitos_dir = comm.reduce(suspeitos_dir, op=MPI.SUM, root=0)
    maximo_global = comm.reduce(meu_maximo, op=MPI.MAX, root=0)

    # Etapa 11 - Coleta de estatísticas detalhadas (Gather)
    # Calculando a linha inicial de cada processo somando as linhas anteriores dinamicamente
    linha_inicio = sum(sendcounts_linhas[:rank])
    linha_fim = linha_inicio + sendcounts_linhas[rank] - 1
    
    meu_relatorio = {
        'rank': rank,
        'linhas_analisadas': f"{linha_inicio} até {linha_fim}",
        'pixels': meus_pixels,
        'suspeitos': meus_suspeitos,
        'altamente_suspeitos': meus_altamente_suspeitos,
        'maior_valor': meu_maximo,
        'resultado': minha_classificacao
    }
    
    todos_os_relatorios = comm.gather(meu_relatorio, root=0)

    # Etapa 12 - Relatório Final
    if rank == 0:
        tempo_total = time.time() - tempo_inicio
        media_intensidade = total_soma / total_pixels
        porcentagem_geral = (total_suspeitos / total_pixels) * 100
        
        if total_suspeitos_esq > total_suspeitos_dir:
            pior_lado = "Pulmão Esquerdo"
        else:
            pior_lado = "Pulmão Direito"
        
        limite_critico_porcentagem = parametros['porcentagem_critica'] * 100
        if porcentagem_geral > limite_critico_porcentagem:
            diagnostico = "Exame com ALTA concentração de áreas suspeitas"
        elif total_suspeitos > 0:
            diagnostico = "Requer ATENÇÃO clínica"
        else:
            diagnostico = "Sem indícios relevantes (NORMAL)"

        print("\n" + "="*50)
        print(" RELATÓRIO FINAL (COM SCATTERV - SEM PERDA DE PIXELS)")
        print("="*50)
        print(f"\n[ INFORMAÇÕES GERAIS ]")
        print(f"- Tamanho da Imagem: {parametros['total_linhas']} x {parametros['total_colunas']}")
        print(f"- Quantidade de Processos: {size}")
        print(f"- Tempo de Execução: {tempo_total:.2f} segundos")
        
        print(f"\n[ ESTATÍSTICAS GLOBAIS ]")
        print(f"- Pixels analisados: {total_pixels} (100% preservados)")
        print(f"- Intensidade média da imagem: {media_intensidade:.1f}")
        print(f"- Maior intensidade encontrada: {maximo_global}")
        print(f"- Total de pixels suspeitos: {total_suspeitos}")
        print(f"- Total de pixels MUITO suspeitos: {total_altamente_suspeitos}")
        print(f"- Porcentagem da área suspeita: {porcentagem_geral:.3f}%")
        
        print(f"\n[ COMPARAÇÃO DOS PULMÕES ]")
        print(f"- Suspeitos no Lado Esquerdo: {total_suspeitos_esq}")
        print(f"- Suspeitos no Lado Direito: {total_suspeitos_dir}")
        print(f"- Lado mais afetado: {pior_lado}")
        
        print(f"\n[ CLASSIFICAÇÃO FINAL ]")
        print(f"- Resultado do exame: {diagnostico}")
        
        print(f"\n[ RESUMO POR PROCESSO ]")
        for rel in todos_os_relatorios:
            print(f"  -> Processo {rel['rank']} (Linhas {rel['linhas_analisadas']} | {rel['pixels']} pixels):")
            print(f"     Classificação: {rel['resultado']} | Suspeitos: {rel['suspeitos']}")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()