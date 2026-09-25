from mpi4py import MPI
import numpy as np
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # Quem eu sou (0 é o chefe, o resto é trabalhador)
    size = comm.Get_size() # Quantos processos existem no total
    
    tempo_inicio = time.time()

    linhas = 2000
    colunas = 2000
    linhas_ajustadas = (linhas // size) * size 
    
    imagem_completa = None
    parametros = None

    # geraçao da radiografia
    if rank == 0:
        # fundo do corpo com tons de cinza entre 80 e 120
        imagem_completa = np.random.randint(80, 120, (linhas_ajustadas, colunas), dtype='i')
        
        # pulmão esquerdo (mais escuro, valores baixos)
        imagem_completa[:, 200:900] = np.random.randint(30, 60, (linhas_ajustadas, 700), dtype='i')
        
        # pulmão direito (mais escuro, valores baixos)
        imagem_completa[:, 1100:1800] = np.random.randint(30, 60, (linhas_ajustadas, 700), dtype='i')
        
        # áreas suspeitas
        mascara = np.random.random((linhas_ajustadas, colunas)) > 0.995
        imagem_completa[mascara] = np.random.randint(201, 255, np.sum(mascara))

        parametros = {
            'total_linhas': linhas_ajustadas,
            'total_colunas': colunas,
            'limiar_leve': 200,
            'limiar_alta': 230,
            'porcentagem_critica': 0.05
        }

    # root envia as regras para todos os trabalhadores
    parametros = comm.bcast(parametros, root=0)

    comm.Barrier()

    # define quantas linhas cada um vai analisar
    linhas_minhas = parametros['total_linhas'] // size
    
    # O root divide a imagem em partes iguais
    pedacos_da_imagem = None
    if rank == 0:
        pedacos_da_imagem = np.array_split(imagem_completa, size, axis=0)
    
    # cada processo recebe sua parte
    minha_faixa = comm.scatter(pedacos_da_imagem, root=0)

    # calcula os números da parte da imagem que cada processo recebe
    meus_pixels = minha_faixa.size
    minha_soma = int(np.sum(minha_faixa))
    meu_maximo = int(np.max(minha_faixa))
    
    # contando pixeis suspeitos
    meus_suspeitos = int(np.sum(minha_faixa > parametros['limiar_leve']))
    meus_altamente_suspeitos = int(np.sum(minha_faixa > parametros['limiar_alta']))
    
    # separa esquerdo e direito
    meio = parametros['total_colunas'] // 2
    metade_esquerda = minha_faixa[:, :meio]
    metade_direita = minha_faixa[:, meio:]
    
    suspeitos_esq = int(np.sum(metade_esquerda > parametros['limiar_leve']))
    suspeitos_dir = int(np.sum(metade_direita > parametros['limiar_leve']))

    # porcentagem de suspeitos na minha faixa
    minha_porcentagem = meus_suspeitos / meus_pixels
    
    if minha_porcentagem > parametros['porcentagem_critica']:
        minha_classificacao = "Critico"
    elif minha_porcentagem > (parametros['porcentagem_critica'] / 2) or meus_altamente_suspeitos > 0:
        minha_classificacao = "Atencao"
    else:
        minha_classificacao = "Normal"

    # fazemos os processos ímpares esperarem por 1 segundo para simular pc lento
    if rank % 2 != 0:
        time.sleep(1) 

    comm.Barrier()

    # O rank 0 soma os valores de todo mundo
    total_pixels = comm.reduce(meus_pixels, op=MPI.SUM, root=0)
    total_soma = comm.reduce(minha_soma, op=MPI.SUM, root=0)
    total_suspeitos = comm.reduce(meus_suspeitos, op=MPI.SUM, root=0)
    total_altamente_suspeitos = comm.reduce(meus_altamente_suspeitos, op=MPI.SUM, root=0)
    total_suspeitos_esq = comm.reduce(suspeitos_esq, op=MPI.SUM, root=0)
    total_suspeitos_dir = comm.reduce(suspeitos_dir, op=MPI.SUM, root=0)
    
    # pega o maior valor encontrado entre todos os processos
    maximo_global = comm.reduce(meu_maximo, op=MPI.MAX, root=0)

    # cada processo faz um resumo da sua tarefa
    linha_inicio = rank * linhas_minhas
    linha_fim = linha_inicio + linhas_minhas - 1
    
    meu_relatorio = {
        'rank': rank,
        'linhas_analisadas': f"{linha_inicio} até {linha_fim}",
        'pixels': meus_pixels,
        'suspeitos': meus_suspeitos,
        'altamente_suspeitos': meus_altamente_suspeitos,
        'maior_valor': meu_maximo,
        'resultado': minha_classificacao
    }
    
    # o root recolhe o relatório de todos e guarda numa lista
    todos_os_relatorios = comm.gather(meu_relatorio, root=0)

    # root imprime na tela
    if rank == 0:
        tempo_total = time.time() - tempo_inicio
        media_intensidade = total_soma / total_pixels
        porcentagem_geral = (total_suspeitos / total_pixels) * 100
        
        # lado com mais pixeis suspeitos
        if total_suspeitos_esq > total_suspeitos_dir:
            pior_lado = "Pulmao Esquerdo"
        else:
            pior_lado = "Pulmao Direito"
        
        # relatorio final
        limite_critico_porcentagem = parametros['porcentagem_critica'] * 100
        if porcentagem_geral > limite_critico_porcentagem:
            diagnostico = "Exame com areas suspeitas"
        elif total_suspeitos > 0:
            diagnostico = "Requer atencao"
        else:
            diagnostico = "Normal"

        print(" Relatorio Final Da Analise De Radiografia\n")
        
        print("\nInformações Gerais\n")
        print(f"Tamanho da Imagem: {parametros['total_linhas']} x {parametros['total_colunas']}")
        print(f"Quantidade de Processos: {size}")
        print(f"Tempo de Execução: {tempo_total:.2f} segundos")
        
        print("\nEstatísticas Globais\n")
        print(f"Pixels analisados: {total_pixels}")
        print(f"Intensidade média da imagem: {media_intensidade:.1f}")
        print(f"Maior intensidade encontrada: {maximo_global}")
        print(f"Total de pixels suspeitos: {total_suspeitos}")
        print(f"Total de pixels muito suspeitos: {total_altamente_suspeitos}")
        print(f"Porcentagem da área suspeita: {porcentagem_geral:.3f}%")
        
        print("\nComparação dos Pulmões\n")
        print(f"Suspeitos no Lado Esquerdo: {total_suspeitos_esq}")
        print(f"Suspeitos no Lado Direito: {total_suspeitos_dir}")
        print(f"Lado mais afetado: {pior_lado}")
        
        print("\nClassificação Final")
        print(f"Resultado do exame: {diagnostico}")
        
        print("\nResumo pro processo")
        for rel in todos_os_relatorios:
            print(f"-Processo {rel['rank']} (Linhas {rel['linhas_analisadas']}):")
            print(f"-Classificação: {rel['resultado']}\n Suspeitos: {rel['suspeitos']}\nMáximo: {rel['maior_valor']}")
            
        print("\n")

if __name__ == "__main__":
    main()