import pandas as pd

def processar_benchmark():
    arquivo_input = 'resultados_benchmark.csv'
    arquivo_output = 'resultados_analise_final.csv'
    
    try:
        # 1. Carrega o arquivo CSV gerado pelo benchmark

        df = pd.read_csv(arquivo_input)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{arquivo_input}' não foi encontrado.")
        print("Certifique-se de executar o script do PowerShell ('rodar_benchmark.ps1') primeiro.")
        return

    # Limpa possíveis espaços em branco nos nomes das colunas e dados

    df.columns = df.columns.str.strip()
    df['Arquivo'] = df['Arquivo'].astype(str).str.strip()
    
    # 2. Separa os dados do código Serial de Referência (Processos_Threads == 0)

    df_serial = df[df['Processos_Threads'] == 0].copy()
    df_serial = df_serial.rename(columns={'Tempo_s': 'Tempo_Serial_s'})
    df_serial = df_serial.drop(columns=['Processos_Threads']) # Remove coluna id de thread do mestre

    # 3. Separa os dados da execução paralela com MPI (Processos_Threads > 0)

    df_mpi = df[df['Processos_Threads'] > 0].copy()
    df_mpi = df_mpi.rename(columns={'Processos_Threads': 'Processos_MPI', 'Tempo_s': 'Tempo_MPI_s'})

    # 4. Faz o Merge (cruzamento) para alinhar cada teste MPI com seu respectivo tempo Serial
    
    df_final = pd.merge(df_mpi, df_serial, on=['Arquivo', 'Kernel'], how='left')

    # 5. Calcula as métricas de Computação de Alto Desempenho (HPC)

    # Speedup: S = Tempo_Serial / Tempo_Paralelo
    
    df_final['Speedup'] = df_final['Tempo_Serial_s'] / df_final['Tempo_MPI_s']
    
    # Eficiência: E = Speedup / Quantidade_de_Processos
    
    df_final['Eficiencia'] = df_final['Speedup'] / df_final['Processos_MPI']

    # 6. Salva o resultado bruto consolidado em um novo arquivo para backup ou gráficos
    
    df_final.to_csv(arquivo_output, index=False)
    
    print("=" * 70)
    print("       TABELAS GERADAS COM SUCESSO PARA O RELATÓRIO DO PROFESSOR")
    print("=" * 70)

    # TABELA A: TEMPOS DE EXECUÇÃO COMPLETOS (INCLUINDO SERIAL)
   
    print("\nTABELA 1: TEMPO DE EXECUÇÃO TOTAL (em segundos)")
    print("Nota: Use estes dados para a tabela comparativa de tempos brutos.")
    
    # Pivota os dados para colocar os Processos/Serial nas colunas
    
    pivot_tempo = df.pivot_table(index=['Arquivo', 'Kernel'], columns='Processos_Threads', values='Tempo_s')
    
    # Renomeia a coluna 0 (que era o serial) para ficar legível no relatório
    
    pivot_tempo = pivot_tempo.rename(columns={0: 'Serial puro'})
    print(pivot_tempo.round(4).to_markdown())

    
    # TABELA B: CALCULO DE SPEEDUP
   
    print("\nTABELA 2: MÉTRICA DE SPEEDUP (S)")
    print("Quanto maior o valor, mais rápido o código rodou em relação ao sequencial.")
    pivot_speedup = df_final.pivot_table(index=['Arquivo', 'Kernel'], columns='Processos_MPI', values='Speedup')
    print(pivot_speedup.round(2).to_markdown())

    
    # TABELA C: CÁLCULO DE EFICIÊNCIA
    
    print("\nTABELA 3: MÉTRICA DE EFICIÊNCIA (E)")
    print("Mede o aproveitamento real dos núcleos (1.00 = 100% de uso ideal do hardware).")
    pivot_eficiencia = df_final.pivot_table(index=['Arquivo', 'Kernel'], columns='Processos_MPI', values='Eficiencia')
    print(pivot_eficiencia.round(2).to_markdown())
    
    print("\n" + "=" * 70)
    print(f"Arquivo detalhado gravado com sucesso em: {arquivo_output}")
    print("=" * 70)

if __name__ == "__main__":
    processar_benchmark()