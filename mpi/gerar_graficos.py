import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plotar_desempenho():

    # 1. Carrega o arquivo final consolidado pelo Pandas

    try:
        df = pd.read_csv('resultados_analise_final.csv')
    except FileNotFoundError:
        print("Erro: O arquivo 'resultados_analise_final.csv' não foi encontrado.")
        print("Certifique-se de rodar primeiro o script 'processar_resultados.py'.")
        return

    # Garante tipos corretos para plotagem limpa

    df['Processos_MPI'] = df['Processos_MPI'].astype(int)
    df['Arquivo'] = df['Arquivo'].astype(str)
    
    # Define uma paleta de cores bonita e estilo limpo para relatórios acadêmicos

    sns.set_theme(style="whitegrid")
    ordem_resolucoes = ['512', '1024', '2048', '4095', '4096']
    
    print("=" * 70)
    print("       GERANDO GRÁFICOS AUTOMATIZADOS PARA O RELATÓRIO")
    print("=" * 70)

    
    # GRÁFICO 1: TEMPO DE EXECUÇÃO BRUTO
    
    print("Plotando gráfico de Tempos de Execução")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    for i, kernel in enumerate([3, 5]):
        ax = axes[i]
        sub_df = df[df['Kernel'] == kernel]
        
        sns.lineplot(data=sub_df, x='Processos_MPI', y='Tempo_MPI_s', hue='Arquivo', 
                     hue_order=ordem_resolucoes, marker='o', ax=ax, linewidth=2.5, markersize=8)
        
        ax.set_title(f'Tempo de Execução - Kernel {kernel}x{kernel}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantidade de Processos MPI', fontsize=10)
        ax.set_ylabel('Tempo Total (segundos)', fontsize=10)
        ax.set_xticks([1, 2, 4, 8])
        ax.legend(title='Resolução')
        
    plt.tight_layout()
    plt.savefig('grafico_1_tempos.png', dpi=300)
    plt.close()

   
    # GRÁFICO 2: SPEEDUP (S)
   
    print("Plotando gráfico de Speedup")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    for i, kernel in enumerate([3, 5]):
        ax = axes[i]
        sub_df = df[df['Kernel'] == kernel]
        
        sns.lineplot(data=sub_df, x='Processos_MPI', y='Speedup', hue='Arquivo', 
                     hue_order=ordem_resolucoes, marker='s', ax=ax, linewidth=2.5, markersize=8)
        
        # Desenha a linha de Speedup Linear Ideal (Referência teórica de Amdahl)

        processos_ideais = np.array([1, 2, 4, 8])
        ax.plot(processos_ideais, processos_ideais, 'r--', linewidth=2, label='Ideal (Linear)')
        
        ax.set_title(f'Speedup Escalável - Kernel {kernel}x{kernel}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantidade de Processos MPI', fontsize=10)
        ax.set_ylabel('Ganho de Velocidade (Speedup)', fontsize=10)
        ax.set_xticks([1, 2, 4, 8])
        ax.legend(title='Resolução / Alinhamento')
        
    plt.tight_layout()
    plt.savefig('grafico_2_speedup.png', dpi=300)
    plt.close()

   
    # GRÁFICO 3: EFICIÊNCIA (E)
   
    print("Plotando gráfico de Eficiência")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    for i, kernel in enumerate([3, 5]):
        ax = axes[i]
        sub_df = df[df['Kernel'] == kernel]
        
        sns.lineplot(data=sub_df, x='Processos_MPI', y='Eficiencia', hue='Arquivo', 
                     hue_order=ordem_resolucoes, marker='^', ax=ax, linewidth=2.5, markersize=8)
        
        # Desenha a linha horizontal de Eficiência Ideal (1.0 = 100% de uso do core)

        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Ideal (100%)')
        
        ax.set_title(f'Eficiência de Hardware - Kernel {kernel}x{kernel}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantidade de Processos MPI', fontsize=10)
        ax.set_ylabel('Fator de Eficiência (E)', fontsize=10)
        ax.set_xticks([1, 2, 4, 8])
        ax.legend(title='Resolução')
        
    plt.tight_layout()
    plt.savefig('grafico_3_eficiencia.png', dpi=300)
    plt.close()

    print("\nOs Três gráficos foram gerados com sucesso!")
    print(" -> grafico_1_tempos.png")
    print(" -> grafico_2_speedup.png")
    print(" -> grafico_3_eficiencia.png")
    print("=" * 70)

if __name__ == "__main__":
    plotar_desempenho()