// Arthur e João - Versão CUDA

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <cuda_runtime.h>

/* -------------------------------------------------------------------------- */
/* Utilitários para arrays 2D armazenados em vetor 1D                        */
/* -------------------------------------------------------------------------- */

static double *alloc_image(int rows, int cols) {
    double *m = (double *)calloc((size_t)rows * cols, sizeof(double));
    if (!m) {
        printf("Erro: Nao foi possivel alocar memoria\n");
        exit(1);
    }
    return m;
}

static void free_image(double *m) {
    free(m);
}

/* Lê uma imagem PGM e retorna o vetor de pixels */
double *read_pgm(const char *filename, int *rows, int *cols) {

    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Erro: Nao foi possivel abrir o arquivo %s\n", filename);
        exit(1);
    }

    char magic[3];
    int max_val;

    if (fscanf(file, "%2s", magic) != 1) {
        printf("Erro: Arquivo PGM invalido (magic number)\n");
        fclose(file);
        exit(1);
    }

    int c;
    while ((c = fgetc(file)) == '#') {
        while ((c = fgetc(file)) != '\n' && c != EOF);
    }
    ungetc(c, file);

    if (fscanf(file, "%d %d %d", cols, rows, &max_val) != 3) {
        printf("Erro: Arquivo PGM invalido (dimensoes)\n");
        fclose(file);
        exit(1);
    }

    fgetc(file);

    double *image = alloc_image(*rows, *cols);
    unsigned char pixel;

    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            if (fread(&pixel, 1, 1, file) != 1) {
                printf("Erro: Falha ao ler pixel [%d][%d]\n", i, j);
                fclose(file);
                exit(1);
            }
            image[i * (*cols) + j] = (double)pixel;
        }
    }

    fclose(file);
    printf("Imagem lida: %s (%dx%d)\n", filename, *cols, *rows);
    return image;
}

void write_pgm(const char *filename, double *image, int rows, int cols) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Erro: Nao foi possivel criar o arquivo %s\n", filename);
        exit(1);
    }

    fprintf(file, "P5\n");
    fprintf(file, "%d %d\n", cols, rows);
    fprintf(file, "255\n");

    unsigned char pixel;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double val = image[i * cols + j];
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            pixel = (unsigned char)val;
            fwrite(&pixel, 1, 1, file);
        }
    }

    fclose(file);
    printf("Imagem salva: %s\n", filename);
}

/* -------------------------------------------------------------------------- */
/* Kernel CUDA para a Convolução Gaussiana                                    */
/* -------------------------------------------------------------------------- */
__global__ void convolution_kernel(const double *current, double *next, int rows, int cols, const double *kernel, int k_size) {
    // Mapeamento bidimensional das threads
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Coluna (X)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Linha (Y)

    // Verifica se a thread está dentro dos limites da imagem
    if (i < rows && j < cols) {
        int pad = k_size / 2;
        double sum = 0.0;

        // Executa a convolução aplicando o padding dinamicamente (clamping nas bordas)
        for (int ki = 0; ki < k_size; ki++) {
            for (int kj = 0; kj < k_size; kj++) {
                int si = i + ki - pad;
                int sj = j + kj - pad;

                // Tratamento de borda (clamping) idêntico ao serial
                if (si < 0) si = 0;
                if (si >= rows) si = rows - 1;
                if (sj < 0) sj = 0;
                if (sj >= cols) sj = cols - 1;

                sum += current[si * cols + sj] * kernel[ki * k_size + kj];
            }
        }
        next[i * cols + j] = sum;
    }
}

/* -------------------------------------------------------------------------- */

void create_gaussian_kernel(int size, double sigma, double *kernel) {
    if (size % 2 == 0) {
        printf("Erro: O tamanho do kernel deve ser ímpar.\n");
        exit(1);
    }

    int half = size / 2;
    double sum = 0.0;

    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            double value = exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            kernel[(i + half) * size + (j + half)] = value;
            sum += value;
        }
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[i * size + j] /= sum;
        }
    }
}

void iterative_gaussian_blur_cuda(double *image, int rows, int cols, int k_size, int iterations, double sigma, double *final_out) {
    size_t img_bytes = (size_t)rows * cols * sizeof(double);
    size_t kernel_bytes = (size_t)k_size * k_size * sizeof(double);

    // Alocação do Kernel no Host
    double *h_kernel = (double *)malloc(kernel_bytes);
    create_gaussian_kernel(k_size, sigma, h_kernel);

    // Alocação de memória no Device (GPU)
    double *d_current, *d_next, *d_kernel;
    cudaMalloc((void **)&d_current, img_bytes);
    cudaMalloc((void **)&d_next, img_bytes);
    cudaMalloc((void **)&d_kernel, kernel_bytes);

    // Cópia inicial de dados para o Device
    cudaMemcpy(d_current, image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    // Configuração de Blocos e Grid (Tamanho típico 16x16 = 256 threads por bloco)
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Loop iterativo controlado pela CPU, mas executado na GPU
    for (int iter = 0; iter < iterations; iter++) {
        convolution_kernel<<<gridSize, blockSize>>>(d_current, d_next, rows, cols, d_kernel, k_size);
        
        // Sincronização implícita a cada iteração devido ao fluxo de dados, 
        // mas fazemos o swap dos ponteiros na CPU de forma extremamente rápida.
        double *tmp = d_current;
        d_current = d_next;
        d_next = tmp;
    }

    // Copia o resultado final de volta para o Host
    cudaMemcpy(final_out, d_current, img_bytes, cudaMemcpyDeviceToHost);

    // Liberação de memória
    free(h_kernel);
    cudaFree(d_current);
    cudaFree(d_next);
    cudaFree(d_kernel);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Uso: %s <caminho_imagem_entrada> <caminho_imagem_saida> <kernel_size> <iteracoes>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    int k_size = atoi(argv[3]);
    int iterations = atoi(argv[4]);

    double sigma = 1.0;
    int rows, cols;

    double *image = read_pgm(input_file, &rows, &cols);
    double *output = alloc_image(rows, cols);

    // Sincroniza a GPU antes de iniciar a cronometragem para garantir precisão
    cudaDeviceSynchronize();
    double start = omp_get_wtime();

    iterative_gaussian_blur_cuda(image, rows, cols, k_size, iterations, sigma, output);

    // Garante que todas as operações da GPU terminaram antes de parar o cronômetro
    cudaDeviceSynchronize();
    double end = omp_get_wtime();

    // Linha lida pelo script (Select-String "Tempo" + regex de decimal)
    printf("Modo: CUDA | Kernel: %d | Iteracoes: %d | Tempo: %.4f s\n", k_size, iterations, end - start);

    // Salva a imagem de saída no caminho indicado pelo script
    write_pgm(output_file, output, rows, cols);

    // Limpeza
    free_image(image);
    free_image(output);

    return 0;
}