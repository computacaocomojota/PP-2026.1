// Arthur e João - Versão CUDA (Shared Memory + Memória Constante)
//
// Variante otimizada de filtro_cuda.cu: usa memoria compartilhada com
// "tiling + halo" para reduzir acessos redundantes a memoria global, e
// memoria constante para o kernel gaussiano (pequeno, read-only, mesmo
// valor lido por todas as threads de um warp -> acesso em broadcast).
//
// Mantida como arquivo separado para permitir comparação direta de tempos
// contra a versão que usa apenas memória global (filtro_cuda.cu).

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
/* Memória constante: kernel gaussiano (pequeno, read-only, mesmo para        */
/* todas as threads -> acesso em broadcast, cache dedicado).                  */
/* Suporta kernels de até 5x5 (25 doubles).      */
/* -------------------------------------------------------------------------- */
#define MAX_K_SIZE 7
__constant__ double d_kernel_const[MAX_K_SIZE * MAX_K_SIZE];

/* -------------------------------------------------------------------------- */
/* Kernel CUDA com memória compartilhada (tiling + halo)                     */
/* -------------------------------------------------------------------------- */
__global__ void convolution_kernel_shared(const double *current, double *next, int rows, int cols, int k_size) {
    extern __shared__ double tile[]; // tamanho definido dinamicamente no lançamento

    int pad = k_size / 2;
    int tile_w = blockDim.x + 2 * pad;
    int tile_h = blockDim.y + 2 * pad;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Coordenadas globais do pixel "dono" desta thread
    int j = blockIdx.x * blockDim.x + tx;
    int i = blockIdx.y * blockDim.y + ty;

    // Carrega o tile (incluindo o halo) para a memória compartilhada.
    // Cada thread pode carregar mais de um elemento quando o tile é maior
    // que o bloco (loop em passos de blockDim).
    for (int dy = ty; dy < tile_h; dy += blockDim.y) {
        int gi = blockIdx.y * blockDim.y + dy - pad;
        if (gi < 0) gi = 0;
        if (gi >= rows) gi = rows - 1;

        for (int dx = tx; dx < tile_w; dx += blockDim.x) {
            int gj = blockIdx.x * blockDim.x + dx - pad;
            if (gj < 0) gj = 0;
            if (gj >= cols) gj = cols - 1;

            tile[dy * tile_w + dx] = current[gi * cols + gj];
        }
    }

    // Garante que todo o tile foi carregado antes de qualquer thread usá-lo
    __syncthreads();

    if (i < rows && j < cols) {
        double sum = 0.0;

        // Convolução lida inteiramente a partir da memória compartilhada
        // (tile) e da memória constante (d_kernel_const) — nenhum acesso
        // adicional à memória global é feito aqui.
        for (int ki = 0; ki < k_size; ki++) {
            for (int kj = 0; kj < k_size; kj++) {
                sum += tile[(ty + ki) * tile_w + (tx + kj)] * d_kernel_const[ki * k_size + kj];
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

void iterative_gaussian_blur_cuda_shared(double *image, int rows, int cols, int k_size, int iterations, double sigma, double *final_out) {
    if (k_size > MAX_K_SIZE) {
        printf("Erro: k_size (%d) excede MAX_K_SIZE (%d) definido para memoria constante.\n", k_size, MAX_K_SIZE);
        exit(1);
    }

    size_t img_bytes = (size_t)rows * cols * sizeof(double);
    size_t kernel_bytes = (size_t)k_size * k_size * sizeof(double);

    // Alocação e cálculo do kernel no Host
    double *h_kernel = (double *)malloc(kernel_bytes);
    create_gaussian_kernel(k_size, sigma, h_kernel);

    // Copia o kernel gaussiano para a MEMÓRIA CONSTANTE do device
    cudaMemcpyToSymbol(d_kernel_const, h_kernel, kernel_bytes);

    // Alocação de memória GLOBAL no Device para os buffers de imagem
    double *d_current, *d_next;
    cudaMalloc((void **)&d_current, img_bytes);
    cudaMalloc((void **)&d_next, img_bytes);

    cudaMemcpy(d_current, image, img_bytes, cudaMemcpyHostToDevice);

    // Configuração de Blocos e Grid (mesmo block size da versão global, para comparação justa)
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Tamanho da memória compartilhada: tile do bloco + halo em cada borda
    int pad = k_size / 2;
    size_t shmem_bytes = (size_t)(blockSize.x + 2 * pad) * (blockSize.y + 2 * pad) * sizeof(double);

    for (int iter = 0; iter < iterations; iter++) {
        convolution_kernel_shared<<<gridSize, blockSize, shmem_bytes>>>(d_current, d_next, rows, cols, k_size);

        double *tmp = d_current;
        d_current = d_next;
        d_next = tmp;
    }

    cudaMemcpy(final_out, d_current, img_bytes, cudaMemcpyDeviceToHost);

    free(h_kernel);
    cudaFree(d_current);
    cudaFree(d_next);
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

    cudaDeviceSynchronize();
    double start = omp_get_wtime();

    iterative_gaussian_blur_cuda_shared(image, rows, cols, k_size, iterations, sigma, output);

    cudaDeviceSynchronize();
    double end = omp_get_wtime();

    // Linha lida pelo script (Select-String "Tempo" + regex de decimal)
    printf("Modo: CUDA-Shared | Kernel: %d | Iteracoes: %d | Tempo: %.4f s\n", k_size, iterations, end - start);

    write_pgm(output_file, output, rows, cols);

    free_image(image);
    free_image(output);

    return 0;
}
