//Arthur e João

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <ctype.h>

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
    while (1) {
        c = fgetc(file);
        if (isspace(c)) {
            continue;
        } else if (c == '#') {
            while ((c = fgetc(file)) != '\n' && c != EOF);
        } else {
            ungetc(c, file);
            break;
        }
    }

    if (fscanf(file, "%d %d %d", cols, rows, &max_val) != 3) {
        printf("Erro: Arquivo PGM invalido (dimensoes)\n");
        fclose(file);
        exit(1);
    }

    fgetc(file); // Pula o último espaço/newline antes dos dados binários

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

void apply_convolution(double *image, int rows, int cols, double *kernel, int k_size, double *padded, double *output) {
    int pad = k_size / 2;
    int pad_h = rows + 2 * pad;
    int pad_w = cols + 2 * pad;

    for (int i = 0; i < pad_h; i++) {
        for (int j = 0; j < pad_w; j++) {
            int si = i - pad;
            int sj = j - pad;
            if (si < 0) si = 0;
            if (si >= rows) si = rows - 1;
            if (sj < 0) sj = 0;
            if (sj >= cols) sj = cols - 1;
            padded[i * pad_w + j] = image[si * cols + sj];
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int ki = 0; ki < k_size; ki++) {
                for (int kj = 0; kj < k_size; kj++) {
                    sum += padded[(i + ki) * pad_w + (j + kj)] * kernel[ki * k_size + kj];
                }
            }
            output[i * cols + j] = sum;
        }
    }
}

void iterative_gaussian_blur(double *image, int rows, int cols, int k_size, int iterations, double sigma, double *final_out) {
    double *kernel = (double *)malloc((size_t)k_size * k_size * sizeof(double));
    if (!kernel) {
        printf("Erro: Nao foi possivel alocar kernel\n");
        exit(1);
    }

    create_gaussian_kernel(k_size, sigma, kernel);

    int pad = k_size / 2;
    int pad_h = rows + 2 * pad;
    int pad_w = cols + 2 * pad;

    double *current = alloc_image(rows, cols);
    double *next = alloc_image(rows, cols);
    double *padded = alloc_image(pad_h, pad_w);

    memcpy(current, image, (size_t)rows * cols * sizeof(double));

    for (int iter = 0; iter < iterations; iter++) {
        apply_convolution(current, rows, cols, kernel, k_size, padded, next);
        double *tmp = current;
        current = next;
        next = tmp;
    }

    memcpy(final_out, current, (size_t)rows * cols * sizeof(double));

    free(kernel);
    free_image(current);
    free_image(next);
    free_image(padded);
}

int main(int argc, char *argv[]) {
    
    if (argc == 1) {
        
        int rows = 5;
        int cols = 5;
        double initial_data[] = {
            1, 2, 3, 2, 1,
            2, 4, 6, 4, 2,
            3, 6, 9, 6, 3,
            2, 4, 6, 4, 2,
            1, 2, 3, 2, 1
        };

        double *image = alloc_image(rows, cols);
        for (int i = 0; i < rows * cols; i++) {
            image[i] = initial_data[i];
        }

        printf("Imagem original:\n[");
        for (int i = 0; i < rows; i++) {
            if (i > 0) printf(" ");
            printf("[");
            for (int j = 0; j < cols; j++) {
                printf("%g", image[i * cols + j]);
                if (j < cols - 1) printf(". ");
                else if (i == rows - 1) printf("]");
                else printf("]\n");
            }
        }
        printf("]\n");

        int k_size = 3;
        int iterations = 2;
        double sigma = 1.0;
        double *output = alloc_image(rows, cols);

        iterative_gaussian_blur(image, rows, cols, k_size, iterations, sigma, output);

        printf("\nImagem após filtro (arredondada):\n[");
        for (int i = 0; i < rows; i++) {
            if (i > 0) printf(" ");
            printf("[");
            for (int j = 0; j < cols; j++) {
                printf("%.2f", output[i * cols + j]);
                if (j < cols - 1) printf(" ");
                else if (i == rows - 1) printf("]");
                else printf("]\n");
            }
        }
        printf("]\n");

        free_image(image);
        free_image(output);

        return 0;
    }

    if (argc < 4) {
        printf("Uso: %s <caminho_imagem> <tamanho_label> <kernel_size>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    char *img_label = argv[2]; // Ex: "512"
    int k_size = atoi(argv[3]);
    
    int iterations = 1000; // Mantenha igual ao teste paralelo
    double sigma = 1.0;
    int rows, cols;

    // Carregamento

    double *image = read_pgm(input_file, &rows, &cols);
    double *output = alloc_image(rows, cols);

    // Medição de tempo (usando omp_get_wtime para precisão total)

    double start = omp_get_wtime();
    iterative_gaussian_blur(image, rows, cols, k_size, iterations, sigma, output);
    double end = omp_get_wtime();

    // SAÍDA FORMATADA: Arquivo,Kernel,Threads,Tempo_s
    // Usamos '0' em threads para identificar o Serial no CSV

    printf("%s, %d, 0, %.4f\n", img_label, k_size, end - start);

    // Salva a imagem de saída em PGM

    char out_filename[256];
    snprintf(out_filename, sizeof(out_filename), "%s_k%d_out.pgm", img_label, k_size);
    write_pgm(out_filename, output, rows, cols);

    // Limpeza
    free_image(image);
    free_image(output);

    return 0;
}