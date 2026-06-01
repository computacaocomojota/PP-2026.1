//Arthur e João

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>



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

void apply_convolution_mpi(double *local_current_with_halos, int local_rows, int cols, double *kernel, int k_size, double *padded, double *output) {
    int pad = k_size / 2;
    int pad_h = local_rows + 2 * pad;
    int pad_w = cols + 2 * pad;

    for (int i = 0; i < pad_h; i++) {
        for (int j = 0; j < pad_w; j++) {
            int sj = j - pad;
            if (sj < 0) sj = 0;           
            if (sj >= cols) sj = cols - 1; 
            
            padded[i * pad_w + j] = local_current_with_halos[i * cols + sj];
        }
    }

  
    for (int i = 0; i < local_rows; i++) { 
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

        apply_convolution_mpi(current, rows, cols, kernel, k_size, padded, next);
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

void iterative_gaussian_blur_mpi(double *image, int rows, int cols, int k_size, int iterations, double sigma, double *output, int rank, int size) {

    int pad = k_size / 2;
    int *sendcounts = NULL;
    int *displs = NULL;
    double *kernel = (double *)malloc((size_t)k_size * k_size * sizeof(double));
    
    if (rank == 0) {

        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        create_gaussian_kernel(k_size, sigma, kernel);
        
        int offset = 0;
        for (int i = 0; i < size; i++) {

            int r_rows = (rows / size) + (i < (rows % size) ? 1 : 0);
            sendcounts[i] = r_rows * cols;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Bcast(kernel, k_size * k_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int local_rows = (rows / size) + (rank < (rows % size) ? 1 : 0);

    int pad_h = local_rows + 2 * pad;
    int pad_w = cols + 2 * pad;
    double *local_padded = (double *)calloc(pad_h * pad_w, sizeof(double));

    double *local_current = (double *)calloc(local_rows * cols, sizeof(double));

    MPI_Scatterv(image, sendcounts, displs, MPI_DOUBLE, local_current, local_rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int total_local_elements = (local_rows + 2 * pad) * cols;
    double *local_current_with_halos = (double *)calloc(total_local_elements, sizeof(double));
    double *local_next = (double *)calloc(local_rows * cols, sizeof(double));

    memcpy(&local_current_with_halos[pad * cols], local_current, local_rows * cols * sizeof(double));

    int target_top = rank - 1;
    int target_bottom = rank + 1;
    
    if (rank == 0) target_top = MPI_PROC_NULL;
    if (rank == size - 1) target_bottom = MPI_PROC_NULL;
    
    for (int iter = 0; iter < iterations; iter++) {
        
        // 3.1: Troca para o Halo Superior (Envia para cima, recebe de baixo)

        MPI_Sendrecv(&local_current_with_halos[pad * cols], pad * cols, MPI_DOUBLE, target_top, 0,
                     &local_current_with_halos[(pad + local_rows) * cols], pad * cols, MPI_DOUBLE, target_bottom, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 3.2: Troca para o Halo Inferior (Envia para baixo, recebe de cima)

        MPI_Sendrecv(&local_current_with_halos[local_rows * cols], pad * cols, MPI_DOUBLE, target_bottom, 1,
                     &local_current_with_halos[0], pad * cols, MPI_DOUBLE, target_top, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 3.3: Tratamento das Bordas Físicas

        if (rank == 0) {

            // Replicar a primeira linha real (linha pad) para o halo superior

            for (int p = 0; p < pad; p++) {
                memcpy(&local_current_with_halos[p * cols], &local_current_with_halos[pad * cols], cols * sizeof(double));
            }
        }
        if (rank == size - 1) {

            // Replicar a última linha real para o halo inferior

            for (int p = 0; p < pad; p++) {
                int dest_idx = (pad + local_rows + p) * cols;
                int src_idx = (pad + local_rows - 1) * cols;
                memcpy(&local_current_with_halos[dest_idx], &local_current_with_halos[src_idx], cols * sizeof(double));
            }
        }

        apply_convolution_mpi(local_current_with_halos, local_rows, cols, kernel, k_size, local_padded, local_next);

        memcpy(&local_current_with_halos[pad * cols], local_next, local_rows * cols * sizeof(double));
    }

    memcpy(local_current, &local_current_with_halos[pad * cols], local_rows * cols * sizeof(double));

    MPI_Gatherv(local_current, local_rows * cols, MPI_DOUBLE, output, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Limpeza de memória local de cada processo

    free(local_current);
    free(local_current_with_halos);
    free(local_next);
    free(local_padded); 
    free(kernel);       
    
    if (rank == 0) {

        free(sendcounts);
        free(displs);
    }
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4)  {

        if(rank == 0)printf("Uso: %s <caminho_imagem> <tamanho_label> <kernel_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    double start, end;
    
    double *image = NULL ,*output = NULL;
    char *input_file = argv[1];
    char *img_label = argv[2]; // Ex: "512"
    int k_size = atoi(argv[3]);
    
    int iterations = 1000; // Mantenha igual ao teste paralelo
    double sigma = 1.0;
    int rows=0, cols=0;

    // Carregamento

    if(rank == 0){
        image = read_pgm(input_file, &rows, &cols);
        output = alloc_image(rows, cols);
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Medição de tempo (usando omp_get_wtime para precisão total)

    if(rank == 0)start = MPI_Wtime();

    iterative_gaussian_blur_mpi(image, rows, cols, k_size, iterations, sigma, output,rank,size);
    
    //MPI_Barrier(MPI_COMM_WORLD);
    

    // SAÍDA FORMATADA: Arquivo,Kernel,Threads,Tempo_s
    // Usamos '0' em threads para identificar o Serial no CSV
    if (rank == 0)
    {
        end = MPI_Wtime();
        printf("%s, %d, %d, %.4f\n", img_label, k_size, size, end - start);

        // Salva a imagem de saída em PGM
        //char out_filename[256];
        //snprintf(out_filename, sizeof(out_filename), "%s_k%d_out.pgm", img_label, k_size);
        //write_pgm(out_filename, output, rows, cols);

        // Limpeza
        free_image(image);
        free_image(output);

    }

    MPI_Finalize();

    return 0;
}