#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>

constexpr int N1 = 3;  // Первая матрица 3x3
constexpr int N2 = 4;  // Вторая матрица 4x4

void fillRandom(double* mat, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 5);
    for (int i = 0; i < rows * cols; i++) mat[i] = dis(gen);
}

void printMatrix(double* mat, int rows, int cols, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) 
            std::cout << mat[i * cols + j] << " ";
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<double> A(N1 * N1), B(N2 * N2), C(N1 * N2, 0);
    
    if (rank == 0) {
        fillRandom(A.data(), N1, N1);
        fillRandom(B.data(), N2, N2);
        printMatrix(A.data(), N1, N1, "Matrix A (3x3)");
        printMatrix(B.data(), N2, N2, "Matrix B (4x4)");
    }
    
    // Рассылаем матрицу B всем процессам
    MPI_Bcast(B.data(), N2 * N2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Распределяем строки матрицы A
    int rows_per_proc = N1 / size;
    int extra_rows = N1 % size;
    int my_rows = rows_per_proc + (rank < extra_rows ? 1 : 0);
    int start_row = 0;
    
    // Вычисляем стартовую строку для каждого процесса
    for (int i = 0; i < rank; i++) {
        start_row += rows_per_proc + (i < extra_rows ? 1 : 0);
    }
    
    std::vector<double> my_part(my_rows * N1);
    
    // Распределяем строки матрицы A
    if (rank == 0) {
        // Процесс 0 копирует свою часть
        for (int i = 0; i < my_rows; i++) {
            for (int j = 0; j < N1; j++) {
                my_part[i * N1 + j] = A[(start_row + i) * N1 + j];
            }
        }
        // Отправляем остальным процессам их части
        for (int p = 1; p < size; p++) {
            int p_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
            int p_start = 0;
            for (int i = 0; i < p; i++) p_start += rows_per_proc + (i < extra_rows ? 1 : 0);
            
            for (int r = 0; r < p_rows; r++) {
                MPI_Send(&A[(p_start + r) * N1], N1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        // Получаем свои строки от процесса 0
        for (int r = 0; r < my_rows; r++) {
            MPI_Recv(&my_part[r * N1], N1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // Каждый процесс умножает свои строки на матрицу B
    std::vector<double> my_result(my_rows * N2, 0);
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < N2; j++) {
            for (int k = 0; k < N1; k++) {
                my_result[i * N2 + j] += my_part[i * N1 + k] * B[k * N2 + j];
            }
        }
    }
    
    // Собираем результаты в процессе 0
    if (rank == 0) {
        // Копируем результат процесса 0
        for (int i = 0; i < my_rows; i++) {
            for (int j = 0; j < N2; j++) {
                C[(start_row + i) * N2 + j] = my_result[i * N2 + j];
            }
        }
        // Получаем результаты от других процессов
        for (int p = 1; p < size; p++) {
            int p_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
            int p_start = 0;
            for (int i = 0; i < p; i++) p_start += rows_per_proc + (i < extra_rows ? 1 : 0);
            
            std::vector<double> p_result(p_rows * N2);
            MPI_Recv(p_result.data(), p_rows * N2, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < p_rows; i++) {
                for (int j = 0; j < N2; j++) {
                    C[(p_start + i) * N2 + j] = p_result[i * N2 + j];
                }
            }
        }
        
        printMatrix(C.data(), N1, N2, "Result C (3x4)");
    } else {
        // Отправляем результаты процессу 0
        MPI_Send(my_result.data(), my_rows * N2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}
