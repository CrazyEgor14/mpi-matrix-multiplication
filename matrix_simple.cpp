#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <mpi.h>

void fillRandom(double* mat, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 9);
    for (int i = 0; i < size * size; i++) mat[i] = dis(gen);
}

void printMatrix(double* mat, int rows, int cols, const char* name) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) 
            std::cout << mat[i * cols + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

void singleThreadMultiply(double* A, double* B, double* C, int n1, int n2) {
    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            C[i * n2 + j] = 0;
            for (int k = 0; k < n1; k++) {
                C[i * n2 + j] += A[i * n1 + k] * B[k * n2 + j];
            }
        }
    }
}

void multiThreadMultiply(int rank, int size, double* A, double* B, double* C, int n1, int n2) {
    if (rank == 0) {
        // Рассылаем матрицу B всем процессам
        for (int i = 1; i < size; i++) {
            MPI_Send(B, n2 * n2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        
        // Распределяем строки матрицы A
        for (int i = 1; i < size; i++) {
            int start_row = (i - 1) * n1 / (size - 1);
            int end_row = i * n1 / (size - 1);
            int rows_to_send = end_row - start_row;
            
            // Отправляем количество строк
            MPI_Send(&rows_to_send, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            // Отправляем стартовую строку
            MPI_Send(&start_row, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            // Отправляем сами строки
            for (int r = 0; r < rows_to_send; r++) {
                MPI_Send(&A[(start_row + r) * n1], n1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD);
            }
        }
        
        // Процесс 0 тоже вычисляет свою часть
        int start_row0 = (size - 1) * n1 / (size - 1);
        int rows0 = n1 - start_row0;
        for (int i = 0; i < rows0; i++) {
            for (int j = 0; j < n2; j++) {
                C[(start_row0 + i) * n2 + j] = 0;
                for (int k = 0; k < n1; k++) {
                    C[(start_row0 + i) * n2 + j] += A[(start_row0 + i) * n1 + k] * B[k * n2 + j];
                }
            }
        }
        
        // Получаем результаты от других процессов
        for (int i = 1; i < size; i++) {
            int rows_to_receive;
            MPI_Recv(&rows_to_receive, 1, MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int start_row;
            MPI_Recv(&start_row, 1, MPI_INT, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int r = 0; r < rows_to_receive; r++) {
                MPI_Recv(&C[(start_row + r) * n2], n2, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Получаем матрицу B
        MPI_Recv(B, n2 * n2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Получаем информацию о строках
        int rows_to_process;
        MPI_Recv(&rows_to_process, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        int start_row;
        MPI_Recv(&start_row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Получаем строки матрицы A
        std::vector<double> my_rows(rows_to_process * n1);
        for (int r = 0; r < rows_to_process; r++) {
            MPI_Recv(&my_rows[r * n1], n1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // Вычисляем результат
        std::vector<double> my_result(rows_to_process * n2);
        for (int i = 0; i < rows_to_process; i++) {
            for (int j = 0; j < n2; j++) {
                my_result[i * n2 + j] = 0;
                for (int k = 0; k < n1; k++) {
                    my_result[i * n2 + j] += my_rows[i * n1 + k] * B[k * n2 + j];
                }
            }
        }
        
        // Отправляем результаты обратно
        MPI_Send(&rows_to_process, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
        MPI_Send(&start_row, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
        for (int r = 0; r < rows_to_process; r++) {
            MPI_Send(&my_result[r * n2], n2, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
        }
    }
}

void runTest(int matrix_size, bool use_mpi, int num_processes) {
    int n1 = matrix_size;
    int n2 = matrix_size;
    
    std::vector<double> A(n1 * n1), B(n2 * n2), C(n1 * n2, 0);
    
    fillRandom(A.data(), n1);
    fillRandom(B.data(), n2);
    
    std::cout << "=== Матрица " << n1 << "x" << n1 << " - " 
              << (use_mpi ? "Многопоточный подход" : "Однопоточный подход") << " ===\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (use_mpi) {
        multiThreadMultiply(0, num_processes, A.data(), B.data(), C.data(), n1, n2);
    } else {
        singleThreadMultiply(A.data(), B.data(), C.data(), n1, n2);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (!use_mpi || (use_mpi && num_processes == 1)) {
        printMatrix(A.data(), n1, n1, "Матрица A");
        printMatrix(B.data(), n2, n2, "Матрица B");
        printMatrix(C.data(), n1, n2, "Результат C");
    }
    
    std::cout << "Время выполнения: " << duration.count() << " микросекунд\n";
    std::cout << "========================================\n\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "Количество процессов: " << size << "\n\n";
        
        // Вариант 1: Многопоточный подход с матрицей 3x3
        runTest(3, true, size);
        
        // Вариант 2: Однопоточный подход с матрицей 3x3  
        runTest(3, false, 1);
        
        // Вариант 3: Многопоточный подход с матрицей 4x4
        runTest(4, true, size);
        
        // Вариант 4: Однопоточный подход с матрицей 4x4
        runTest(4, false, 1);
    } else {
        // Рабочие процессы
        for (int test = 0; test < 2; test++) { // Два многопоточных теста
            std::vector<double> B(16); // Максимальный размер 4x4
            
            // Получаем матрицу B
            MPI_Recv(B.data(), 16, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Получаем информацию о строках
            int rows_to_process, start_row;
            MPI_Recv(&rows_to_process, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&start_row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Получаем строки матрицы A
            std::vector<double> my_rows(rows_to_process * 4); // Максимальный размер
            for (int r = 0; r < rows_to_process; r++) {
                MPI_Recv(&my_rows[r * 4], 4, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            
            // Вычисляем результат
            std::vector<double> my_result(rows_to_process * 4);
            int n1 = 4; // Максимальный размер
            int n2 = 4;
            
            for (int i = 0; i < rows_to_process; i++) {
                for (int j = 0; j < n2; j++) {
                    my_result[i * n2 + j] = 0;
                    for (int k = 0; k < n1; k++) {
                        my_result[i * n2 + j] += my_rows[i * n1 + k] * B[k * n2 + j];
                    }
                }
            }
            
            // Отправляем результаты обратно
            MPI_Send(&rows_to_process, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
            MPI_Send(&start_row, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
            for (int r = 0; r < rows_to_process; r++) {
                MPI_Send(&my_result[r * n2], n2, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
