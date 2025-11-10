#include <iostream>
#include <vector>
#include <random>
#include <mpi.h>

constexpr int N1 = 3;
constexpr int N2 = 4;

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
    std::cout << "\n";
}

void singleThreadMultiply(double* A, double* B, double* C) {
    std::cout << "=== ВЫЧИСЛЕНИЕ ОДНОПОТОЧНЫМ МЕТОДОМ ===\n";
    for (int i = 0; i < N1; i++) {
        std::cout << "Вычисление строки " << i << ":\n";
        for (int j = 0; j < N2; j++) {
            C[i * N2 + j] = 0;
            std::cout << "  Столбец " << j << ": ";
            for (int k = 0; k < N1; k++) {
                C[i * N2 + j] += A[i * N1 + k] * B[k * N2 + j];
                std::cout << A[i * N1 + k] << "*" << B[k * N2 + j];
                if (k < N1 - 1) std::cout << " + ";
            }
            std::cout << " = " << C[i * N2 + j] << "\n";
        }
        std::cout << "Результат строки " << i << ": ";
        for (int j = 0; j < N2; j++) {
            std::cout << C[i * N2 + j] << " ";
        }
        std::cout << "\n\n";
    }
}

void multiThreadMultiply(int rank, int size, double* A, double* B, double* C) {
    if (rank == 0) {
        std::cout << "=== ВЫЧИСЛЕНИЕ МНОГОПОТОЧНЫМ МЕТОДОМ ===\n";
        std::cout << "Распределение работы между " << size << " процессами:\n";
        
        // Рассылаем матрицу B и строки A ВСЕМ процессам (включая процесс 0 для его строки)
        for (int i = 0; i < size && i < N1; i++) {
            if (i == 0) {
                // Процесс 0 уже имеет матрицу B, ничего не отправляем себе
                continue;
            }
            MPI_Send(B, N2 * N2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&A[i * N1], N1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            std::cout << "Отправлена строка " << i << " процессу " << i << "\n";
        }
        
        // Процесс 0 вычисляет строку 0
        std::cout << "\nПроцесс 0 вычисляет строку 0:\n";
        for (int j = 0; j < N2; j++) {
            C[0 * N2 + j] = 0;  // ← ИСПРАВЛЕНО: записываем в строку 0
            std::cout << "  Столбец " << j << ": ";
            for (int k = 0; k < N1; k++) {
                C[0 * N2 + j] += A[0 * N1 + k] * B[k * N2 + j];
                std::cout << A[0 * N1 + k] << "*" << B[k * N2 + j];
                if (k < N1 - 1) std::cout << " + ";
            }
            std::cout << " = " << C[0 * N2 + j] << "\n";
        }
        std::cout << "Результат строки 0: ";
        for (int j = 0; j < N2; j++) {
            std::cout << C[0 * N2 + j] << " ";
        }
        std::cout << "\n";
        
        // Получаем результаты от других процессов
        std::cout << "\nПолучение результатов от других процессов:\n";
        for (int i = 1; i < size && i < N1; i++) {
            MPI_Recv(&C[i * N2], N2, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Получена строка " << i << " от процесса " << i << ": ";
            for (int j = 0; j < N2; j++) {
                std::cout << C[i * N2 + j] << " ";
            }
            std::cout << "\n";
        }
        
    } else {
        // Рабочие процессы (rank 1, 2, ...)
        if (rank < N1) {  // Только если есть строка для этого процесса
            std::vector<double> B_recv(N2 * N2);
            std::vector<double> my_row(N1);
            
            MPI_Recv(B_recv.data(), N2 * N2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(my_row.data(), N1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::cout << "Процесс " << rank << " получил строку " << rank << ": ";
            for (int k = 0; k < N1; k++) {
                std::cout << my_row[k] << " ";
            }
            std::cout << "\n";
            
            std::vector<double> my_result(N2, 0);
            std::cout << "Процесс " << rank << " вычисляет строку " << rank << ":\n";
            for (int j = 0; j < N2; j++) {
                my_result[j] = 0;
                std::cout << "  Столбец " << j << ": ";
                for (int k = 0; k < N1; k++) {
                    my_result[j] += my_row[k] * B_recv[k * N2 + j];
                    std::cout << my_row[k] << "*" << B_recv[k * N2 + j];
                    if (k < N1 - 1) std::cout << " + ";
                }
                std::cout << " = " << my_result[j] << "\n";
            }
            
            MPI_Send(my_result.data(), N2, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            std::cout << "Процесс " << rank << " отправил результат строки " << rank << "\n";
        }
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
        
        std::cout << "УМНОЖЕНИЕ МАТРИЦ " << N1 << "x" << N1 << " НА " << N2 << "x" << N2 << "\n";
        std::cout << "========================================\n\n";
        
        printMatrix(A.data(), N1, N1, "Матрица A");
        printMatrix(B.data(), N2, N2, "Матрица B");
        
        // Однопоточный подход
        singleThreadMultiply(A.data(), B.data(), C.data());
        printMatrix(C.data(), N1, N2, "Финальный результат C (однопоточный)");
        
        std::cout << "========================================\n\n";
        
        // Многопоточный подход
        std::fill(C.begin(), C.end(), 0);
        multiThreadMultiply(rank, size, A.data(), B.data(), C.data());
        printMatrix(C.data(), N1, N2, "Финальный результат C (многопоточный)");
        
    } else {
        multiThreadMultiply(rank, size, A.data(), B.data(), C.data());
    }
    
    MPI_Finalize();
    return 0;
}
