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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== Запущено процессов: " << size << " ===\n";
        
        // Тест с матрицей 3x3
        int n1 = 4, n2 = 4;
        std::vector<double> A(n1 * n1), B(n2 * n2), C(n1 * n2, 0);
        
        fillRandom(A.data(), n1);
        fillRandom(B.data(), n2);
        
        std::cout << "Матрица 3x3 - " << (size > 1 ? "Многопоточный" : "Однопоточный") << " подход\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        singleThreadMultiply(A.data(), B.data(), C.data(), n1, n2);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        printMatrix(A.data(), n1, n1, "Матрица A");
        printMatrix(B.data(), n2, n2, "Матрица B");
        printMatrix(C.data(), n1, n2, "Результат C");
        std::cout << "Время: " << duration.count() << " микросекунд\n\n";
    }
    
    MPI_Finalize();
    return 0;
}
