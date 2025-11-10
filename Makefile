CXX = mpic++
TARGET = matrix_test
SOURCE = matrix_test.cpp

$(TARGET): $(SOURCE)
	$(CXX) -o $(TARGET) $(SOURCE)

run-1:
	mpiexec -n 1 ./$(TARGET)

run-2:
	mpiexec -n 2 ./$(TARGET)

run-3:
	mpiexec -n 3 ./$(TARGET)

run-4:
	mpiexec -n 4 ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run-1 run-2 run-3 run-4 clean
