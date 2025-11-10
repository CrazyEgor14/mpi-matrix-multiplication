CXX = mpic++
TARGET = matrix_simple
SOURCE = matrix_simple.cpp

$(TARGET): $(SOURCE)
	$(CXX) -o $(TARGET) $(SOURCE)

run: $(TARGET)
	mpiexec -n 3 ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
