CC = nvcc

# Compiler flags //unecessary compiler flags // -Wall: warnings -g: debug info
CFLAGS = -Xcudafe --diag_suppress=2464

# Source files
SRCS = stencil.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Executable name
TARGET = assignment5

# Default target
all: $(TARGET)

# Linking the executable // $@ sourcefile $^ all files/prerequesites
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compiling .c to .o //$< sourcefile $@ generated file name
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up // build clean
clean:
	rm -f $(OBJS) $(TARGET)
.PHONY: all clean
