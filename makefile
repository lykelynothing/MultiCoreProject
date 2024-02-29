CC = mpicc
CFLAGS = -g -Wall -Wno-unknown-pragmas -Werror -fopenmp
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj

PROCESSES = -n 2
VEC_DIM = 1000

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
EXEC = out

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

run:
	mpiexec $(PROCESSES) ./$(EXEC) $(VEC_DIM)

clean:
	rm -rf $(OBJ_DIR) $(EXEC)
