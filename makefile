CC = mpicc
CFLAGS = -g -Wall -Werror -Wno-unknown-pragmas -fopenmp 
LDFLAGS = -lm

SRC_DIR = src
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))
EXEC = out
processes = 2
var = 100

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(EXEC)

run:
	mpiexec -n $(processes) ./$(EXEC) $(var)
