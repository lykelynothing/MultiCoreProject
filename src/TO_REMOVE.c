
/* Calls mpi to receive struct and quantized vector. Handles
 * all kinds of struct and attaches to vec field the received
 * quantized vector (uint8). Returns pointer to received struct */
// Receives an already allocated struct
void * Receive(int algo, int dim, int source, void * void_ptr){
  MPI_Datatype * type_ptr;

  switch(algo){
  // lloyd also contains codebook
  case 0:
    struct lloyd_max_quant * str_ptr1 = (struct lloyd_max_quant *) void_ptr;
    
    MPI_Recv(str_ptr1->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr1->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr1->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    break;
  case 1:
    struct non_linear_quant * str_ptr2 = (struct non_linear_quant*) void_ptr;

    MPI_Recv(str_ptr2 -> min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr2 -> max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr2 -> vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    break; 
  case 2:
    struct unif_quant * str_ptr3 = (struct unif_quant *) void_ptr;

    MPI_Recv(&str_ptr3->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&str_ptr3->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr3->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    break;
  case 3:
    struct unif_quant * str_ptr4 = (struct unif_quant *) void_ptr;

    MPI_Recv(&str_ptr4->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&str_ptr4->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr4->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    break;
  default:
    printf("ERROR!! Quant algo not valid\n");
    break;
  }

  return void_ptr;
}


/* Sends the struct and its vec field (and codebook with LLOYD) to dest.
 * Remeber to deallocate space outside of function. */
int Send(void * struct_ptr, int algo, int dim, int dest){
  MPI_Datatype * type_ptr;

  switch(algo){
    // lloyd also contains codebook
    case 0:
      struct unif_quant * str_ptr1 = (struct lloyd_max_quant *) struct_ptr;
      MPI_Send(&str_ptr1->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr1->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Type_free(type_ptr);
      break;

    case 1:
      struct unif_quant * str_ptr2 = (struct non_linear_quant_quant *) struct_ptr;
      MPI_Send(&str_ptr2->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr2->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr2->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      break;
    case 2:
      struct unif_quant * str_ptr3 = (struct unif_quant *) struct_ptr;
      MPI_Send(&str_ptr3->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr3->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr3->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      break;
    case 3:
      struct unif_quant * str_ptr4 = (struct unif_quant *) struct_ptr;
      MPI_Send(&str_ptr4->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr4->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr4->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      break;
    default:
      printf("ERROR!! Quant algo not valid (send_call)\n");
      break;
  }
  
  return MPI_SUCCESS;
}
