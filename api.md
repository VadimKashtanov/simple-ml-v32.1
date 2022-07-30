## Etc.cuh ##

```c
#define compare_floats(a, b, p) (fabs(a-b) < p)

#define pseudo_randomi(seed) ((123456*(seed+12345))% 0x100000000 )
#define pseudo_randomf(seed) (pseudo_randomi(seed) / 0x100000000 )

#define MSG(str, ...) printf("[\033[35;1;41mWarrning\033[0m]:\033[96m%s:(%d)\033[35m: " str "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__);
#define ERR(str, ...) do {printf("[\033[30;101mError\033[0m]:\033[96m%s:(%d)\033[30m: " str "\033[0m\n", __FILE__, __LINE__, ##__VA_ARGS__);raise(SIGINT);} while (0);

__constant__ float const_mem[1 << 14];

extern __shared__ float dynamic__shared__[];	//no needs for size

#define SAFE_CUDA(call) do { cudaError_t err = call; if (err != cudaSuccess)ERR("Cuda Error : %s", cudaGetErrorString(err));} while(0);

#define KERN_DIV(elements, thx) (((elements - elements%thx)/thx)+1)

void etc_parse_arguments(uint argc, char ** argv, uint paramc, char ** paramv, char ** correspondance);
```

## Data.cuh ##

```c
typedef struct data_struct {
	uint inputs, outputs;
	uint lines;

	float * input, * output;

	float * input_d, * output_d;
} Data_t;

Data_t * data_open(char * file);
Data_t * data_load(uint inputs, uint outputs, uint lines);
void data_cudmalloc(Data_t * data);

void data_open_batch(Data_t * data, char * file, uint batch);
void data_load_batch(Data_t * data, FILE * fp, uint batch);
void data_cudamemcpy(Data_t * data);

void data_free(Data_t * data);
```

## Mdl.cuh ##

```c
typedef struct {
	uint insts;
	uint * id;
	uint ** param;

	uint inputs;
	uint outputs;
	
	uint vars, weights, locds;
	uint total;

	float * weight;
	float * step;
} Mdl_t;

Mdl_t* mdl_fp_load(FILE * fp);
void mdl_fp_write(Mdl_t * mdl, FILE * fp);

void mdl_check_correctness(Mdl_t * mdl);

void mdl_free(Mdl_t * mdl);	//useless

void mdl_print_inst(Mdl_t * mdl, uint inst);
```

# Cpu.cuh #

```c
typedef struct cpu_model_forward {
	Mdl_t * mdl;

	Data_t * data;

	float * var;
} Cpu_t;

Cpu_t* cpu_mk(Mdl_t * mdl, Data_t * data, uint times);
void cpu_free(Cpu_t * cpu);

void cpu_set_input(Cpu_t * cpu);
void cpu_forward(Cpu_t * cpu);
```

## Use.cuh ##

```c
typedef struct gpu_model_forward {
	Mdl_t * mdl;
	Data_t * data;

	float * weight_d;

	float * var_d;
} Use_t;

Use_t* use_mk(Mdl_t * mdl, Data_t * data);

void use_set_input(Use_t * use);
void use_forward(Use_t * use);

void use_free(Use_t * use);
```

## Train.cuh ##

```c
typedef struct train_model {
	Mdl_t * mdl;
	uint sets;
	
	Data_t * data;

	float * _weight;	//_weight[sets ][wsize]
	float * _var;		//   _var[times][sets ][vsize]
	float * _locd;		//  _locd[times][sets ][lsize]
	float * _grad;		//  _grad[times][sets ][vsize]
	float * _meand;		// _meand[sets ][wsize]
} Train_t;

Train_t* mk_train(Mdl_t * mdl, Data_t * data, uint sets);
void train_random_weights(Train_t * train);
void train_random_weights_from_mdl(Train_t * train);
void train_cpy_ws_to_mdl(Train_t * train, uint set);
Train_t * extract_to_new_train(Train_t * old, uint amount, uint * set_id);	//extract set[0], set[4], set[2], set[1] and set[23]  to the new train and in this order

void train_set_input(Train_t * train);
void train_null_grad_meand(Train_t * train);
void train_forward(Train_t * train, uint start_seed);
void train_backward(Train_t * train, uint start_seed);

void train_free(Train_t * train);
```

## Optis.cuh ##

```c
typedef struct optimizer_and_score {
	Train_t * train;

	float * set_score;
	float * set_score_d;
	uint * set_rank;
	uint * set_rank_d;
	uint * podium;

	uint score_algo, opti_algo;
	void * score_space, * opti_space;
} Opti_t;

Opti_t * opti_mk(Train_t * train, uint score_algo, uint gtic_algo);

void opti_score(Opti_t * opti);
void opti_dloss(Opti_t * opti);
void opti_opti(Opti_t * opti);

void opti_free(Opti_t * opti);
```

## Gtics.cuh ##

```c
typedef struct {
	Opti_t * opti;

	uint  gtic_algo;
	void * gtic_space;
} Gtic_t;

Gtic_t * gtic_mk(Opti_t * opti, uint gtic_algo);

void gtic_select(Gtic_t * gtic);

void gtic_free(Gtic_t * gtic);
```