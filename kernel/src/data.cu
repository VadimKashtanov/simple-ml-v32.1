#include "kernel/head/data.cuh"

Data_t * data_open(char * file) {
	FILE * fp = fopen(file, "rb");

	float batchs, lines, inputs, outputs;

	fread(&batchs, sizeof(uint), 1, fp);
	fread(&lines, sizeof(uint), 1, fp);
	fread(&inputs, sizeof(uint), 1, fp);
	fread(&outputs, sizeof(uint), 1, fp);

	fclose(fp);

	return data_load(inputs, outputs, lines);
};

Data_t * data_load(uint inputs, uint outputs, uint lines) {
	Data_t * ret = (Data_t*)malloc(sizeof(Data_t));

	ret->inputs = inputs;
	ret->outputs = outputs;
	ret->lines = lines;

	ret->input = (float*)malloc(sizeof(float) * ret->lines * ret->inputs);
	ret->output = (float*)malloc(sizeof(float) * ret->lines * ret->outputs);

	ret->input_d = 0;
	ret->output_d = 0;

	return retl
};

void data_cudmalloc(Data_t * data) {
	SAFE_CUDA(cudaMalloc((void**)&ret->input_d, sizeof(float) * data->lines * data->inputs));
	SAFE_CUDA(cudaMalloc((void**)&ret->output_d, sizeof(float) * data->lines * data->outputs));
};

void data_load_batch(Data_t * data, FILE * fp, uint batchs, uint batch) {
	//	Seek to input `batch` batch
	fseek(data->fp,
		sizeof(uint)*4 + sizeof(float)*(batch * data->lines*data->inputs),
		SEEK_SET);
	fread(data->input, sizeof(float), data->lines*data->inputs, fp);

	//	Seek to output `batch` batch
	fseek(data->fp,
		sizeof(uint)*4 + sizeof(float)*(batchs*data->lines*data->inputs + batch*data->lines*data->outputs),
		SEEK_SET);
	fread(data->output, sizeof(float), data->lines*data->outputs, fp);
};

void data_cudamemcpy(Data_t * data) {
	SAFE_CUDA(cudaMemcpy(
		data->input_d,
		data->input,
		sizeof(float) * data->inputs * data->lines,
		cudaMemcpyHostToDevice))

	SAFE_CUDA(cudaMemcpy(
		data->output_d,
		data->output,
		sizeof(float) * data->outputs * data->lines,
		cudaMemcpyHostToDevice))
};

void data_free(Data_t * data) {
	if (data->input) free(data->input);
	if (data->output) free(data->output);
	if (data->input_d) SAFE_CUDA(cudaFreee(data->input_d));
	if (data->output_d) SAFE_CUDA(cudaFree(data->output_d));

	free(data);
};