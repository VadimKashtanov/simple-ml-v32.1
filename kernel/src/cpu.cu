#include "kernel/head/cpu.cuh"

Cpu_t* cpu_mk(Mdl_t * mdl, uint times) {
	Cpu_t * ret = (Cpu_t*)malloc(sizeof(Cpu_t));
	ret->mdl = mdl;
	ret->times = times;
	ret->var = (float*)malloc(sizeof(float) * times * mdl->total);
	return ret;
};

void cpu_set_input(Cpu_t * cpu, Data_t * data) {
	for (uint t=0; t < cpu->times; t++)
		memcpy(cpu->var + t*cpu->mdl->total, data->input + t*cpu->mdl->inputs, sizeof(float) * cpu->mdl->inputs);
};

void cpu_forward(Cpu_t * cpu) {
	for (uint t=0; t < cpu->times; t++)
		for (uint i=0; i < cpu->mdl->insts; i++)
			INST_CPU[cpu->mdl->id[i]](cpu, i, t);
};

void cpu_free(Cpu_t * cpu) {
	free(cpu->var);
	free(cpu);
};