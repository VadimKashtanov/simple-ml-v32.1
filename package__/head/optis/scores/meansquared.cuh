#pragma once

#include "kernel/head/optis.cuh"

void * MEANSQUARED_space_mk(Opti_t * opti);
void MEANSQUARED_free(Opti_t * opti);

void MEANSQUARED_dloss(Opti_t * opti);
void MEANSQUARED_score(Opti_t * opti);