#pragma once

#include "kernel/head/optis.cuh"

void * CROSSENTROPY_space_mk(Opti_t * opti);
void CROSSENTROPY_free(Opti_t * opti);

void CROSSENTROPY_dloss(Opti_t * opti);
void CROSSENTROPY_score(Opti_t * opti);