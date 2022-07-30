#pragma once

//	This includes Global #define for both kernel and package headers
#include "package/meta.cuh"

//	This includes all the kernel headers
#include "kernel/head/gtics.cuh"

//	This include all the package headers
#include "package/head/insts.cuh"
#include "package/head/opti.cuh"
#include "package/head/gtic.cuh"

//	Arrays are declared in headers and writed in package/src/*.cu