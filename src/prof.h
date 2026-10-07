// In-graph time stamps (Nsight Systems loses most kernels with this driver). With Q27_PROF=1, prof::mark()
// adds a one-thread kernel that writes %globaltimer into a slot of a per-device buffer. Slots are assigned at
// capture time, so a captured graph writes the same slots on every launch. collect() reads the slots of one
// range after a launch and adds each gap (stamp minus the stamp before it) to the label of the later stamp.
// Each stamp adds about 1 us to the pass, so compare totals with Q27_PROF off.
#pragma once
#include <cuda_runtime.h>
#include <string>

namespace q27::prof {

bool on();
// Stamp now on stream s (current device). Returns the slot, or -1 when profiling is off.
int mark(cudaStream_t s, const char* label);
// Eager (non-graph) marks of the current device go to a separate slot range; eager(true) also forgets the old ones.
// Use: eager(true), marks, run, collect(eager_base(), next_slot()), eager(false).
void eager(bool on);
int eager_base();
// Next free slot on the current device (use before and after a capture to get its range).
int next_slot();
// Copy slots [a, b) of the current device and add the gaps to the per-label totals (one sample).
void collect(int a, int b);
// Print totals per label (ms per sample), sorted by time, and clear them.
void report(const char* title);

}  // namespace q27::prof
