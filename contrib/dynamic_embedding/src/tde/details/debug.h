#pragma once
#include <stdio.h>

#define TDE_IS_DEBUGGING

#ifdef TDE_IS_DEBUGGING
#define TDE_DEBUG() fprintf(stderr, __FILE__ ":%d\n", __LINE__)
#else
#define TDE_DEBUG() \
  do {              \
  } while (0)
#endif
