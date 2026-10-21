# Writes OUT with the checkout's short commit as BASALT_BENCH_GIT and whether tracked files differ from it as
# BASALT_BENCH_DIRTY.  The file is only rewritten when the commit
# changed, so an unchanged build recompiles nothing.
execute_process(COMMAND git -C ${SRC} rev-parse --short=10 HEAD
                OUTPUT_VARIABLE _git OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT _git)
  set(_git unknown)
endif()
execute_process(COMMAND git -C ${SRC} status --porcelain --untracked-files=no
                OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(_dirty)
  set(_dirty 1)
else()
  set(_dirty 0)
endif()
set(_text "#define BASALT_BENCH_GIT \"${_git}\"\n#define BASALT_BENCH_DIRTY ${_dirty}\n")
set(_old "")
if(EXISTS ${OUT})
  file(READ ${OUT} _old)
endif()
if(NOT _old STREQUAL _text)
  file(WRITE ${OUT} "${_text}")
endif()
