# Writes OUT with the checkout's short commit as BASALT_BENCH_GIT.  The file is only rewritten when the commit
# changed, so an unchanged build recompiles nothing.
execute_process(COMMAND git -C ${SRC} rev-parse --short=10 HEAD
                OUTPUT_VARIABLE _git OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT _git)
  set(_git unknown)
endif()
set(_text "#define BASALT_BENCH_GIT \"${_git}\"\n")
set(_old "")
if(EXISTS ${OUT})
  file(READ ${OUT} _old)
endif()
if(NOT _old STREQUAL _text)
  file(WRITE ${OUT} "${_text}")
endif()
