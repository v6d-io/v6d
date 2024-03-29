# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckLinkerFlag
---------------

Check whether the compiler supports a given link flag.

.. command:: check_linker_flag

  .. code-block:: cmake

    check_linker_flag(<lang> <flag> <var>)

Check that the link ``<flag>`` is accepted by the ``<lang>`` compiler without
a diagnostic.  Stores the result in an internal cache entry named ``<var>``.

This command temporarily sets the ``CMAKE_REQUIRED_LINK_OPTIONS`` variable
and calls the ``check_<lang>_source_compiles`` macro from the
``Check<lang>SourceCompiles`` module (:module:`CheckCSourceCompiles`,
:module:`CheckCSourceCompiles`, :module:`CheckCXXSourceCompiles`,
:module:`CheckOBJCSourceCompiles`, :module:`CheckOBJCXXSourceCompiles` or
:module:`CheckFortranSourceCompiles`).  See documentation of these
modules for a listing of variables that can otherwise modify the build.

The underlying implementation rely on :prop_tgt:`LINK_OPTIONS` property to
check the specified flag. The ``LINKER:`` prefix, as described in
:command:`target_link_options` command, can be used as well.

A positive result from this check indicates only that the compiler did not
issue a diagnostic message when given the link flag.  Whether the flag has any
effect or even a specific one is beyond the scope of this module.

.. note::
  Since the :command:`try_compile` command forwards flags from variables
  like :variable:`CMAKE_<LANG>_FLAGS`, unknown flags in such variables may
  cause a false negative for this check.
#]=======================================================================]

include_guard(GLOBAL)

include(CMakeCheckCompilerFlagCommonPatterns)

function(CHECK_LINKER_FLAG _lang _flag _var)
  get_property (_supported_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if (NOT _lang IN_LIST _supported_languages)
    message (SEND_ERROR "check_linker_flag: ${_lang}: unknown language.")
    return()
  endif()

  include (Check${_lang}SourceCompiles)

  set(CMAKE_REQUIRED_LINK_OPTIONS "${_flag}")

  # Normalize locale during test compilation.
  set(_locale_vars LC_ALL LC_MESSAGES LANG)
  foreach(v IN LISTS _locale_vars)
    set(_locale_vars_saved_${v} "$ENV{${v}}")
    set(ENV{${v}} C)
  endforeach()

  if (_lang MATCHES "^(C|CXX)$")
    set (_source "int main() { return 0; }")
  elseif (_lang STREQUAL "Fortran")
    set (_source "       program test\n       stop\n       end program")
  elseif (_lang MATCHES "^(OBJC|OBJCXX)$")
    set (_source "#ifndef __OBJC__\n#  error \"Not an Objective-C++ compiler\"\n#endif\nint main(void) { return 0; }")
  else()
    message (SEND_ERROR "check_linker_flag: ${_lang}: unsupported language.")
    return()
  endif()
  check_compiler_flag_common_patterns(_common_patterns)

  if (_lang STREQUAL "C")
    check_C_source_compiles("${_source}" ${_var} ${_common_patterns})
  elseif (_lang STREQUAL "CXX")
    check_CXX_source_compiles("${_source}" ${_var} ${_common_patterns})
  elseif (_lang STREQUAL "Fortran")
    check_Fortran_source_compiles("${_source}" ${_var} ${_common_patterns})
  elseif (_lang STREQUAL "OBJC")
    check_OBJC_source_compiles("${_source}" ${_var} ${_common_patterns})
  elseif (_lang STREQUAL "OBJCXX")
    check_OBJCXX_source_compiles("${_source}" ${_var} ${_common_patterns})
  endif ()

  foreach(v IN LISTS _locale_vars)
    set(ENV{${v}} ${_locale_vars_saved_${v}})
  endforeach()
  set(${_var} "${${_var}}" PARENT_SCOPE)
endfunction()