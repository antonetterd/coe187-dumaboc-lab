# /*******************************************************************************
# * Simplified core Makefile for fish CNN example on MAX78000
# * Based on MaximSDK generic Makefile, but with a fixed, relative MAXIM_PATH
# * to avoid broken paths like:
# *   C:/MaximSDK/Examples/MAX78000/CNN/fish/C:/MaximSDK/Libraries/libs.mk
# *******************************************************************************

# *******************************************************************************
# Set the target microcontroller and board to compile for.
# *******************************************************************************

ifeq "$(TARGET)" ""
# Default target microcontroller
TARGET := MAX78000
TARGET_UC := MAX78000
TARGET_LC := max78000
else
# "TARGET" has been overridden in the environment or on the command-line.
TARGET_UC := $(subst m,M,$(subst a,A,$(subst x,X,$(TARGET))))
TARGET_LC := $(subst M,m,$(subst A,a,$(subst X,x,$(TARGET))))
endif

# Default board: use Feather board (change to EvKit_V1 if needed)
BOARD ?= FTHR_RevA

# *******************************************************************************
# Locate the MaximSDK  (RELATIVE, NO C:/ PATHS)
# *******************************************************************************

# This project lives in:
#   C:/MaximSDK/Examples/MAX78000/CNN/fish
# So the SDK root (C:/MaximSDK) is four levels up: ../../../../

ifeq "$(MAXIM_PATH)" ""
DEPTH := ../../../../
MAXIM_PATH := $(DEPTH)
$(warning Warning:  MAXIM_PATH is not set!  Using relative SDK root $(MAXIM_PATH))
endif

# Normalize slashes, but KEEP it relative (no abspath to C:/...)
MAXIM_PATH := $(subst \,/,$(MAXIM_PATH))

# Derive Libraries and CMSIS root from MAXIM_PATH
LIBS_DIR   := $(MAXIM_PATH)/Libraries
CMSIS_ROOT := $(LIBS_DIR)/CMSIS

# *******************************************************************************
# Include project Makefile (for overrides: TARGET, BOARD, SRCS, etc.)
# *******************************************************************************

include ./project.mk
$(info Loaded project.mk)

# Re-normalize if project.mk changed MAXIM_PATH
MAXIM_PATH := $(subst \,/,$(MAXIM_PATH))
LIBS_DIR   := $(MAXIM_PATH)/Libraries
CMSIS_ROOT := $(LIBS_DIR)/CMSIS

# One final UC/LC check in case TARGET was changed in project.mk
TARGET_UC := $(subst m,M,$(subst a,A,$(subst x,X,$(TARGET))))
TARGET_LC := $(subst M,m,$(subst A,a,$(subst X,x,$(TARGET))))

export TARGET
export TARGET_UC
export TARGET_LC
export CMSIS_ROOT

# *******************************************************************************
# Set up search paths, and auto-detect all source code on those paths.
# *******************************************************************************

# Where to find source files for this project.
VPATH += .
VPATH += src
VPATH := $(VPATH)

# Where to find header files for this project.
IPATH += .
IPATH += include
IPATH := $(IPATH)

AUTOSEARCH ?= 1
ifeq ($(AUTOSEARCH), 1)
# Auto-detect all C source files on VPATH
SRCS += $(wildcard $(addsuffix /*.c, $(VPATH)))
endif

# Explicitly ensure the generated CNN sources are included
SRCS += main.c
SRCS += cnn.c

# Collapse SRCS before passing them on
SRCS := $(SRCS)

# *******************************************************************************
# Set the output filename
# *******************************************************************************

# The default value creates a file named after the target micro.  Ex: max78000.elf
PROJECT ?= fish

# *******************************************************************************
# Compiler options
# *******************************************************************************

# Select 'GCC' or 'IAR' compiler
ifeq "$(COMPILER)" ""
COMPILER := GCC
endif

# Optimization levels
ifeq "$(MAKECMDGOALS)" "release"
MXC_OPTIMIZE_CFLAGS ?= -O2
DEBUG = 0
endif

ifeq ($(DEBUG),1)
MXC_OPTIMIZE_CFLAGS := -Og
endif

# Default optimization if not debug/release
MXC_OPTIMIZE_CFLAGS ?= -O2

# Base compiler flags
PROJ_CFLAGS += -Wall -Wcast-align
PROJ_CFLAGS += -DMXC_ASSERT_ENABLE
PROJ_CFLAGS += -DARM_MATH_CM4

# Floating point ABI
MFLOAT_ABI ?= softfp
export MFLOAT_ABI

# Linker file (Arm core projects)
ifeq "$(RISCV_CORE)" ""
LINKERFILE ?= $(TARGET_LC).ld
LINKERPATH ?= $(CMSIS_ROOT)/Device/Maxim/$(TARGET_UC)/Source/GCC

# If linkerfile doesn't exist at CMSIS_ROOT, fall back to project root.
ifeq ("$(wildcard $(LINKERPATH)/$(LINKERFILE))","")
LINKERPATH := .
endif

LINKERFILE := $(LINKERPATH)/$(LINKERFILE)
endif

# This path contains system-level initialization files for the target micro.
VPATH += $(CMSIS_ROOT)/Device/Maxim/$(TARGET_UC)/Source

# *******************************************************************************
# Secure Boot Tools (disabled by default)
# *******************************************************************************

SBT ?= 0
ifeq ($(SBT), 1)
MAXIM_SBT_DIR ?= $(MAXIM_PATH)/Tools/SBT
MAXIM_SBT_DIR := $(subst \,/,$(MAXIM_SBT_DIR))
export MAXIM_SBT_DIR
endif # SBT

# *******************************************************************************
# Default goal
# *******************************************************************************

ifeq "$(.DEFAULT_GOAL)" ""
ifeq ($(SBT),1)
override .DEFAULT_GOAL := sla
else
override .DEFAULT_GOAL := all
endif
endif

ifeq "$(MAKECMDGOALS)" ""
MAKECMDGOALS := $(.DEFAULT_GOAL)
endif

# *******************************************************************************
# Include SBT config (if enabled)
# *******************************************************************************

ifeq ($(SBT),1)
include $(MAXIM_PATH)/Tools/SBT/SBT-config.mk
endif

# *******************************************************************************
# Libraries
# *******************************************************************************

# Toggle and configure MaximSDK libraries (Board, PeriphDrivers, etc.)
include $(LIBS_DIR)/libs.mk

# *******************************************************************************
# Rules: include target/compiler-specific build rules
# *******************************************************************************

include $(CMSIS_ROOT)/Device/Maxim/$(TARGET_UC)/Source/$(COMPILER)/$(TARGET_LC).mk

# Extend rules with SBTs if enabled
ifeq ($(SBT), 1)
include $(MAXIM_PATH)/Tools/SBT/SBT-rules.mk
endif

# Ensure MAKECMDGOALS is set
ifeq "$(MAKECMDGOALS)" ""
MAKECMDGOALS := $(.DEFAULT_GOAL)
endif

# No custom 'all' / 'clean' rules here; we rely on the included rules.
# This avoids any extra tab/recipe pitfalls.
