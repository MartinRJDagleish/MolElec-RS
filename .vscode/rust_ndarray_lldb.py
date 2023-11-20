#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# source: https://github.com/hep-rs/boltzmann-solver/blob/master/.vscode/formatter.py
# BIG THANKS TO JP-Ellis for this awesome script!

# Edited by @mrjd to work with SCF_MRJD_RS
# 21.01.2023
# """
# Formatting of Rust NDarray and types from the Boltzmann-Solver.

# To use this, add the following to your workspace settings:

# ```json
# {
#     "lldb.launch.initCommands": [
#         "command script import ${workspaceFolder}/.vscode/formatter.py"
#     ]
# }
# ```
# """

import lldb
import sys
import logging
import weakref
from functools import reduce


log = logging.getLogger(__name__)
module = sys.modules[__name__]

scf_mrjd_rs_category: lldb.SBTypeCategory

synth_by_id = weakref.WeakValueDictionary()


def initialize_category(debugger: lldb.SBDebugger):
    global module, scf_mrjd_rs_category

    scf_mrjd_rs_category = debugger.CreateCategory("SCF_MRJD_RS")
    scf_mrjd_rs_category.SetEnabled(True)

    attach_synthetic_to_type(
        ArrayBaseSynthProvider, r"^ndarray::ArrayBase<.+,.+>$", True
    )
    attach_summary_to_type(
        # particle_summary_provider, "boltzmann_solver::model::particle::Particle"
        wfn_prim_gauss_position_summary_provider, "SCF_MRJD_RS::molecule::wfn::PrimitiveGaussian"
    )


def attach_synthetic_to_type(synth_class, type_name, is_regex=False):
    global module, scf_mrjd_rs_category

    synth: lldb.SBTypeSynthetic = lldb.SBTypeSynthetic.CreateWithClassName(
        __name__ + "." + synth_class.__name__
    )
    synth.SetOptions(lldb.eTypeOptionCascade)
    scf_mrjd_rs_category.AddTypeSynthetic(
        lldb.SBTypeNameSpecifier(type_name, is_regex), synth
    )

    def summary_fn(valobj, dict):
        return get_synth_summary(synth_class, valobj, dict)

    # LLDB accesses summary fn's by name, so we need to create a unique one.
    summary_fn.__name__ = "_get_synth_summary_" + synth_class.__name__
    setattr(module, summary_fn.__name__, summary_fn)
    attach_summary_to_type(summary_fn, type_name, is_regex)


def attach_summary_to_type(summary_fn, type_name, is_regex=False):
    global module, scf_mrjd_rs_category
    log.debug(
        f'attaching summary {summary_fn.__name__} to "{type_name}", is_regex={is_regex}',
    )
    summary = lldb.SBTypeSummary.CreateWithFunctionName(
        __name__ + "." + summary_fn.__name__
    )
    summary.SetOptions(lldb.eTypeOptionCascade)
    scf_mrjd_rs_category.AddTypeSummary(
        lldb.SBTypeNameSpecifier(type_name, is_regex), summary
    )


# 'get_summary' is annoyingly not a part of the standard LLDB synth provider API.
# This trick allows us to share data extraction logic between synth providers and their sibling summary providers.
def get_synth_summary(synth_class, valobj, dict):
    try:
        ns_valobj = valobj.GetNonSyntheticValue()
        synth = synth_by_id.get(ns_valobj.GetID())
        if synth is None:
            synth = synth_class(ns_valobj, dict)
        return str(synth.get_summary())
    except Exception as e:
        log.error("%s", e)
        raise


# Chained GetChildMemberWithName lookups
def gcm(valobj, *chain):
    for name in chain:
        valobj = valobj.GetChildMemberWithName(name)
    return valobj


def read_unique_ptr(valobj: lldb.SBValue):
    # Rust-enabled LLDB using DWARF debug info will strip tuple field prefixes.
    # If LLDB is not Rust-enalbed or if using PDB debug info, they will be underscore-prefixed.
    pointer: lldb.SBValue = valobj.GetChildMemberWithName("pointer")
    child: lldb.SBValue = pointer.GetChildMemberWithName("__0")  # Plain lldb
    if child.IsValid():
        return child
    child = pointer.GetChildMemberWithName("0")  # rust-lldb
    if child.IsValid():
        return child
    return pointer  # pointer no longer contains NonZero since Rust 1.33


def string_from_ptr(pointer, length):
    if length <= 0:
        return ""
    error = lldb.SBError()
    process = pointer.GetProcess()
    data = process.ReadMemory(pointer.GetValueAsUnsigned(), length, error)
    if error.Success():
        return data.decode("utf8", "replace")
    else:
        log.error("ReadMemory error: %s", error.GetCString())


def get_template_params(type_name):
    params = []
    level = 0
    start = 0
    for i, c in enumerate(type_name):
        if c == "<":
            level += 1
            if level == 1:
                start = i + 1
        elif c == ">":
            level -= 1
            if level == 0:
                params.append(type_name[start:i].strip())
        elif c == "," and level == 1:
            params.append(type_name[start:i].strip())
            start = i + 1
    return params


def obj_summary(valobj, unavailable="{...}"):
    summary = valobj.GetSummary()
    if summary is not None:
        return summary
    summary = valobj.GetValue()
    if summary is not None:
        return summary
    return unavailable


def sequence_summary(childern, maxsize=32):
    s = ""
    for child in childern:
        if len(s) > 0:
            s += ", "
        s += obj_summary(child)
        if len(s) > maxsize:
            s += ", ..."
            break
    return s


# ----- Summaries -----


def particle_summary_provider(obj, dict={}):
    return "{name} ({mass} {width})".format(
        name=obj_summary(gcm(obj, "name")).strip('"'),
        mass=obj_summary(gcm(obj, "mass")),
        width=obj_summary(gcm(obj, "width")),
    )

# def wfn_prim_gauss_position_summary_provider(obj, dict={}):
#     return "{name}".format(name=obj_summary(gcm(obj, "name")).strip('"'))


# ---- Synthetic Providers ----


class ArrayBaseSynthProvider:
    def __init__(self, valobj: lldb.SBValue, dict={}):
        self.valobj = valobj
        self.ptr = gcm(self.valobj, "ptr", "pointer")
        self.item_type = self.ptr.GetType().GetPointeeType()
        self.item_size = self.item_type.GetByteSize()

        # The dim and strides requires more processing
        self.dim = [
            v.GetValueAsUnsigned()
            for v in gcm(self.valobj, "dim", "index").get_value_child_list()
        ]
        self.strides = [
            v.GetValueAsUnsigned()
            for v in gcm(self.valobj, "strides", "index").get_value_child_list()
        ]
        self.len = reduce(lambda p, i: p * i, self.dim, 1)

        synth_by_id[valobj.GetID()] = self

    def update(self):
        return False

    def has_children(self):
        return True

    def num_children(self):
        return self.len

    def get_child_at_index(self, index):
        try:
            if not 0 <= index < self.num_children():
                return None
            offset = index * self.item_size
            index_list = []
            for stride in self.strides:
                index_list.append(index // stride)
                index = index % stride
            return self.ptr.CreateChildAtOffset(
                "%s" % index_list, offset, self.item_type
            )
        except Exception as e:
            log.error("%s", e)
            raise

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except Exception as e:
            log.error("%s", e)
            raise

    def get_summary(self):
        return "(%d) array![%s]" % (
            self.len,
            sequence_summary((self.get_child_at_index(i) for i in range(self.len))),
        )


def __lldb_init_module(debugger: lldb.SBDebugger, internal_dict):
    log.info("Initializing")
    initialize_category(debugger)
